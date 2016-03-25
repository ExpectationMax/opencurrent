/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "tests/testframework.h"
#include "ocustorage/grid3dsample.h"
#include "ocustorage/grid3dboundary.h"

#include <cmath>


using namespace ocu;

DECLARE_UNITTEST_BEGIN(SamplingTest);


void allocate_mac(Grid3DHostF &hu, Grid3DHostF &hv, Grid3DHostF &hw, Grid3DDeviceF &u,Grid3DDeviceF &v,Grid3DDeviceF &w, int nx, int ny, int nz)
{
  hu.init(nx+1, ny, nz, 1, 1, 1, true, 0, 1, 1);
  hv.init(nx, ny+1, nz, 1, 1, 1, true, 1, 0, 1);
  hw.init(nx, ny, nz+1, 1, 1, 1, true, 1, 1, 0);
  u.init_congruent(hu);
  v.init_congruent(hv);
  w.init_congruent(hw);

}

void allocate_particles(Grid1DHostF &hposx, Grid1DHostF &hposy, Grid1DHostF &hposz, Grid1DHostF &hvx, Grid1DHostF &hvy, Grid1DHostF &hvz,
  Grid1DDeviceF &posx, Grid1DDeviceF &posy, Grid1DDeviceF &posz, Grid1DDeviceF &vx, Grid1DDeviceF &vy, Grid1DDeviceF &vz, 
    int nparts, float xsize, float ysize, float zsize, bool periodic)
{
  // if periodic is selected, nparts must be even:
  UNITTEST_ASSERT_TRUE(periodic ? (nparts%2 == 0) : true);

  hposx.init(nparts,0);
  hposy.init(nparts,0);
  hposz.init(nparts,0);
  hvx.init(nparts,0);
  hvy.init(nparts,0);
  hvz.init(nparts,0);

  posx.init(nparts,0);
  posy.init(nparts,0);
  posz.init(nparts,0);
  vx.init(nparts,0);
  vy.init(nparts,0);
  vz.init(nparts,0);
  
  for (int p=0; p < nparts; p++) {
    hposx.at(p) = ((((double)rand()) / RAND_MAX) * 2 * xsize) - xsize;
    hposy.at(p) = ((((double)rand()) / RAND_MAX) * 2 * ysize) - ysize;
    hposz.at(p) = ((((double)rand()) / RAND_MAX) * 2 * zsize) - zsize;
    if (periodic) {
      hposx.at(p+1) = hposx.at(p) + xsize * (((int)(rand() % 10)) - 5);
      hposy.at(p+1) = hposy.at(p) + ysize * (((int)(rand() % 10)) - 5);
      hposz.at(p+1) = hposz.at(p) + zsize * (((int)(rand() % 10)) - 5);
      p++;
    }
  }

  UNITTEST_ASSERT_TRUE(posx.copy_all_data(hposx));
  UNITTEST_ASSERT_TRUE(posy.copy_all_data(hposy));
  UNITTEST_ASSERT_TRUE(posz.copy_all_data(hposz));
}

void run_constant_test()
{
  int nx = 10;
  int ny = 20;
  int nz = 15;

  float hx = .5;
  float hy = .2;
  float hz = .7;

  int nparts = 100000;

  Grid3DHostF hu, hv, hw;
  Grid3DDeviceF u,v,w;

  allocate_mac(hu, hv, hw,u,v,w, nx, ny, nz);

  hu.clear(2.4);
  hv.clear(2.3);
  hw.clear(2.1);

  UNITTEST_ASSERT_TRUE(u.copy_all_data(hu));
  UNITTEST_ASSERT_TRUE(v.copy_all_data(hv));
  UNITTEST_ASSERT_TRUE(w.copy_all_data(hw));

  int p;
  Grid1DHostF hposx, hposy, hposz, hvx, hvy, hvz;
  Grid1DDeviceF posx, posy, posz, vx, vy, vz;

  allocate_particles(hposx, hposy, hposz, hvx, hvy, hvz, posx, posy, posz, vx, vy, vz, nparts, nx*hx, ny*hy, nz*hz, false);

  BoundaryCondition periodic;
  periodic.type = BC_PERIODIC;
  BoundaryConditionSet bc(periodic);

  UNITTEST_ASSERT_TRUE(apply_3d_mac_boundary_conditions_level1(u,v,w, bc, hx, hy, hz));
  UNITTEST_ASSERT_TRUE(sample_points_mac_grid_3d(vx, vy, vz, posx, posy, posz, u, v, w, bc, hx, hy, hz));

  UNITTEST_ASSERT_TRUE(hvx.copy_all_data(vx));
  UNITTEST_ASSERT_TRUE(hvy.copy_all_data(vy));
  UNITTEST_ASSERT_TRUE(hvz.copy_all_data(vz));
  
  for (p=0; p < nparts; p++) {
    UNITTEST_ASSERT_EQUAL_FLOAT(hvx.at(p), 2.4, 1e-6f);
    UNITTEST_ASSERT_EQUAL_FLOAT(hvy.at(p), 2.3, 1e-6f);
    UNITTEST_ASSERT_EQUAL_FLOAT(hvz.at(p), 2.1, 1e-6f);
  }
}

void run_peridoic_test()
{
  int nx = 10;
  int ny = 10;
  int nz = 10;

  float hx = 1;
  float hy = 1;
  float hz = 1;

  int nparts = 100000;

  Grid3DHostF hu, hv, hw;
  Grid3DDeviceF u,v,w;

  allocate_mac(hu, hv, hw,u,v,w, nx, ny, nz);


  int i,j,k;

  // init to something random here
  for (i=0; i < nx; i++)
    for (j=0; j < ny; j++)
      for (k=0; k < nz; k++) {
        //hu.at(i,j,k) = ((double)rand()) / RAND_MAX;
        //hv.at(i,j,k) = ((double)rand()) / RAND_MAX;
        //hw.at(i,j,k) = ((double)rand()) / RAND_MAX;
        hu.at(i,j,k) = ((float)i)/nx;
        hv.at(i,j,k) = ((float)j)/ny;
        hw.at(i,j,k) = ((float)k)/nz;
      }

  UNITTEST_ASSERT_TRUE(u.copy_all_data(hu));
  UNITTEST_ASSERT_TRUE(v.copy_all_data(hv));
  UNITTEST_ASSERT_TRUE(w.copy_all_data(hw));

  int p;
  Grid1DHostF hposx, hposy, hposz, hvx, hvy, hvz;
  Grid1DDeviceF posx, posy, posz, vx, vy, vz;

  allocate_particles(hposx, hposy, hposz, hvx, hvy, hvz, posx, posy, posz, vx, vy, vz, nparts, nx*hx, ny*hy, nz*hz, true);

  BoundaryCondition periodic;
  periodic.type = BC_PERIODIC;
  BoundaryConditionSet bc(periodic);

  UNITTEST_ASSERT_TRUE(apply_3d_mac_boundary_conditions_level1(u,v,w, bc, hx, hy, hz));
  UNITTEST_ASSERT_TRUE(sample_points_mac_grid_3d(vx, vy, vz, posx, posy, posz, u, v, w, bc, hx, hy, hz));

  UNITTEST_ASSERT_TRUE(hvx.copy_all_data(vx));
  UNITTEST_ASSERT_TRUE(hvy.copy_all_data(vy));
  UNITTEST_ASSERT_TRUE(hvz.copy_all_data(vz));
  
  for (p=0; p < nparts; p+=2) {
    //printf("(%f, %f, %f) vs (%f %f %f)\n", hposx.at(p), hposy.at(p), hposz.at(p), hposx.at(p+1), hposy.at(p+1), hposz.at(p+1));
    if (fabs(hvx.at(p) - hvx.at(p+1)) > 1e-5f) {
        float x1 = hposx.at(p);
        float y1 = hposy.at(p);
        float z1 = hposz.at(p);

        float x2 = hposx.at(p+1);
        float y2 = hposy.at(p+1);
        float z2 = hposz.at(p+1);
        printf("(%f, %f, %f) vs (%f %f %f)\n", x1,y1,z1,x2,y2,z2);
        printf("u1 point is %d %d %d\n", (int)floorf(x1), (int)floorf(y1-.5f), (int)floorf(z1-.5f));
        printf("u2 point is %d %d %d\n", (int)floorf(x2), (int)floorf(y2-.5f), (int)floorf(z2-.5f));


        x1 = fmodf(x1, 10);
        if(x1 < 0) x1 += nx;  
        x2 = fmodf(x2, 10);
        if(x2 < 0) x2 += nx;  
        y1 = fmodf(y1, 10);
        if(y1 < 0) y1 += ny;  
        y2 = fmodf(y2, 10);
        if(y2 < 0) y2 += ny;  
        z1 = fmodf(z1, 10);
        if(z1 < 0) z1 += nz;  
        z2 = fmodf(z2, 10);
        if(z2 < 0) z2 += nz;  


        printf("u1 point after %f %f %f -> is %d %d %d\n", x1, y1, z1, (int)floorf(x1), (int)floorf(y1-.5f), (int)floorf(z1-.5f));
        printf("u2 point after %f %f %f -> is %d %d %d\n", x2, y2, z2, (int)floorf(x2), (int)floorf(y2-.5f), (int)floorf(z2-.5f));
    }

    UNITTEST_ASSERT_EQUAL_FLOAT(hvx.at(p), hvx.at(p+1), 1e-5f);
    UNITTEST_ASSERT_EQUAL_FLOAT(hvy.at(p), hvy.at(p+1), 1e-5f);
    UNITTEST_ASSERT_EQUAL_FLOAT(hvz.at(p), hvz.at(p+1), 1e-5f);
  }
}

void run_gradient_test()
{
  int nx = 10;
  int ny = 20;
  int nz = 15;

  float hx = 1;
  float hy = 1;
  float hz = 1;
  
  int nparts = 100000;

  Grid3DHostF hu, hv, hw;
  Grid3DDeviceF u,v,w;

  allocate_mac(hu, hv, hw,u,v,w, nx, ny, nz);


  int i,j,k;
  for (i=0; i < nx; i++)
    for (j=0; j < ny; j++)
      for (k=0; k < nz; k++) {
        hu.at(i,j,k) = ((float)i)/nx;
        hv.at(i,j,k) = ((float)j)/ny;
        hw.at(i,j,k) = ((float)k)/nz;
      }

  u.copy_all_data(hu);
  v.copy_all_data(hv);
  w.copy_all_data(hw);



  int p;
  Grid1DHostF hposx, hposy, hposz, hvx, hvy, hvz;
  Grid1DDeviceF posx, posy, posz, vx, vy, vz;

  allocate_particles(hposx, hposy, hposz, hvx, hvy, hvz, posx, posy, posz, vx, vy, vz, nparts, nx*hx, ny*hy, nz*hz, false);

  BoundaryConditionSet bc;
  bc.xpos.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  bc.xpos.value = 1.0f;
  bc.xneg.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  bc.xneg.value = 0.0f;

  bc.ypos.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  bc.ypos.value = 1.0f;
  bc.yneg.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  bc.yneg.value = 0.0f;

  bc.zpos.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  bc.zpos.value = 1.0f;
  bc.zneg.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  bc.zneg.value = 0.0f;

  UNITTEST_ASSERT_TRUE(apply_3d_mac_boundary_conditions_level1(u,v,w, bc, hx, hy, hz));
  UNITTEST_ASSERT_TRUE(sample_points_mac_grid_3d(vx, vy, vz, posx, posy, posz, u, v, w, bc, hx, hy, hz));

  hvx.copy_all_data(vx);
  hvy.copy_all_data(vy);
  hvz.copy_all_data(vz);
  
  for (p=0; p < nparts; p++) {
    UNITTEST_ASSERT_EQUAL_FLOAT(hvx.at(p), std::min(1.0f, std::max(0.0f, hposx.at(p) / nx)), 1e-6f);
    UNITTEST_ASSERT_EQUAL_FLOAT(hvy.at(p), std::min(1.0f, std::max(0.0f, hposy.at(p) / ny)), 1e-6f);
    UNITTEST_ASSERT_EQUAL_FLOAT(hvz.at(p), std::min(1.0f, std::max(0.0f, hposz.at(p) / nz)), 1e-6f);
  }

}

void run()
{
  run_constant_test();
  run_peridoic_test();
  run_gradient_test();
}

DECLARE_UNITTEST_END(SamplingTest);

