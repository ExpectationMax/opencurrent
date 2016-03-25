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
#include "ocustorage/grid3dboundary.h"
#include "ocuequation/sol_project3d.h"

using namespace ocu;

#ifndef M_PI
#define M_PI (3.1415926535)
#endif

DECLARE_UNITTEST_BEGIN(ProjectTest);

void run_test(int nx, int ny, int nz, float hx, float hy, float hz, 
              BoundaryCondition xpos, BoundaryCondition xneg, BoundaryCondition ypos, BoundaryCondition yneg, BoundaryCondition zpos, BoundaryCondition zneg)
{

  std::vector<Grid3DDimension> dimensions(3);
  dimensions[0].init(nx+1,ny,nz,1,1,1);
  dimensions[1].init(nx,ny+1,nz,1,1,1);
  dimensions[2].init(nx,ny,nz+1,1,1,1);  
  Grid3DDimension::pad_for_congruence(dimensions);

  Grid3DDeviceF u, v, w;
  Grid3DHostF hu, hv, hw;

  u.init_congruent(dimensions[0]);
  v.init_congruent(dimensions[1]);
  w.init_congruent(dimensions[2]);
  hu.init_congruent(dimensions[0],true);
  hv.init_congruent(dimensions[1],true);
  hw.init_congruent(dimensions[2],true);
/*
  Grid3DDeviceF u, v, w;
  Grid3DHostF hu, hv, hw;

  u.init(nx+1, ny, nz, 1, 1, 1, 0, 1, 1);
  v.init(nx, ny+1, nz, 1, 1, 1, 1, 0, 1);
  w.init(nx, ny, nz+1, 1, 1, 1, 1, 1, 0);
  hu.init(nx+1, ny, nz, 1, 1, 1, true, 0, 1, 1);
  hv.init(nx, ny+1, nz, 1, 1, 1, true, 1, 0, 1);
  hw.init(nx, ny, nz+1, 1, 1, 1, true, 1, 1, 0);
*/
  int i,j,k;
  for (i=0; i < nx; i++) {
    for (j=0; j < ny; j++) {
      for (k=0; k < nz; k++) {
        hu.at(i,j,k) = sin(((float)k) / nz * 4 * M_PI) + sin(((float)j) / ny * 4 * M_PI);
        hv.at(i,j,k) = sin(((float)k) / nz * 8 * M_PI) + sin(((float)i) / nx * 6 * M_PI);
        hw.at(i,j,k) = cos(((float)j) / ny * 8 * M_PI) + sin(((float)k) / nz * 6 * M_PI);
      }
    }
  }


  
  UNITTEST_ASSERT_TRUE(u.copy_all_data(hu));
  UNITTEST_ASSERT_TRUE(v.copy_all_data(hv));
  UNITTEST_ASSERT_TRUE(w.copy_all_data(hw));

  Sol_ProjectDivergence3DDeviceF project;
  project.bc.xpos = xpos;
  project.bc.xneg = xneg;
  project.bc.ypos = ypos;
  project.bc.yneg = yneg;
  project.bc.zpos = zpos;
  project.bc.zneg = zneg;

  apply_3d_mac_boundary_conditions_level1(u,v,w,project.bc, hx, hy, hz);

  UNITTEST_ASSERT_TRUE(project.initialize_storage(nx, ny, nz, hx, hy, hz, &u, &v, &w));

  Grid3DHostF h_div;


  float max_div_dev;
  UNITTEST_ASSERT_TRUE(project.solve_divergence_only());
  UNITTEST_ASSERT_TRUE(project.divergence.reduce_maxabs(max_div_dev));
  printf("before max divergence = %f\n", max_div_dev);


  // only do this test if everything is periodic
  if (xpos.type == BC_PERIODIC && xneg.type == BC_PERIODIC && ypos.type == BC_PERIODIC && yneg.type == BC_PERIODIC && zpos.type == BC_PERIODIC && zneg.type == BC_PERIODIC) {
    float max_div_host = 0;
    for (i=0; i < nx; i++) {
      for (j=0; j < ny; j++) {
        for (k=0; k < nz; k++) {
          float div = hu.at((i+1)%nx,j,k) - hu.at(i,j,k) + 
            hv.at(i,(j+1)%ny,k) - hv.at(i,j,k) + 
            hw.at(i,j,(k+1)%nz) - hw.at(i,j,k);
            if (fabs(div) > max_div_host) max_div_host = fabs(div);
        }
      }
    }
    UNITTEST_ASSERT_EQUAL_FLOAT(max_div_dev, max_div_host, 1e-6);
  }


  UNITTEST_ASSERT_TRUE(project.solve(2e-4));

  UNITTEST_ASSERT_TRUE(project.solve_divergence_only());
  float max_div_after;
  UNITTEST_ASSERT_TRUE(project.divergence.reduce_maxabs(max_div_after));
  printf("after max divergence = %f\n", max_div_after);

  Grid3DHostF hd;
  hd.init(project.divergence.nx(), project.divergence.ny(), project.divergence.nz(), project.divergence.gx(), project.divergence.gy(), project.divergence.gz(), true,
    project.divergence.paddingx(), project.divergence.paddingy(), project.divergence.paddingz());

  hd.copy_all_data(project.divergence);
  for (i=0; i < nx; i++) 
    for (j=0; j < ny; j++) 
      for (k=0; k < nz; k++) {
        if (fabs(hd.at(i,j,k)) > .001) {
          printf("%d %d %d: %f\n", i,j,k, hd.at(i,j,k));
        }
      }



  UNITTEST_ASSERT_EQUAL_FLOAT(max_div_after, 0, 2e-4);

}


void run() {
  int nx = 128;
  int ny = 64;
  int nz = 64;

  float hx = .5;
  float hy = .5;
  float hz = 1;

  BoundaryCondition xpos, xneg, ypos, yneg, zpos, zneg;
  xpos.type = BC_PERIODIC;
  xneg.type = BC_PERIODIC;
  ypos.type = BC_PERIODIC;
  yneg.type = BC_PERIODIC;
  zpos.type = BC_PERIODIC;
  zneg.type = BC_PERIODIC;

  run_test(nx, ny, nz, hx, hy, hz, xpos, xneg, ypos, yneg, zpos, zneg);

  xpos.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  xpos.value = .2;
  xneg.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  xneg.value = .2;
  ypos.type = BC_PERIODIC;
  yneg.type = BC_PERIODIC;
  zpos.type = BC_PERIODIC;
  zneg.type = BC_PERIODIC;

  run_test(nx, ny, nz, hx, hy, hz, xpos, xneg, ypos, yneg, zpos, zneg);
  xpos.type = BC_PERIODIC;
  xneg.type = BC_PERIODIC;
  ypos.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  ypos.value = 0;
  yneg.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  yneg.value = 0;
  zpos.type = BC_PERIODIC;
  zneg.type = BC_PERIODIC;

  run_test(nx, ny, nz, hx, hy, hz, xpos, xneg, ypos, yneg, zpos, zneg);


  xpos.type = BC_PERIODIC;
  xneg.type = BC_PERIODIC;
  ypos.type = BC_PERIODIC;
  yneg.type = BC_PERIODIC;
  zpos.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  zpos.value = 0;
  zneg.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  zneg.value = 0;

  run_test(nx, ny, nz, hx, hy, hz, xpos, xneg, ypos, yneg, zpos, zneg);

  xpos.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  xpos.value = 0;
  xneg.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  xneg.value = 0;
  ypos.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  ypos.value = 0;
  yneg.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  yneg.value = 0;
  zpos.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  zpos.value = 0;
  zneg.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  zneg.value = 0;

  run_test(nx, ny, nz, hx, hy, hz, xpos, xneg, ypos, yneg, zpos, zneg);

  nx = 128;
  ny = 128;
  nz = 96;

  hx = .5;
  hy = .5;
  hz = .5;
  run_test(nx, ny, nz, hx, hy, hz, xpos, xneg, ypos, yneg, zpos, zneg);
}

DECLARE_UNITTEST_END(ProjectTest);
