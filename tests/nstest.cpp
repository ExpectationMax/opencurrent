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
#include "ocuequation/eqn_incompressns3d.h"

#include "ocustorage/grid1d.h"
#include "ocustorage/grid3dboundary.h"
#include "ocuequation/eqn_incompressns3d.h"
#include "ocuutil/timer.h"

using namespace ocu;

DECLARE_UNITTEST_BEGIN(NSTest);


void allocate_particles(Grid1DHostF &hposx, Grid1DHostF &hposy, Grid1DHostF &hposz, Grid1DHostF &hvx, Grid1DHostF &hvy, Grid1DHostF &hvz,
  Grid1DDeviceF &posx, Grid1DDeviceF &posy, Grid1DDeviceF &posz, Grid1DDeviceF &vx, Grid1DDeviceF &vy, Grid1DDeviceF &vz, 
    int nparts, float xsize, float ysize, float zsize)
{
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
    hposx.at(p) = ((((double)rand()) / RAND_MAX) * xsize);
    hposy.at(p) = ((((double)rand()) / RAND_MAX) * ysize);
    hposz.at(p) = ((((double)rand()) / RAND_MAX) * zsize);
  }

  posx.copy_all_data(hposx);
  posy.copy_all_data(hposy);
  posz.copy_all_data(hposz);
}


void run()
{
  Eqn_IncompressibleNS3DParamsF params;
  Eqn_IncompressibleNS3DF eqn;

  int nx = 128;
  int ny = 128;
  int nz = 128;

  params.init_grids(nx, ny, nz);
  params.hx = 1;
  params.hy = 1;
  params.hz = 1;

  BoundaryCondition closed;
  closed.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  params.flow_bc = BoundaryConditionSet(closed);

  BoundaryCondition neumann;
  neumann.type = BC_NEUMANN;
  params.temp_bc = BoundaryConditionSet(neumann);

  int i,j,k;
  params.init_u.clear_zero();
  params.init_v.clear_zero();
  params.init_w.clear_zero();
  
  for (i=0; i < nx; i++)
    for (j=0; j < ny; j++)
      for (k=0; k < nz; k++) {
        params.init_temp.at(i,j,k) = (i < nx / 2) ? -1 : 1;
      }
  params.max_divergence = 1e-4;

  UNITTEST_ASSERT_TRUE(eqn.set_parameters(params));

  Grid1DHostF hposx, hposy, hposz;
  Grid1DHostF hvx, hvy, hvz;
  Grid1DDeviceF posx, posy, posz;
  Grid1DDeviceF vx, vy, vz;

  int nparts = 1000;
  allocate_particles(hposx, hposy, hposz, hvx, hvy, hvz, posx, posy, posz, vx, vy, vz, nparts, nx, ny, nz);

  double dt = .1;

  //FILE *file = fopen("fluidsim.prt", "wb");

  CPUTimer timer;

  timer.start();
  for (int t=0; t < 10; t++) {
    printf("Frame %d\n", t);
    UNITTEST_ASSERT_TRUE(eqn.advance(dt));
    // trace points
    /*
    sample_points_mac_grid_3d(vx, vy, vz, posx, posy, posz, eqn.get_u(), eqn.get_v(), eqn.get_w(), params.flow_bc, 1,1,1);

    hvx.copy_all_data(vx); hvy.copy_all_data(vy); hvz.copy_all_data(vz);
    
    fwrite(&nparts, sizeof(int), 1, file);
    for (int p=0; p < hvx.nx(); p++) {

      // forward Euler
      hposx.at(p) += hvx.at(p) * dt;
      hposy.at(p) += hvy.at(p) * dt;
      hposz.at(p) += hvz.at(p) * dt;

      // add to file
      fwrite(&hposx.at(p), sizeof(float), 1, file);
      fwrite(&hposy.at(p), sizeof(float), 1, file);
      fwrite(&hposz.at(p), sizeof(float), 1, file);
    }
    // copy positions back to device
    posx.copy_all_data(hposx); posy.copy_all_data(hposy); posz.copy_all_data(hposz);
    */
  }

  timer.stop();
  printf("Elapsed: %f, or %f fps\n", timer.elapsed_sec(), 100 / timer.elapsed_sec());

  //fclose(file);
}

DECLARE_UNITTEST_END(NSTest);
