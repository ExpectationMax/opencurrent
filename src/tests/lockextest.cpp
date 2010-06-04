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

#include <cmath>
#include <algorithm>


#include "tests/testframework.h"
#include "ocuequation/eqn_incompressns3d.h"
#include "ocustorage/grid3d.h"

using namespace ocu;


DECLARE_UNITTEST_BEGIN(LockExTest);


void run()
{
  int nx = 128;
  int ny = 128;
  int nz = 128;

  Eqn_IncompressibleNS3DParams<float> params;
  Eqn_IncompressibleNS3D<float> eqn;

  params.init_grids(nx, ny, nz);
  params.hx = .4;
  params.hy = .7;
  params.hz = 1;
  params.max_divergence = 1e-4;
  params.viscosity = 0;
  params.thermal_diffusion = 0;
  params.gravity = -1;
//params.advection_scheme = IT_FIRST_ORDER_UPWIND;
  params.advection_scheme = IT_SECOND_ORDER_CENTERED;
  params.time_step = TS_ADAMS_BASHFORD2;
  params.cfl_factor = .7;

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

  UNITTEST_ASSERT_TRUE(eqn.set_parameters(params));

  double dt = .1;
  double this_dt = .1;

  // verify no nans
  float nancheck;
  UNITTEST_ASSERT_TRUE(eqn.get_temperature().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_u().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_v().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_w().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);

  for (float t=0; t < 10; t += this_dt) {
    this_dt = std::min(eqn.get_max_stable_timestep(), dt);
    UNITTEST_ASSERT_TRUE(eqn.advance_one_step(this_dt));
  }

  // verify no nans
  UNITTEST_ASSERT_TRUE(eqn.get_temperature().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_u().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_v().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_w().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);


}


DECLARE_UNITTEST_END(LockExTest);


DECLARE_UNITTEST_DOUBLE_BEGIN(LockExDoubleTest);

void run()
{
  int nx = 128;
  int ny = 128;
  int nz = 128;

  Eqn_IncompressibleNS3DParams<double> params;
  Eqn_IncompressibleNS3D<double> eqn;

  params.init_grids(nx, ny, nz);
  params.hx = .4;
  params.hy = .7;
  params.hz = 1;
  params.max_divergence = 1e-6;
  params.viscosity = .1;
  params.thermal_diffusion = .1;
  params.gravity = -1;
  params.advection_scheme = IT_FIRST_ORDER_UPWIND;
//  params.advection_scheme = IT_SECOND_ORDER_CENTERED;
//  params.time_step = TS_ADAMS_BASHFORD2;
//  params.cfl_factor = .7;

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

  UNITTEST_ASSERT_TRUE(eqn.set_parameters(params));

  double nancheck;
  UNITTEST_ASSERT_TRUE(eqn.get_temperature().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_u().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_v().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_w().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);

  double dt = .1;
  double this_dt = .1;

  for (float t=0; t < 10; t += this_dt) {
    this_dt = std::min(eqn.get_max_stable_timestep(), dt);
    UNITTEST_ASSERT_TRUE(eqn.advance_one_step(this_dt));
  }

  // verify no nans
  UNITTEST_ASSERT_TRUE(eqn.get_temperature().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_u().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_v().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);
  UNITTEST_ASSERT_TRUE(eqn.get_w().reduce_checknan(nancheck));
  UNITTEST_ASSERT_FINITE(nancheck);

  // check for symmetry

  Grid3DHostD h_temp;
  h_temp.init_congruent(eqn.get_temperature());
  h_temp.copy_all_data(eqn.get_temperature());

  // tolerances are fairly high because error accumulates over several time steps above
  for (i=0; i < nx; i++) {
    for (j=0; j < ny; j++) {
      double val = h_temp.at(i,j,0);
      for (k=0; k < nz; k++) {
        // should be uniform in z
        UNITTEST_ASSERT_EQUAL_DOUBLE(val, h_temp.at(i,j,k), 1e-4);
      }
      // anti-symmetric around center point in x,y
      int ri = nx -i-1;
      int rj = ny -j-1;
      UNITTEST_ASSERT_EQUAL_DOUBLE(val, -h_temp.at(ri,rj,k), 1e-4);

    }
  }
}

DECLARE_UNITTEST_DOUBLE_END(LockExDoubleTest);

