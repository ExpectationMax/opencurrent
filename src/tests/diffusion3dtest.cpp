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
#include "ocuutil/boundary_condition.h"
#include "ocuequation/eqn_diffusion3d.h"
#include "ocuutil/float_routines.h"

using namespace ocu;

DECLARE_UNITTEST_BEGIN(Diffusion3DTest);


void run_test(int nx, int ny, int nz, float hx, float hy, float hz, BoundaryConditionSet bc, bool no_border_contrib)
{
  Eqn_Diffusion3DParams<float> params;
  params.nx = nx;
  params.ny = ny;
  params.nz = nz;
  params.hx = hx;
  params.hy = hy;
  params.hz = hz;
  params.bc = bc;
  params.diffusion_coefficient = 5;
  params.initial_values.init(nx, ny, nz, 1, 1, 1);

  float midpt_x = (nx-1) * hx * .5;
  float midpt_y = (ny-1) * hy * .5;
  float midpt_z = (nz-1) * hz * .5;

  float mindim = min3(nx*hx, ny*hy, nz*hz);

  int i,j,k;
  for (i=0; i < nx; i++)
    for (j=0; j < ny; j++)
      for (k=0; k < nz; k++) {
        float px = i * hx;
        float py = j * hy;
        float pz = k * hz;

        float rad = sqrt((px - midpt_x)*(px - midpt_x) + (py - midpt_y)*(py - midpt_y) + (pz - midpt_z)*(pz - midpt_z));
        if (rad < (mindim/4)) {
          params.initial_values.at(i,j,k) = 1;
        }
        else {
          params.initial_values.at(i,j,k) = 0;
        }
      }


  // set up a symmetric initial shape
  float variation_before;
  float integral_before;

  params.initial_values.reduce_sqrsum(variation_before);
  params.initial_values.reduce_sum(integral_before);

  Eqn_Diffusion3D<float> eqn;
  UNITTEST_ASSERT_TRUE(eqn.set_parameters(params));

  // diffuse it.
  UNITTEST_ASSERT_TRUE(eqn.advance(.1));
  float check_nan;
  eqn.density().reduce_checknan(check_nan);
  UNITTEST_ASSERT_FINITE(check_nan);

  if (no_border_contrib) {
    // verify TVD
    // verify conservative
    float variation_after;
    float integral_after;
    eqn.density().reduce_sqrsum(variation_after);
    eqn.density().reduce_sum(integral_after);
    UNITTEST_ASSERT_EQUAL_DOUBLE(integral_after, integral_before, .01f);
    UNITTEST_ASSERT_TRUE(variation_after <= variation_before);
  }

  Grid3DHost<float> h_grid;
  h_grid.init_congruent(eqn.density());
  h_grid.copy_all_data(eqn.density());

  // verify symmetric
  for (i=0; i < nx; i++)
    for (j=0; j < ny; j++)
      for (k=0; k < nz; k++) {
        // get reflected point
        int ri = nx-1-i;
        int rj = ny-1-j;
        int rk = nz-1-k;
        float val = h_grid.at(i,j,k);
        float rval = h_grid.at(ri, rj, rk);
        UNITTEST_ASSERT_EQUAL_DOUBLE(val, rval, 0);
      }
}



void run_all_bcs(int nx, int ny, int nz, float hx, float hy, float hz) {
  BoundaryCondition example;
  BoundaryConditionSet bc;

  example.type = BC_PERIODIC;
  bc = BoundaryCondition(example);
  run_test(nx, ny, nz, hx, hy, hz, bc, true);

  example.type = BC_NEUMANN;
  example.value = 0;
  bc = BoundaryCondition(example);
  run_test(nx, ny, nz, hx, hy, hz, bc, true);

  example.type = BC_NEUMANN;
  example.value = 1;
  bc = BoundaryCondition(example);
  // change signs for positive sides so it will be symmetric
  bc.xpos.value = -1;
  bc.ypos.value = -1;
  bc.zpos.value = -1;
  run_test(nx, ny, nz, hx, hy, hz, bc, false);

  example.type = BC_DIRICHELET;
  example.value = 0;
  bc = BoundaryCondition(example);
  run_test(nx, ny, nz, hx, hy, hz, bc, true);

  example.type = BC_DIRICHELET;
  example.value = 1;
  bc = BoundaryCondition(example);
  run_test(nx, ny, nz, hx, hy, hz, bc, false);
}

void run() {
  run_all_bcs(128, 128, 128, 1, 1, 1);
  run_all_bcs(128, 128, 128, .5, .3, 1);
  run_all_bcs(60, 128, 128, .5, .3, 1);
  run_all_bcs(128, 60, 128, .5, .7, 1);
  run_all_bcs(128, 128, 60, .5, .3, 1);
  run_all_bcs(34, 57, 92, 1, 1, 1);
  run_all_bcs(39, 92, 57, 1, 1, 1);
}

DECLARE_UNITTEST_END(Diffusion3DTest);

