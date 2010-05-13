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
#include "ocuequation/eqn_diffusion3d.h"
#include "ocustorage/coarray.h"
#include "ocustorage/gridnetcdf.h"

using namespace ocu;


DECLARE_UNITTEST_MULTIGPU_DOUBLE_BEGIN(Diffusion3DMultiTest);

void run()
{
  int tid = ThreadManager::this_image();

  BoundaryCondition example;
  example.type = BC_PERIODIC;

  int nx = 64;
  int ny = 128;
  int nz = 128;

  Eqn_Diffusion3DCoParams<double> params;
  params.nx = nx;
  params.ny = ny;
  params.nz = nz;
  params.hx = 1;
  params.hy = 1;
  params.hz = 1;
  params.bc = BoundaryConditionSet(example);
  params.diffusion_coefficient = 1;
  params.initial_values.init(nx, ny, nz, 1, 1, 1);
  params.initial_values.clear(tid);


  Eqn_Diffusion3DCo<double> solver("solver");
  UNITTEST_ASSERT_TRUE(solver.set_parameters(params));
  double total_before;
  solver.density().co_reduce_sum(total_before);

  CPUTimer timer;
  timer.start();
  for (int T=0; T < 10; T++) {
    UNITTEST_ASSERT_TRUE(solver.advance(1));
  }
  ThreadManager::barrier_and_fence();
  timer.stop();
  printf("Elapsed: %f\n", timer.elapsed_sec());

  double total_after;
  solver.density().co_reduce_sum(total_after);
  printf("before = %f, after = %f\n", total_before, total_after);

  Grid3DHostD d_diff;
  d_diff.init_congruent(solver.density());
  d_diff.copy_all_data(solver.density());
/*
  if (tid == 0) {
  NetCDFGrid3DWriter writer;
  char name[1024];
  sprintf(name, "diff3d_%d.nc", tid);
  writer.open(name, nx, ny, nz, 1,1,1);
  writer.define_variable("val", NC_DOUBLE, GS_CENTER_POINT);
  writer.add_data("val", d_diff);
  writer.close();
  }
*/
}

DECLARE_UNITTEST_DOUBLE_END(Diffusion3DMultiTest);

