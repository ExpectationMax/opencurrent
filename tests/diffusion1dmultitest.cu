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
#include "ocuequation/eqn_diffusion1d.h"
#include "ocustorage/coarray.h"
#include "ocustorage/grid1d.h"

using namespace ocu;


DECLARE_UNITTEST_MULTIGPU_BEGIN(Diffusion1DMultiTest);

void run()
{
  int tid = ThreadManager::this_image();

  int N = 1<<8;

  Eqn_Diffusion1DParams params;
  params.initial_values.init(N, 1);
  params.initial_values.clear((tid == 0) ? 1.0f : 0.0f);
  params.nx = N;
  params.diffusion_coefficient = 1;
  params.h = 1;

  Eqn_Diffusion1DCoF solver("solver");
  UNITTEST_ASSERT_TRUE(solver.set_parameters(params));

  Grid1DHostF host;
  UNITTEST_ASSERT_TRUE(host.init(N, 1));

  for (int T=0; T < 800; T++) {
    UNITTEST_ASSERT_TRUE(solver.advance(1));
  }
  ThreadManager::barrier_and_fence();

  // download values, output
  UNITTEST_ASSERT_TRUE(host.copy_all_data(solver.density()));

  // check symmetries
  int i;
  for (i=0; i < N/2; i++) {
    UNITTEST_ASSERT_EQUAL_FLOAT(host.at(i), host.at(N-1-i), .00001f);
    UNITTEST_ASSERT_FINITE(host.at(i));
  }

  // TODO: check that image0 is 1-image1
  if (tid == 0) {
  }
}

DECLARE_UNITTEST_END(Diffusion1DMultiTest);

