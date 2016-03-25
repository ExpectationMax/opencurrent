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

using namespace ocu;

float expected[100] = {
  0.00158441, 0.00169128, 0.00190964, 0.00224876, 0.00272273, 0.00335055, 0.00415634, 0.00516957, 0.00642522, 0.00796396, 0.0098323,
  0.0120825, 0.0147728, 0.0179666, 0.0217326, 0.0261441, 0.0312777, 0.0372126, 0.0440292, 0.0518073, 0.0606241,
  0.0705523, 0.0816574, 0.0939955, 0.10761, 0.12253, 0.138767, 0.156312, 0.175132, 0.195173, 0.216352,
  0.238562, 0.261665, 0.285499, 0.309875, 0.334579, 0.359377, 0.384016, 0.408227, 0.431733, 0.454251,
  0.4755, 0.495205, 0.513101, 0.528944, 0.542512, 0.553611, 0.562081, 0.567799, 0.570679, 0.570679,
  0.567799, 0.562081, 0.553611, 0.542512, 0.528944, 0.513101, 0.495205, 0.4755, 0.454251, 0.431733,
  0.408227, 0.384016, 0.359377, 0.334579, 0.309875, 0.285499, 0.261665, 0.238562, 0.216352, 0.195173,
  0.175132, 0.156312, 0.138767, 0.12253, 0.10761, 0.0939955, 0.0816574, 0.0705523, 0.0606241, 0.0518073,
  0.0440292, 0.0372126, 0.0312777, 0.0261441, 0.0217326, 0.0179666, 0.0147728, 0.0120826, 0.0098323, 0.00796397,
  0.00642522, 0.00516957, 0.00415634, 0.00335055, 0.00272273, 0.00224876, 0.00190964, 0.00169128, 0.00158441,
};

DECLARE_UNITTEST_BEGIN(Diffusion1DTest);

void run()
{
  Eqn_Diffusion1DParams params;
  params.h = 1;
  params.nx = 100;
  params.left.type = BC_NEUMANN;
  params.left.value = 0;
  params.right.type = BC_NEUMANN;
  params.right.value = 0;
  params.diffusion_coefficient = 1;

  params.initial_values.init(100, 0);

  int i,t;

  // initialize to an impulse
  for (i=0; i < 100; i++) {
    if (i < 40) params.initial_values.at(i) = 0;
    else if (i < 60) params.initial_values.at(i) = 1;
    else params.initial_values.at(i) = 0;
  }

  Eqn_Diffusion1D eqn;
  if (!eqn.set_parameters(params))
    UNITTEST_ASSERT_TRUE(false);


  for (t=0; t < 800; t++) {
    eqn.advance(.1);
  }

  eqn.copy_density_to_host();
  for (int i=0; i < 100; i++) {
    float soln_val = eqn.density().at(i);

    // should match expected results
    UNITTEST_ASSERT_EQUAL_FLOAT(soln_val, expected[i], 1e-6f);

    // should be symmetric
    UNITTEST_ASSERT_EQUAL_FLOAT(soln_val, eqn.density().at(99-i), 1e-7);
  }
}

DECLARE_UNITTEST_END(Diffusion1DTest);

