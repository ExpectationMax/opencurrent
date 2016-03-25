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

#include "ocuequation/equation.h"
#include "ocuutil/float_routines.h"

namespace ocu {



Equation::~Equation()
{
}




bool Equation::advance(double dt)
{
  if (!check_float(dt)) {
    printf("[ERROR] Equation::advance - bad dt value\n");
    return false;
  }

  double dtleft = dt;
  int substeps = 0;

  while (dtleft > 0) {
    int subds = (int) ceil(dtleft / get_max_stable_timestep());
    double subdt = dtleft / subds;

    if (!advance_one_step(subdt)) {
      printf("[ERROR] Equation::advance - failed to advance equation\n");
      return false;
    }

    dtleft -= subdt;
    substeps++;
  }

  //printf("[INFO] Equation::advance - %d substeps\n", substeps);
  return true;
}



} // end namespace

