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

#include <cstdio>
#include "ocuequation/sol_laplaciancent1d.h"

namespace ocu {


bool 
Sol_LaplacianCentered1DHost::initialize_storage(int nx)
{
  density.init(nx, 1);
  deriv_densitydt.init(nx, 0);
  _nx = nx;

  return true;
}

void 
Sol_LaplacianCentered1DHost::apply_boundary_conditions()
{
  if (left.type == BC_PERIODIC) {
    density.at(-1) = density.at(_nx-1);
  }
  else if (left.type == BC_DIRICHELET) {
    density.at(-1) = 2 * left.value - density.at(0);
  }
  else if (left.type == BC_NEUMANN) {
    density.at(-1) = density.at(0) + h() * left.value;
  }
  else {
    printf("[WARNING] Sol_LaplacianCentered1DHost::apply_boundary_conditions - Invalid left boundary condition\n");
  }

  if (right.type == BC_PERIODIC) {
    density.at(_nx) = density.at(1);
  }
  else if (right.type == BC_DIRICHELET) {
    density.at(_nx) = 2 * right.value - density.at(_nx-1);
  }
  else if (right.type == BC_NEUMANN) {
    density.at(_nx) = density.at(_nx-1) + h() * right.value;
  }
  else {
    printf("[WARNING] Sol_LaplacianCentered1DHost::apply_boundary_conditions - Invalid right boundary condition\n");
  }
}

bool
Sol_LaplacianCentered1DHost::solve()
{
  // centered differencing
  float inv_h2 = coefficient() / (h() * h());

  apply_boundary_conditions();

  int i;
  for (i=0; i < _nx; i++) {
    deriv_densitydt.at(i) = inv_h2 * (density.at(i-1) - 2.0f * density.at(i) + density.at(i+1));
  }

  return true;
}


}

