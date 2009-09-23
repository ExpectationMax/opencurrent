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

#include "ocuutil/float_routines.h"
#include "ocustorage/grid3dops.h"
#include "ocuequation/sol_project3d.h"


namespace ocu {

BoundaryCondition
Sol_ProjectDivergence3DBase::convert_bc_to_poisson_eqn(const BoundaryCondition &bc) const
{
  BoundaryCondition ret;

  if (bc.type == BC_PERIODIC) {
    ret.type = BC_PERIODIC;
  }
  else if (bc.type == BC_FORCED_INFLOW_VARIABLE_SLIP) {
    ret.type = BC_NEUMANN;
    ret.value = 0;
  }
  else {
    printf("[ERROR] Sol_ProjectDivergence3DBase::convert_bc_to_poisson_eqn - invalid boundary condition type %d\n", bc.type);
  }

  return ret;
}

bool 
Sol_ProjectDivergence3DBase::initialize_base_storage(
  int nx, int ny, int nz, double hx, double hy, double hz, Grid3DUntyped *u_val, Grid3DUntyped *v_val, Grid3DUntyped *w_val)
{
  if (!check_valid_mac_dimensions(*u_val, *v_val, *w_val, nx, ny, nz)) {
    printf("[ERROR] Sol_ProjectDivergence3DBase::initialize_base_storage - u,v,w grid dimensions mismatch\n");
    return false;
  }

  if (!check_float(hx) || !check_float(hy) || !check_float(hz)) {
    printf("[ERROR] Sol_ProjectDivergence3DBase::initialize_base_storage - garbage hx,hy,hz value\n");
    return false;
  }

  _hx = hx;
  _hy = hy;
  _hz = hz;

  _nx = nx;
  _ny = ny;
  _nz = nz;

  return true;
}

} // end namespace

