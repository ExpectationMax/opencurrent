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

#include "ocustorage/grid3dboundary.h"
#include "ocuequation/sol_projectmixed3d.h"

namespace ocu {

#ifdef OCU_DOUBLESUPPORT

bool Sol_ProjectDivergenceMixed3DDeviceD::solve(double tolerance)
{
  clear_error();
  double residual = 0;

  check_ok(apply_3d_mac_boundary_conditions_level1( *u, *v, *w,  bc, _hx, _hy, _hz), "Sol_ProjectDivergenceMixed3DDeviceD::solve - could not enforce boundary conditions");
  check_ok(divergence_solver.solve(), "Sol_ProjectDivergenceMixed3DDeviceD::solve - could not calculate divergence");
  check_ok(pressure_solver.solve(residual, tolerance, 15), "Sol_ProjectDivergenceMixed3DDeviceD::solve - could not solve for pressure\n");
  check_ok(gradient_solver.solve(), "Sol_ProjectDivergenceMixed3DDeviceD::solve - could not subtract gradient of pressure\n");
  check_ok(apply_3d_mac_boundary_conditions_level1( *u, *v, *w,  bc, _hx, _hy, _hz), "Sol_ProjectDivergenceMixed3DDeviceD::solve - could not enforce boundary conditions\n");

  return !any_error();
}

bool Sol_ProjectDivergenceMixed3DDeviceD::solve_divergence_only()
{
  this->clear_error();

  check_ok(apply_3d_mac_boundary_conditions_level1( *this->u, *this->v, *this->w,  this->bc, this->_hx, this->_hy, this->_hz), "Sol_ProjectDivergenceMixed3DDeviceD::solve - could not enforce boundary conditions");
  check_ok(this->divergence_solver.solve(), "Sol_ProjectDivergenceMixed3DDeviceD::solve - could not calculate divergence");

  return !this->any_error();
}


bool Sol_ProjectDivergenceMixed3DDeviceD::initialize_storage(
  int nx, int ny, int nz, double hx, double hy, double hz, Grid3DDeviceD *u_val, Grid3DDeviceD *v_val, Grid3DDeviceD *w_val)
{
  if (!initialize_base_storage(nx,ny,nz,hx,hy,hz,u_val, v_val, w_val)) {
    printf("[ERROR] Sol_ProjectDivergenceMixed3DDeviceD::initialize_storage - failed to initialize base storage\n");
    return false;
  }

  if (!initialize_device_storage(nx,ny,nz,hx,hy,hz,u_val, v_val, w_val)) {
    printf("[ERROR] Sol_ProjectDivergenceMixed3DDeviceD::initialize_storage - failed to initialize device storage\n");
    return false;
  }

  this->pressure_solver.bc.xneg = convert_bc_to_poisson_eqn(this->bc.xneg);
  this->pressure_solver.bc.xpos = convert_bc_to_poisson_eqn(this->bc.xpos);
  this->pressure_solver.bc.yneg = convert_bc_to_poisson_eqn(this->bc.yneg);
  this->pressure_solver.bc.ypos = convert_bc_to_poisson_eqn(this->bc.ypos);
  this->pressure_solver.bc.zneg = convert_bc_to_poisson_eqn(this->bc.zneg);
  this->pressure_solver.bc.zpos = convert_bc_to_poisson_eqn(this->bc.zpos);

  if (!divergence_solver.initialize_storage(nx, ny, nz, hx, hy, hz, this->u, this->v, this->w, &this->divergence)) {
    printf("[ERROR] Sol_ProjectDivergenceMixed3DDeviceD::initialize_storage - failed to initialize divergence_solver\n");
    return false;
  }

  if (!pressure_solver.initialize_storage(nx, ny, nz, hx, hy, hz, &this->divergence)) {
    printf("[ERROR] Sol_ProjectDivergenceMixed3DDeviceD::initialize_storage - failed to initialize pressure_solver\n");
    return false;
  }

  if (!gradient_solver.initialize_storage(nx, ny, nz, hx, hy, hz, this->u, this->v, this->w, &this->pressure_solver.pressure())) {
    printf("[ERROR] Sol_ProjectDivergenceMixed3DDeviceD::initialize_storage - failed to initialize gradient_solver\n");
    return false;
  }
  this->gradient_solver.coefficient = -1;

  if (!u_val->check_layout_match(this->pressure_solver.pressure())) {
    printf("[ERROR] Sol_ProjectDivergenceMixed3DDeviceD::initialize_storage - pressure layout mismatch\n");
    return false;
  }

  this->pressure_solver.pressure().clear_zero();
  this->pressure_solver.convergence = CONVERGENCE_LINF;

  return true;
}

#endif //  OCU_DOUBLESUPPORT

} // end namespace

