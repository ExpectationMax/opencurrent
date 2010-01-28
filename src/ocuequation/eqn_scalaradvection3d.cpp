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

#include "ocuutil/float_routines.h"
#include "ocustorage/grid3dboundary.h"
#include "ocuequation/eqn_scalaradvection3d.h"

namespace ocu {

template<typename T>
bool 
Eqn_ScalarAdvection3D<T>::set_parameters(const Eqn_ScalarAdvection3DParams<T> &params)
{
  bc = params.bc;
  _advection.interp_type = params.advection_scheme;
  
  if (!u.init_congruent(params.u) ||
      !v.init_congruent(params.v) ||
      !w.init_congruent(params.w)) {
    printf("[ERROR] Eqn_ScalarAdvection3D::set_parameters - failed to initialize u,v,w grids\n");
    return false;
  }

  if (!u.copy_all_data(params.u) || 
      !v.copy_all_data(params.v) ||
      !w.copy_all_data(params.w)) {
    printf("[ERROR] Eqn_ScalarAdvection3D::set_parameters - failed to copy data to u,v,w grids\n");
    return false;
  }

  if (!phi.init_congruent(params.initial_values)) {
    printf("[ERROR] Eqn_ScalarAdvection3D::set_parameters - failed to initialize phi grid\n");
    return false;
  }

  if (!phi.copy_all_data(params.initial_values)) {
    printf("[ERROR] Eqn_ScalarAdvection3D::set_parameters - failed to copy data to phi grid\n");
    return false;
  }

  if (!deriv_phidt.init_congruent(params.initial_values)) {
    printf("[ERROR] Eqn_ScalarAdvection3D::set_parameters - failed to initialize deriv_phidt grid\n");
    return false;
  }

  if (!_advection.initialize_storage(params.nx, params.ny, params.nz, params.hx, params.hy, params.hz, &u, &v, &w, &phi, &deriv_phidt)) {
    printf("[ERROR] Eqn_ScalarAdvection3D::set_parameters - failed to initialize advection solver.\n");
    return false;
  }

  if (!apply_3d_boundary_conditions_level1_nocorners(phi, bc, hx(), hy(), hz())) {
    printf("[ERROR] Eqn_ScalarAdvection3D::set_parameters - failed to enforce boundary conditions.\n");
    return false;
  }

  return true;
}

template<typename T>
double Eqn_ScalarAdvection3D<T>::get_max_stable_timestep() const
{
  T max_u, max_v, max_w;
  u.reduce_maxabs(max_u);
  v.reduce_maxabs(max_v);
  w.reduce_maxabs(max_w);
  double ut = hx() / max_u;
  double vt = hy() / max_v;
  double wt = hz() / max_w;

  if (!check_float(ut)) ut = 1e10;
  if (!check_float(vt)) vt = 1e10;
  if (!check_float(wt)) wt = 1e10;

  return min3(ut, vt, wt);
}


template<typename T>
bool Eqn_ScalarAdvection3D<T>::advance_one_step(double dt)
{
  clear_error();

  if (!check_float(dt)) {
    printf("[ERROR] Eqn_ScalarAdvection3D::advance_one_step - bad dt value\n");
    return false;
  }

  check_ok(_advection.solve()); // calc dphidt

  // forward euler
  check_ok(phi.linear_combination((T)1, phi, (T)dt, deriv_phidt)); 
  check_ok(apply_3d_boundary_conditions_level1_nocorners(phi, bc, hx(), hy(), hz()));

  return !any_error();
}


template class Eqn_ScalarAdvection3D<float>;

#ifdef OCU_DOUBLESUPPORT
template class Eqn_ScalarAdvection3D<double>;
#endif // OCU_DOUBLESUPPORT

}

