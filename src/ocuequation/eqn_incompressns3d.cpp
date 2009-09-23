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

#include <algorithm>

#include "ocuequation/eqn_incompressns3d.h"  
#include "ocuutil/float_routines.h"
#include "ocustorage/grid3dboundary.h"

namespace ocu {

template<typename T>
Eqn_IncompressibleNS3D<T>::Eqn_IncompressibleNS3D()
{
  num_steps = 0;
  _nx = 0;
  _ny = 0;
  _nz = 0;
  _hx = 0;
  _hy = 0;
  _hz = 0;
  _lastdt = 0;
  _max_divergence = 0;
  _cfl_factor = 0;
  _bouyancy = 0;
  _gravity = 0;
  _time_step = TS_ERROR;
  _vertical_direction = DIR_ZPOS;
}

template<typename T>
bool 
Eqn_IncompressibleNS3D<T>::set_parameters(const Eqn_IncompressibleNS3DParams<T> &params)
{
  _max_divergence = params.max_divergence;
  _cfl_factor = params.cfl_factor;
  _bouyancy = params.bouyancy;
  _gravity = params.gravity;

  if (!check_float(_max_divergence) || _max_divergence < 0 || _max_divergence > 1) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - invalid max_divergence %f\n", _max_divergence);
    return false;
  }
 
  if (_cfl_factor < 0) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - invalid cfl_factor %f\n", _cfl_factor);
    return false;
  }

  _time_step = params.time_step;
  if (_time_step != TS_ADAMS_BASHFORD2 && _time_step != TS_FORWARD_EULER) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - unknown timestep type %d\n", _time_step);
    return false;
  }

  _nx = params.nx;
  _ny = params.ny;
  _nz = params.nz;
  _hx = params.hx;
  _hy = params.hy;
  _hz = params.hz;
  _lastdt = 0;

  if (params.vertical_direction != DIR_XPOS && params.vertical_direction != DIR_XNEG &&
      params.vertical_direction != DIR_YPOS && params.vertical_direction != DIR_YNEG &&
      params.vertical_direction != DIR_ZPOS && params.vertical_direction != DIR_ZNEG) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - bad vertical_direction %d\n", params.vertical_direction);
    return false;
  }

  _vertical_direction = params.vertical_direction;

  if (!check_float(_hx) || !check_float(_hy) || !check_float(_hz) || _hx <= 0 || _hy <= 0 || _hz <= 0) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - bad hx, hy, hz (%f, %f, %f)\n", _hx, _hy, _hz);
    return false;
  }

  if (!_u.init_congruent(params.init_u) || !_deriv_udt.init_congruent(params.init_u) || !_last_deriv_udt.init_congruent(params.init_u)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on initializing u\n");
    return false;
  }

  if (!_u.copy_all_data(params.init_u)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed copying to u\n");
    return false;  
  }

  if (!_v.init_congruent(params.init_v) || !_deriv_vdt.init_congruent(params.init_v) || !_last_deriv_vdt.init_congruent(params.init_v)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on initializing v\n");
    return false;
  }

  if (!_v.copy_all_data(params.init_v)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed copying to v\n");
    return false;  
  }

  if (!_w.init_congruent(params.init_w) || !_deriv_wdt.init_congruent(params.init_w) || !_last_deriv_wdt.init_congruent(params.init_w)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on initializing w\n");
    return false;
  }

  if (!_w.copy_all_data(params.init_w)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed copying to w\n");
    return false;  
  }

  if (!_temp.init_congruent(params.init_temp) || !_deriv_tempdt.init_congruent(params.init_temp) || !_last_deriv_tempdt.init_congruent(params.init_temp)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on initializing temp\n");
    return false;
  }
  
  if (!_temp.copy_all_data(params.init_temp)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on copying temperature field\n");
    return false;  
  }

  _thermalbc = params.temp_bc;
  _projection_solver.bc = params.flow_bc;
  _advection_solver.interp_type = params.advection_scheme;
  _thermal_solver.interp_type = params.advection_scheme;

  if (!_thermal_solver.initialize_storage(_nx, _ny, _nz, _hx, _hy, _hz, &_u, &_v, &_w, &_temp, &_deriv_tempdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _thermal_solver initialization\n");
    return false;  
  }

  if (!_advection_solver.initialize_storage(_nx, _ny, _nz, _hx, _hy, _hz, &_u, &_v, &_w, &_deriv_udt, &_deriv_vdt, &_deriv_wdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _advection_solver initialization\n");
    return false;  
  }

  if (!_projection_solver.initialize_storage(_nx, _ny, _nz, _hx, _hy, _hz, &_u, &_v, &_w)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _projection_solver initialization\n");
    return false;  
  }

  if (!_thermal_diffusion.initialize_storage(_nx, _ny, _nz, _hx, _hy, _hz, &_temp, &_deriv_tempdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _thermal_diffusion initialization\n");
    return false;
  }

  if (!check_float(params.thermal_diffusion) || params.thermal_diffusion < 0) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - invalid thermal diffusion %f\n", params.thermal_diffusion);
    return false;
  }

  _thermal_diffusion.coefficient = params.thermal_diffusion;

  if (!_u_diffusion.initialize_storage(_nx+1, _ny, _nz, _hx, _hy, _hz, &_u, &_deriv_udt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _u_diffusion initialization\n");
    return false;
  }

  if (!check_float(params.viscosity) || params.viscosity < 0) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - invalid viscosity %f\n", params.viscosity);
    return false;
  }
  _u_diffusion.coefficient = params.viscosity;

  if (!_v_diffusion.initialize_storage(_nx, _ny+1, _nz, _hx, _hy, _hz, &_v, &_deriv_vdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _v_diffusion initialization\n");
    return false;
  }
  _v_diffusion.coefficient = params.viscosity;

  if (!_w_diffusion.initialize_storage(_nx, _ny, _nz+1, _hx, _hy, _hz, &_w, &_deriv_wdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _w_diffusion initialization\n");
    return false;
  }
  _w_diffusion.coefficient = params.viscosity;

  if (!apply_3d_mac_boundary_conditions_level1(_u, _v, _w, params.flow_bc, _hx, _hy, _hz)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on enforcing flow boundary conditions\n");
    return false;  
  }

  if (!apply_3d_boundary_conditions_level1(_temp, _thermalbc, _hx, _hy, _hz)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on enforcing thermal boundary conditions\n");
    return false;  
  }

  // all grid layouts should match
  if (!_u.check_layout_match(_v) || 
      !_u.check_layout_match(_w) || 
      !_u.check_layout_match(_deriv_udt) || 
      !_u.check_layout_match(_deriv_vdt) || 
      !_u.check_layout_match(_deriv_wdt) ||
      !_u.check_layout_match(_temp) ||
      !_u.check_layout_match(_deriv_tempdt) ||
      !_u.check_layout_match(_last_deriv_tempdt) ||
      !_u.check_layout_match(_last_deriv_udt) ||
      !_u.check_layout_match(_last_deriv_vdt) ||
      !_u.check_layout_match(_last_deriv_wdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - grid layouts do not all match\n");
    return false;  
  }

  return true;
}

template<typename T>
double Eqn_IncompressibleNS3D<T>::get_max_stable_timestep() const
{
  T max_u, max_v, max_w;
  _u.reduce_maxabs(max_u);
  _v.reduce_maxabs(max_v);
  _w.reduce_maxabs(max_w);
  double ut = hx() / max_u;
  double vt = hy() / max_v;
  double wt = hz() / max_w;

  if (!check_float(ut)) ut = 1e10;
  if (!check_float(vt)) vt = 1e10;
  if (!check_float(wt)) wt = 1e10;

  double step = _cfl_factor * min3(ut, vt, wt);

  double minh = min3(hx(), hy(), hz());

  if (thermal_diffusion_coefficient() > 0)
    step = std::min(step, (minh * minh) / (6 * thermal_diffusion_coefficient()));
  if (viscosity_coefficient() > 0)
    step = std::min(step, (minh * minh) / (6 * viscosity_coefficient()));

  return step;
}


template<typename T>
bool Eqn_IncompressibleNS3D<T>::advance_one_step(double dt)
{
  clear_error();
  num_steps++;

  // update dudt
  check_ok(_advection_solver.solve()); // updates dudt, dvdt, dwdt, overwrites whatever is there

  if (viscosity_coefficient() > 0) {
    check_ok(_u_diffusion.solve()); // dudt += \nu \nabla^2 u
    check_ok(_v_diffusion.solve()); // dvdt += \nu \nabla^2 v
    check_ok(_w_diffusion.solve()); // dwdt += \nu \nabla^2 w
  }

  // eventually this will be replaced with a grid-wide operation.
  add_thermal_force();

  // update dTdt

  check_ok(_thermal_solver.solve());   // updates dTdt, overwrites whatever is there
  if (thermal_diffusion_coefficient() > 0) {
    check_ok(_thermal_diffusion.solve()); // dTdt += k \nabla^2 T
  }

  T ab_coeff = -dt*dt / (2 * _lastdt);

  // advance T 
  if (_time_step == TS_ADAMS_BASHFORD2 && _lastdt > 0) {
    check_ok(_temp.linear_combination((T)1.0, _temp, (T)(dt - ab_coeff), _deriv_tempdt));
    check_ok(_temp.linear_combination((T)1.0, _temp, (T)ab_coeff, _last_deriv_tempdt));
  } 
  else {
    check_ok(_temp.linear_combination((T)1.0, _temp, (T)dt, _deriv_tempdt));
  }

  check_ok(apply_3d_boundary_conditions_level1_nocorners(_temp, _thermalbc, _hx, _hy, _hz));

  // advance u,v,w
  if (_time_step == TS_ADAMS_BASHFORD2 && _lastdt > 0) {
    check_ok(_u.linear_combination((T)1.0, _u, (T)(dt - ab_coeff), _deriv_udt));
    check_ok(_u.linear_combination((T)1.0, _u, (T)ab_coeff, _last_deriv_udt));

    check_ok(_v.linear_combination((T)1.0, _v, (T)(dt - ab_coeff), _deriv_vdt));
    check_ok(_v.linear_combination((T)1.0, _v, (T)ab_coeff, _last_deriv_vdt));

    check_ok(_w.linear_combination((T)1.0, _w, (T)(dt - ab_coeff), _deriv_wdt));
    check_ok(_w.linear_combination((T)1.0, _w, (T)ab_coeff, _last_deriv_wdt));

  }
  else {
    check_ok(_u.linear_combination((T)1.0, _u, (T)dt, _deriv_udt));
    check_ok(_v.linear_combination((T)1.0, _v, (T)dt, _deriv_vdt)); 
    check_ok(_w.linear_combination((T)1.0, _w, (T)dt, _deriv_wdt));
  }

  // copy state for AB2
  if (_time_step == TS_ADAMS_BASHFORD2) {
    _lastdt = dt;
    _last_deriv_tempdt.copy_all_data(_deriv_tempdt);
    _last_deriv_udt.copy_all_data(_deriv_udt);
    _last_deriv_vdt.copy_all_data(_deriv_vdt);
    _last_deriv_wdt.copy_all_data(_deriv_wdt);
  }

  // enforce incompressibility - this enforces bc's before and after projection
  check_ok(_projection_solver.solve(_max_divergence));

  return !any_error();
}


template bool Eqn_IncompressibleNS3D<float>::advance_one_step(double dt);
template double Eqn_IncompressibleNS3D<float>::get_max_stable_timestep() const;
template Eqn_IncompressibleNS3D<float>::Eqn_IncompressibleNS3D();
template bool Eqn_IncompressibleNS3D<float>::set_parameters(const Eqn_IncompressibleNS3DParams<float> &params);

#ifdef OCU_DOUBLESUPPORT
template bool Eqn_IncompressibleNS3D<double>::advance_one_step(double dt);
template double Eqn_IncompressibleNS3D<double>::get_max_stable_timestep() const;
template Eqn_IncompressibleNS3D<double>::Eqn_IncompressibleNS3D();
template bool Eqn_IncompressibleNS3D<double>::set_parameters(const Eqn_IncompressibleNS3DParams<double> &params);
#endif





} // end namespace

