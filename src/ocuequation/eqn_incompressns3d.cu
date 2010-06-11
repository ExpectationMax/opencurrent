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

#include "ocuutil/float_routines.h"
#include "ocustorage/grid3dboundary.h"
#include "ocuutil/kernel_wrapper.h"
#include "ocustorage/coarray.h"
#include "ocuequation/eqn_incompressns3d.h"

template<typename T>
__global__ void Eqn_IncompressibleNS3D_add_thermal_force(T *dvdt, T coefficient, const T *temperature,
  int xstride, int ystride, int nbr_stride, 
  int nx, int ny, int nz, int blocksInY, float invBlocksInY)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);

  // transpose for coalescing since k is the fastest changing index 
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  // shift so we will get maximum coalescing.  This means that we will need to test if k>0 below.
  int idx = __mul24(i, xstride) + __mul24(j,ystride) + k;

  if (i < nx && j < ny && k < nz) {
    dvdt[idx] += ((T).5) * coefficient * (temperature[idx] + temperature[idx-nbr_stride]);
  }
}

namespace ocu {


template<typename T>
Eqn_IncompressibleNS3DBase<T>::Eqn_IncompressibleNS3DBase()
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
Eqn_IncompressibleNS3DBase<T>::set_base_parameters(const Eqn_IncompressibleNS3DParams<T> &params)
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
  _thermalbc = params.temp_bc;

  return true;
}

template<typename T>
void 
Eqn_IncompressibleNS3D<T>::add_thermal_force()
{
  // apply thermal force by adding -gkT to dvdt (let g = -1, k = 1, so this is just dvdt += T)
  //_advection_solver.deriv_vdt.linear_combination((T)1.0, _advection_solver.deriv_vdt, (T)1.0, _thermal_solver.phi);

  int tnx = this->nz();
  int tny = this->ny();
  int tnz = this->nx();

  int threadsInX = 16;
  int threadsInY = 2;
  int threadsInZ = 2;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  T direction_mult = this->_vertical_direction & DIR_NEGATIVE_FLAG ? 1 : -1;
  T *uvw = (this->_vertical_direction & DIR_XAXIS_FLAG) ? &_deriv_udt.at(0,0,0) :
           (this->_vertical_direction & DIR_YAXIS_FLAG) ? &_deriv_vdt.at(0,0,0) : &_deriv_wdt.at(0,0,0);

  KernelWrapper wrapper;
  wrapper.PreKernel();

  Eqn_IncompressibleNS3D_add_thermal_force<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(uvw, direction_mult * this->_gravity * this->_bouyancy, &_temp.at(0,0,0),
    _temp.xstride(), _temp.ystride(), _temp.stride(this->_vertical_direction), this->nx(), this->ny(), this->nz(), 
    blocksInY, 1.0f / (float)blocksInY);

  if (!wrapper.PostKernel("Eqn_IncompressibleNS3D_add_thermal_force"))
    this->add_error();

}



template<typename T>
Eqn_IncompressibleNS3D<T>::Eqn_IncompressibleNS3D()
{
}

template<typename T>
bool 
Eqn_IncompressibleNS3D<T>::set_parameters(const Eqn_IncompressibleNS3DParams<T> &params)
{
  if (!set_base_parameters(params)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on base parameters\n");
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

  _projection_solver.bc = params.flow_bc;
  _advection_solver.interp_type = params.advection_scheme;
  _thermal_solver.interp_type = params.advection_scheme;

  if (!_thermal_solver.initialize_storage(this->_nx, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_u, &_v, &_w, &_temp, &_deriv_tempdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _thermal_solver initialization\n");
    return false;  
  }

  if (!_advection_solver.initialize_storage(this->_nx, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_u, &_v, &_w, &_deriv_udt, &_deriv_vdt, &_deriv_wdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _advection_solver initialization\n");
    return false;  
  }

  if (!_projection_solver.initialize_storage(this->_nx, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_u, &_v, &_w)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _projection_solver initialization\n");
    return false;  
  }

  if (!_thermal_diffusion.initialize_storage(this->_nx, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_temp, &_deriv_tempdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _thermal_diffusion initialization\n");
    return false;
  }

  if (!check_float(params.thermal_diffusion) || params.thermal_diffusion < 0) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - invalid thermal diffusion %f\n", params.thermal_diffusion);
    return false;
  }

  _thermal_diffusion.coefficient = params.thermal_diffusion;

  if (!_u_diffusion.initialize_storage(this->_nx+1, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_u, &_deriv_udt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _u_diffusion initialization\n");
    return false;
  }

  if (!check_float(params.viscosity) || params.viscosity < 0) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - invalid viscosity %f\n", params.viscosity);
    return false;
  }
  _u_diffusion.coefficient = params.viscosity;

  if (!_v_diffusion.initialize_storage(this->_nx, this->_ny+1, this->_nz, this->_hx, this->_hy, this->_hz, &_v, &_deriv_vdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _v_diffusion initialization\n");
    return false;
  }
  _v_diffusion.coefficient = params.viscosity;

  if (!_w_diffusion.initialize_storage(this->_nx, this->_ny, this->_nz+1, this->_hx, this->_hy, this->_hz, &_w, &_deriv_wdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on _w_diffusion initialization\n");
    return false;
  }
  _w_diffusion.coefficient = params.viscosity;

  if (!apply_3d_mac_boundary_conditions_level1(_u, _v, _w, params.flow_bc, this->_hx, this->_hy, this->_hz)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on enforcing flow boundary conditions\n");
    return false;  
  }

  if (!apply_3d_boundary_conditions_level1(_temp, this->_thermalbc, this->_hx, this->_hy, this->_hz)) {
    printf("[ERROR] Eqn_IncompressibleNS3D::set_parameters - failed on enforcing thermal boundary conditions\n");
    return false;  
  }

  _deriv_udt.clear_zero();
  _deriv_vdt.clear_zero();
  _deriv_wdt.clear_zero();
  _deriv_tempdt.clear_zero();

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
  double ut = this->hx() / max_u;
  double vt = this->hy() / max_v;
  double wt = this->hz() / max_w;

  if (!check_float(ut)) ut = 1e10;
  if (!check_float(vt)) vt = 1e10;
  if (!check_float(wt)) wt = 1e10;

  double step = this->_cfl_factor * min3(ut, vt, wt);

  double minh = min3(this->hx(), this->hy(), this->hz());

  if (thermal_diffusion_coefficient() > 0)
    step = std::min(step, (minh * minh) / (6 * thermal_diffusion_coefficient()));
  if (viscosity_coefficient() > 0)
    step = std::min(step, (minh * minh) / (6 * viscosity_coefficient()));

  printf("Eqn_IncompressibleNS3D<T>::get_max_stable_timestep - return %f\n", step);

  return step;
}


template<typename T>
bool Eqn_IncompressibleNS3D<T>::advance_one_step(double dt)
{
  this->clear_error();
  this->num_steps++;

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

  T ab_coeff = -dt*dt / (2 * this->_lastdt);

  // advance T 
  if (this->_time_step == TS_ADAMS_BASHFORD2 && this->_lastdt > 0) {
    check_ok(_temp.linear_combination((T)1.0, _temp, (T)(dt - ab_coeff), _deriv_tempdt));
    check_ok(_temp.linear_combination((T)1.0, _temp, (T)ab_coeff, _last_deriv_tempdt));
  } 
  else {
    check_ok(_temp.linear_combination((T)1.0, _temp, (T)dt, _deriv_tempdt));
  }

  check_ok(apply_3d_boundary_conditions_level1_nocorners(_temp, this->_thermalbc, this->_hx, this->_hy, this->_hz));

  // advance u,v,w
  if (this->_time_step == TS_ADAMS_BASHFORD2 && this->_lastdt > 0) {
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
  if (this->_time_step == TS_ADAMS_BASHFORD2) {
    this->_lastdt = dt;
    _last_deriv_tempdt.copy_all_data(_deriv_tempdt);
    _last_deriv_udt.copy_all_data(_deriv_udt);
    _last_deriv_vdt.copy_all_data(_deriv_vdt);
    _last_deriv_wdt.copy_all_data(_deriv_wdt);
  }

  // enforce incompressibility - this enforces bc's before and after projection
  check_ok(_projection_solver.solve(this->_max_divergence));

  return !this->any_error();
}





template<typename T>
Eqn_IncompressibleNS3DCo<T>::Eqn_IncompressibleNS3DCo(const char *name):
  _projection_solver((std::string(name) + std::string("._projection_solver")).c_str()),
  _u((std::string(name) + std::string("._u")).c_str()),
  _v((std::string(name) + std::string("._v")).c_str()),
  _w((std::string(name) + std::string("._w")).c_str()),
  _temp((std::string(name) + std::string("._temp")).c_str()),
  _deriv_udt((std::string(name) + std::string("._deriv_udt")).c_str()),
  _deriv_vdt((std::string(name) + std::string("._deriv_vdt")).c_str()),
  _deriv_wdt((std::string(name) + std::string("._deriv_wdt")).c_str()),
  _deriv_tempdt((std::string(name) + std::string("._deriv_tempdt")).c_str())
{
  _u_negx_hdl = -1;
  _v_negx_hdl = -1;
  _w_negx_hdl = -1;
  _t_negx_hdl = -1;

  _u_posx_hdl = -1;
  _v_posx_hdl = -1;
  _w_posx_hdl = -1;
  _t_posx_hdl = -1;
}

template<typename T>
Eqn_IncompressibleNS3DCo<T>::~Eqn_IncompressibleNS3DCo()
{
  CoArrayManager::barrier_deallocate(_u_negx_hdl);
  CoArrayManager::barrier_deallocate(_v_negx_hdl);
  CoArrayManager::barrier_deallocate(_w_negx_hdl);
  CoArrayManager::barrier_deallocate(_t_negx_hdl);

  CoArrayManager::barrier_deallocate(_u_posx_hdl);
  CoArrayManager::barrier_deallocate(_v_posx_hdl);
  CoArrayManager::barrier_deallocate(_w_posx_hdl);
  CoArrayManager::barrier_deallocate(_t_posx_hdl);
}

template<typename T>
void
Eqn_IncompressibleNS3DCo<T>::do_halo_exchange_uvw()
{
  CoArrayManager::barrier_exchange(_u_negx_hdl);
  CoArrayManager::barrier_exchange(_v_negx_hdl);
  CoArrayManager::barrier_exchange(_w_negx_hdl);
  CoArrayManager::barrier_exchange(_u_posx_hdl);
  CoArrayManager::barrier_exchange(_v_posx_hdl);
  CoArrayManager::barrier_exchange(_w_posx_hdl);
}

template<typename T>
void
Eqn_IncompressibleNS3DCo<T>::do_halo_exchange_t()
{
  CoArrayManager::barrier_exchange(_t_negx_hdl);
  CoArrayManager::barrier_exchange(_t_posx_hdl);
}


template<typename T>
bool 
Eqn_IncompressibleNS3DCo<T>::set_parameters(const Eqn_IncompressibleNS3DParams<T> &params)
{
  int tid = ThreadManager::this_image();
  int num_images = ThreadManager::num_images();

  if (!set_base_parameters(params)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on base parameters\n");
    return false;
  }

  if (!_u.init_congruent(params.init_u) || !_deriv_udt.init_congruent(params.init_u) || !_last_deriv_udt.init_congruent(params.init_u)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on initializing u\n");
    return false;
  }

  if (!_u.copy_all_data(params.init_u)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed copying to u\n");
    return false;  
  }

  if (!_v.init_congruent(params.init_v) || !_deriv_vdt.init_congruent(params.init_v) || !_last_deriv_vdt.init_congruent(params.init_v)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on initializing v\n");
    return false;
  }

  if (!_v.copy_all_data(params.init_v)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed copying to v\n");
    return false;  
  }

  if (!_w.init_congruent(params.init_w) || !_deriv_wdt.init_congruent(params.init_w) || !_last_deriv_wdt.init_congruent(params.init_w)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on initializing w\n");
    return false;
  }

  if (!_w.copy_all_data(params.init_w)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed copying to w\n");
    return false;  
  }

  if (!_temp.init_congruent(params.init_temp) || !_deriv_tempdt.init_congruent(params.init_temp) || !_last_deriv_tempdt.init_congruent(params.init_temp)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on initializing temp\n");
    return false;
  }
  
  if (!_temp.copy_all_data(params.init_temp)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on copying temperature field\n");
    return false;  
  }

  _projection_solver.bc = params.flow_bc;
  _advection_solver.interp_type = params.advection_scheme;
  _thermal_solver.interp_type = params.advection_scheme;

  if (!_thermal_solver.initialize_storage(this->_nx, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_u, &_v, &_w, &_temp, &_deriv_tempdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on _thermal_solver initialization\n");
    return false;  
  }

  if (!_advection_solver.initialize_storage(this->_nx, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_u, &_v, &_w, &_deriv_udt, &_deriv_vdt, &_deriv_wdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on _advection_solver initialization\n");
    return false;  
  }

  if (!_projection_solver.initialize_storage(this->_nx, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_u, &_v, &_w)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on _projection_solver initialization\n");
    return false;  
  }

  if (!_thermal_diffusion.initialize_storage(this->_nx, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_temp, &_deriv_tempdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on _thermal_diffusion initialization\n");
    return false;
  }

  if (!check_float(params.thermal_diffusion) || params.thermal_diffusion < 0) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - invalid thermal diffusion %f\n", params.thermal_diffusion);
    return false;
  }

  _thermal_diffusion.coefficient = params.thermal_diffusion;

  if (!_u_diffusion.initialize_storage(this->_nx+1, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_u, &_deriv_udt)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on _u_diffusion initialization\n");
    return false;
  }

  if (!check_float(params.viscosity) || params.viscosity < 0) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - invalid viscosity %f\n", params.viscosity);
    return false;
  }
  _u_diffusion.coefficient = params.viscosity;

  if (!_v_diffusion.initialize_storage(this->_nx, this->_ny+1, this->_nz, this->_hx, this->_hy, this->_hz, &_v, &_deriv_vdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on _v_diffusion initialization\n");
    return false;
  }
  _v_diffusion.coefficient = params.viscosity;

  if (!_w_diffusion.initialize_storage(this->_nx, this->_ny, this->_nz+1, this->_hx, this->_hy, this->_hz, &_w, &_deriv_wdt)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on _w_diffusion initialization\n");
    return false;
  }
  _w_diffusion.coefficient = params.viscosity;


  // figure out who our neighbors are
  int negx_image = (tid - 1 + num_images) % num_images;
  int posx_image = (tid + 1) % num_images;

  if (tid == 0 && params.flow_bc.xneg.type != BC_PERIODIC)
    negx_image = -1;

  if (tid == num_images - 1 && params.flow_bc.xpos.type != BC_PERIODIC) 
    posx_image = -1;

  // set up bc's & transfers here
  this->_local_bc = params.flow_bc;
  this->_local_thermalbc = this->_thermalbc;
  if (negx_image != -1) {
    _local_bc.xneg.type = BC_NONE;
    _local_thermalbc.xneg.type = BC_NONE;
  }

  if (posx_image != -1) {
    _local_bc.xpos.type = BC_NONE;
    _local_thermalbc.xpos.type = BC_NONE;
  }

  if (posx_image != -1) {
    Region3D uto   = _u                .region(this->_nx, this->_nx+1)()();
    Region3D ufrom = _u.co(posx_image)->region(0,1)()();

    _u_posx_hdl = CoArrayManager::barrier_allocate(uto, ufrom);
    if (_u_posx_hdl == -1)
      printf("[ERROR] Eqn_IncompressibleNS3DCo::initialize_storage - failed to allocate _u_posx_hdl\n");

    Region3D vto   = _v                .region(this->_nx)()();
    Region3D vfrom = _v.co(posx_image)->region(0)()();

    _v_posx_hdl = CoArrayManager::barrier_allocate(vto, vfrom);
    if (_v_posx_hdl == -1)
      printf("[ERROR] Eqn_IncompressibleNS3DCo::initialize_storage - failed to allocate _v_posx_hdl\n");

    Region3D wto   = _w                .region(this->_nx)()();
    Region3D wfrom = _w.co(posx_image)->region(0)()();

    _w_posx_hdl = CoArrayManager::barrier_allocate(wto, wfrom);
    if (_w_posx_hdl == -1)
      printf("[ERROR] Eqn_IncompressibleNS3DCo::initialize_storage - failed to allocate _w_posx_hdl\n");
    
    Region3D tto   = _temp                .region(this->_nx)()();
    Region3D tfrom = _temp.co(posx_image)->region(0)()();

    _t_posx_hdl = CoArrayManager::barrier_allocate(tto, tfrom);
    if (_t_posx_hdl == -1)
      printf("[ERROR] Eqn_IncompressibleNS3DCo::initialize_storage - failed to allocate _t_posx_hdl\n");

  }
  else {
    _u_posx_hdl = CoArrayManager::barrier_allocate();
    _v_posx_hdl = CoArrayManager::barrier_allocate();
    _w_posx_hdl = CoArrayManager::barrier_allocate();
    _t_posx_hdl = CoArrayManager::barrier_allocate();
  }  

  if (negx_image != -1) {
    Region3D uto   = _u                .region(-1)()();
    Region3D ufrom = _u.co(negx_image)->region(this->_nx-1)()();

    _u_negx_hdl = CoArrayManager::barrier_allocate(uto, ufrom);
    if (_u_negx_hdl  == -1)
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate _u_negx_hdl\n");

    Region3D vto   = _v                .region(-1)()();
    Region3D vfrom = _v.co(negx_image)->region(this->_nx-1)()();

    _v_negx_hdl = CoArrayManager::barrier_allocate(vto, vfrom);
    if (_v_negx_hdl == -1)
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate _v_negx_hdl\n");

    Region3D wto   = _w                .region(-1)()();
    Region3D wfrom = _w.co(negx_image)->region(this->_nx-1)()();

    _w_negx_hdl = CoArrayManager::barrier_allocate(wto, wfrom);
    if (_w_negx_hdl == -1)
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate _w_negx_hdl\n");

    Region3D tto   = _temp                .region(-1)()();
    Region3D tfrom = _temp.co(negx_image)->region(this->_nx-1)()();

    _t_negx_hdl = CoArrayManager::barrier_allocate(tto, tfrom);
    if (_t_negx_hdl == -1)
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate _t_negx_hdl\n");

  }
  else {
    _u_negx_hdl = CoArrayManager::barrier_allocate();
    _v_negx_hdl = CoArrayManager::barrier_allocate();
    _w_negx_hdl = CoArrayManager::barrier_allocate();
    _t_negx_hdl = CoArrayManager::barrier_allocate();
  }  

  do_halo_exchange_uvw();
  if (!apply_3d_mac_boundary_conditions_level1(_u, _v, _w, _local_bc, this->_hx, this->_hy, this->_hz)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on enforcing flow boundary conditions\n");
    return false;  
  }

  do_halo_exchange_t();
  if (!apply_3d_boundary_conditions_level1(_temp, _local_thermalbc, this->_hx, this->_hy, this->_hz)) {
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - failed on enforcing thermal boundary conditions\n");
    return false;  
  }

  _deriv_udt.clear_zero();
  _deriv_vdt.clear_zero();
  _deriv_wdt.clear_zero();
  _deriv_tempdt.clear_zero();

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
    printf("[ERROR] Eqn_IncompressibleNS3DCo::set_parameters - grid layouts do not all match\n");
    return false;  
  }

  ThreadManager::io_fence();
  ThreadManager::barrier();

  return true;
}

template<typename T>
double Eqn_IncompressibleNS3DCo<T>::get_max_stable_timestep() const
{
  T max_u, max_v, max_w;
  _u.co_reduce_maxabs(max_u);
  _v.co_reduce_maxabs(max_v);
  _w.co_reduce_maxabs(max_w);
  double ut = this->hx() / max_u;
  double vt = this->hy() / max_v;
  double wt = this->hz() / max_w;

  if (!check_float(ut)) ut = 1e10;
  if (!check_float(vt)) vt = 1e10;
  if (!check_float(wt)) wt = 1e10;

  double step = this->_cfl_factor * min3(ut, vt, wt);

  double minh = min3(this->hx(), this->hy(), this->hz());

  if (thermal_diffusion_coefficient() > 0)
    step = std::min(step, (minh * minh) / (6 * this->thermal_diffusion_coefficient()));
  if (viscosity_coefficient() > 0)
    step = std::min(step, (minh * minh) / (6 * this->viscosity_coefficient()));

  printf("Eqn_IncompressibleNS3DCo<T>::get_max_stable_timestep - return %f (%f %f %f)\n", step, ut, vt, wt);

  return step;
}

template<typename T>
void 
Eqn_IncompressibleNS3DCo<T>::add_thermal_force()
{
  // apply thermal force by adding -gkT to dvdt (let g = -1, k = 1, so this is just dvdt += T)
  //_advection_solver.deriv_vdt.linear_combination((T)1.0, _advection_solver.deriv_vdt, (T)1.0, _thermal_solver.phi);

  int tnx = this->nz();
  int tny = this->ny();
  int tnz = this->nx();

  int threadsInX = 16;
  int threadsInY = 2;
  int threadsInZ = 2;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  T direction_mult = this->_vertical_direction & DIR_NEGATIVE_FLAG ? 1 : -1;
  T *uvw = (this->_vertical_direction & DIR_XAXIS_FLAG) ? &_deriv_udt.at(0,0,0) :
           (this->_vertical_direction & DIR_YAXIS_FLAG) ? &_deriv_vdt.at(0,0,0) : &_deriv_wdt.at(0,0,0);

  KernelWrapper wrapper;
  wrapper.PreKernel();

  Eqn_IncompressibleNS3D_add_thermal_force<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(uvw, direction_mult * this->_gravity * this->_bouyancy, &_temp.at(0,0,0),
    _temp.xstride(), _temp.ystride(), _temp.stride(this->_vertical_direction), this->nx(), this->ny(), this->nz(), 
    blocksInY, 1.0f / (float)blocksInY);

  if (!wrapper.PostKernel("Eqn_IncompressibleNS3D_add_thermal_force"))
    this->add_error();

}

template<typename T>
bool Eqn_IncompressibleNS3DCo<T>::advance_one_step(double dt)
{
  this->clear_error();
  this->num_steps++;

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

  T ab_coeff = -dt*dt / (2 * this->_lastdt);

  // advance T 
  if (this->_time_step == TS_ADAMS_BASHFORD2 && this->_lastdt > 0) {
    check_ok(_temp.linear_combination((T)1.0, _temp, (T)(dt - ab_coeff), _deriv_tempdt));
    check_ok(_temp.linear_combination((T)1.0, _temp, (T)ab_coeff, _last_deriv_tempdt));
  } 
  else {
    check_ok(_temp.linear_combination((T)1.0, _temp, (T)dt, _deriv_tempdt));
  }

  do_halo_exchange_t();
  check_ok(apply_3d_boundary_conditions_level1_nocorners(_temp, this->_local_thermalbc, this->_hx, this->_hy, this->_hz));

  // advance u,v,w
  if (this->_time_step == TS_ADAMS_BASHFORD2 && this->_lastdt > 0) {
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
  if (this->_time_step == TS_ADAMS_BASHFORD2) {
    this->_lastdt = dt;
    _last_deriv_tempdt.copy_all_data(_deriv_tempdt);
    _last_deriv_udt.copy_all_data(_deriv_udt);
    _last_deriv_vdt.copy_all_data(_deriv_vdt);
    _last_deriv_wdt.copy_all_data(_deriv_wdt);
  }

  // enforce incompressibility - this enforces bc's before and after projection
  check_ok(_projection_solver.solve(this->_max_divergence));

  return !this->any_error();
}



template class Eqn_IncompressibleNS3DBase<float>;
template class Eqn_IncompressibleNS3D<float>;
template class Eqn_IncompressibleNS3DCo<float>;

#ifdef OCU_DOUBLESUPPORT
template class Eqn_IncompressibleNS3DBase<double>;
template class Eqn_IncompressibleNS3D<double>;
template class Eqn_IncompressibleNS3DCo<double>;
#endif


} // end namespace

