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
#include "ocuequation/eqn_diffusion3d.h"

namespace ocu {

template<typename T>
Eqn_Diffusion3DBase<T>::Eqn_Diffusion3DBase()
{
  _hx = _hy = _hz = 0;
  _nx = _ny = _nz = 0;
}


template<typename T>
bool Eqn_Diffusion3DBase<T>::set_base_parameters(const Eqn_Diffusion3DBaseParams<T> &params)
{
  _nx = params.nx;
  _ny = params.ny;
  _nz = params.nz;
  _hx = params.hx;
  _hy = params.hy;
  _hz = params.hz;
  
  if (!check_float(_hx) || !check_float(_hy) || !check_float(_hz) || _hx <= 0 || _hy <= 0 || _hz <= 0) {
    printf("[ERROR] Eqn_Diffusion3DBase::set_base_parameters - illegal hx, hy, hz values (%f %f %f)\n", _hx, _hy, _hz);
    return false;
  }

  return true;
}

template<typename T>
Eqn_Diffusion3D<T>::Eqn_Diffusion3D()
{
}


template<typename T>
bool Eqn_Diffusion3D<T>::set_parameters(const Eqn_Diffusion3DParams<T> &params)
{
  if (!Eqn_Diffusion3DBase<T>::set_base_parameters(params)) {
    return false;
  }

  _bc = params.bc;
  
  if (!_density.init(this->_nx, this->_ny, this->_nz, 1, 1, 1, 0, 0, 0)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not initialize _density\n");
    return false;
  }

  if (!_deriv_densitydt.init(this->_nx, this->_ny, this->_nz, 1, 1, 1, 0, 0, 0)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not initialize _deriv_densitydt\n");
    return false;
  }

  if (!_density.copy_all_data(params.initial_values)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not set initial values\n");
    return false;
  }

  if (!_diffusion_solver.initialize_storage(this->_nx, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_density, &_deriv_densitydt)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not initialize diffusion solver\n");
    return false;
  }
  _diffusion_solver.coefficient = params.diffusion_coefficient;

  return true;
}

template<typename T>
double Eqn_Diffusion3D<T>::get_max_stable_timestep() const
{
  double minh = min3(this->hx(), this->hy(), this->hz());
  return (minh * minh) / (6 * diffusion_coefficient());
}


template<typename T>
bool Eqn_Diffusion3D<T>::advance_one_step(double dt)
{
  if (!apply_3d_boundary_conditions_level1_nocorners(_density, _bc, this->_hx, this->_hy, this->_hz)) {
    printf("[ERROR] Eqn_Diffusion3D::advance_one_step - error enforcing boundary conditions\n");
    return false;
  }


  if (!_deriv_densitydt.clear_zero()) {
    printf("[ERROR] Eqn_Diffusion3D::advance_one_step - error setting derivative to zero\n");
    return false;
  }

  if (!_diffusion_solver.solve()) {
    printf("[ERROR] Eqn_Diffusion3D::advance_one_step - error calculating laplacian\n");
    return false;
  }

  if (!_density.linear_combination((T)1.0, _density, (T)dt, _deriv_densitydt)) {
    printf("[ERROR] Eqn_Diffusion3D::advance_one_step - error calculating linear_combination\n");
    return false;
  }

  return true;
}



template<typename T>
Eqn_Diffusion3DCo<T>::Eqn_Diffusion3DCo(const char *id) :
  _density((std::string(id) + std::string(".density")).c_str())
{
  _posx_handle = -1;
  _negx_handle = -1;
}

template<typename T>
Eqn_Diffusion3DCo<T>::~Eqn_Diffusion3DCo()
{
  CoArrayManager::barrier_deallocate(_posx_handle);
  CoArrayManager::barrier_deallocate(_negx_handle);
}


template<typename T>
bool Eqn_Diffusion3DCo<T>::set_parameters(const Eqn_Diffusion3DCoParams<T> &params)
{
  if (!Eqn_Diffusion3DBase<T>::set_base_parameters(params)) {
    return false;
  }

  if (!_density.init(this->_nx, this->_ny, this->_nz, 1, 1, 1, 0, 0, 0)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not initialize _density\n");
    return false;
  }

  if (!_deriv_densitydt.init(this->_nx, this->_ny, this->_nz, 1, 1, 1, 0, 0, 0)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not initialize _deriv_densitydt\n");
    return false;
  }

  if (!_density.copy_all_data(params.initial_values)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not set initial values\n");
    return false;
  }

  if (!_diffusion_solver.initialize_storage(this->_nx, this->_ny, this->_nz, this->_hx, this->_hy, this->_hz, &_density, &_deriv_densitydt)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not initialize diffusion solver\n");
    return false;
  }
  _diffusion_solver.coefficient = params.diffusion_coefficient;

  int tid = ThreadManager::this_image();
  int num_images = ThreadManager::num_images();
  _left_image = (tid - 1 + num_images) % num_images;
  _right_image = (tid + 1) % num_images;

  _ext_bc = params.bc;
  _local_bc = _ext_bc;

  // convert from ext_bc to local_bc...
  if (tid == 0) {
    if (_local_bc.xneg.type == BC_PERIODIC) 
      _local_bc.xneg.type = BC_NONE;
    else
      _left_image = -1;
  }
  
  if (tid == num_images - 1) {
    if (_local_bc.xpos.type == BC_PERIODIC) 
      _local_bc.xpos.type = BC_NONE;
    else
      _right_image = -1;
  }
  
  if (!(tid == 0 || tid == num_images - 1)) {
    _local_bc.xneg.type = BC_NONE;
    _local_bc.xpos.type = BC_NONE;
  }

  if (_left_image != -1) {
    Region3D negx_from = _density.co(_left_image)->region(this->nx()-1)()();
    Region3D negx_to   = _density.region(-1)()();
    _negx_handle = CoArrayManager::barrier_allocate(negx_to, negx_from);
  }
  else {
    _negx_handle = -1;
    CoArrayManager::barrier_allocate();
  }

  if (_right_image != -1) {
    Region3D posx_from = _density.co(_right_image)->region(0)()();
    Region3D posx_to   = _density.region(this->nx())()();    
    _posx_handle = CoArrayManager::barrier_allocate(posx_to, posx_from);
  }
  else {
    _posx_handle = -1;
    CoArrayManager::barrier_allocate();
  }

  return true;
}

template<typename T>
double Eqn_Diffusion3DCo<T>::get_max_stable_timestep() const
{
  double minh = min3(this->hx(), this->hy(), this->hz());
  return (minh * minh) / (6 * diffusion_coefficient());
}

template<typename T>
bool Eqn_Diffusion3DCo<T>::advance_one_step(double dt)
{
  this->clear_error();

  // make sure previous step finished
  ThreadManager::compute_fence();

  check_ok(CoArrayManager::barrier_exchange(this->_negx_handle));
  check_ok(CoArrayManager::barrier_exchange(this->_posx_handle));

  check_ok(apply_3d_boundary_conditions_level1_nocorners(_density, _local_bc, this->_hx, this->_hy, this->_hz));
  check_ok(_deriv_densitydt.clear_zero());

  ThreadManager::io_fence();

  check_ok(_diffusion_solver.solve());
  check_ok(_density.linear_combination((T)1.0, _density, (T)dt, _deriv_densitydt));

  return !this->any_error();
}



template class Eqn_Diffusion3DBase<float>;
template class Eqn_Diffusion3D<float>;
template class Eqn_Diffusion3DCo<float>;

#ifdef OCU_DOUBLESUPPORT
template class Eqn_Diffusion3DBase<double>;
template class Eqn_Diffusion3D<double>;
template class Eqn_Diffusion3DCo<double>;
#endif

}

