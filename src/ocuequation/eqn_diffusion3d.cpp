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
Eqn_Diffusion3D<T>::Eqn_Diffusion3D()
{
  _hx = _hy = _hz = 0;
  _nx = _ny = _nz = 0;
}

template<typename T>
bool Eqn_Diffusion3D<T>::set_parameters(const Eqn_Diffusion3DParams<T> &params)
{
  _bc = params.bc;
  _nx = params.nx;
  _ny = params.ny;
  _nz = params.nz;
  _hx = params.hx;
  _hy = params.hy;
  _hz = params.hz;
  
  if (!check_float(_hx) || !check_float(_hy) || !check_float(_hz) || _hx <= 0 || _hy <= 0 || _hz <= 0) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - illegal hx, hy, hz values (%f %f %f)\n", _hx, _hy, _hz);
    return false;
  }

  if (!_density.init(_nx, _ny, _nz, 1, 1, 1, 0, 0, 0)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not initialize _density\n");
    return false;
  }

  if (!_deriv_densitydt.init(_nx, _ny, _nz, 1, 1, 1, 0, 0, 0)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not initialize _deriv_densitydt\n");
    return false;
  }

  if (!_density.copy_all_data(params.initial_values)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not set initial values\n");
    return false;
  }

  if (!_diffusion_solver.initialize_storage(_nx, _ny, _nz, _hx, _hy, _hz, &_density, &_deriv_densitydt)) {
    printf("[ERROR] Eqn_Diffusion3D::set_parameters - could not initialize diffusion solver\n");
    return false;
  }
  _diffusion_solver.coefficient = params.diffusion_coefficient;

  return true;
}

template<typename T>
double Eqn_Diffusion3D<T>::get_max_stable_timestep() const
{
  double minh = min3(hx(), hy(), hz());
  return (minh * minh) / (6 * diffusion_coefficient());
}



template<typename T>
bool Eqn_Diffusion3D<T>::advance_one_step(double dt)
{
  if (!apply_3d_boundary_conditions_level1_nocorners(_density, _bc, _hx, _hy, _hz)) {
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



template class Eqn_Diffusion3D<float>;

#ifdef OCU_DOUBLESUPPORT
template class Eqn_Diffusion3D<double>;
#endif

}

