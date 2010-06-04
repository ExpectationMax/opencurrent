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
#include "ocustorage/coarray.h"
#include "ocuequation/eqn_diffusion1d.h"

namespace ocu {





double Eqn_Diffusion1DCoF::get_max_stable_timestep() const
{
  return (h() * h()) / (2 * diffusion_coefficient());
}

bool Eqn_Diffusion1DCoF::advance_one_step(double dt)
{
  clear_error();

  if (!check_float(dt)) {
    printf("[ERROR] Eqn_Diffusion1D::advance_one_step - bad dt value %f\n", dt);
    return false;
  }


  // make sure previous step computation has finished
  ThreadManager::compute_fence();

  // boundary exchange
  CoArrayManager::barrier_exchange(_left_handle);
  CoArrayManager::barrier_exchange(_right_handle);

  //make sure exchange has finished
  ThreadManager::io_fence();

  // update derivatives
  _diffusion_solver.solve();
  
  // forward euler
  CHECK_OK(_density.linear_combination(1.0f, _density, dt, _diffusion_solver.deriv_densitydt));

  ThreadManager::barrier();

  return !any_error();
}

bool Eqn_Diffusion1DCoF::set_parameters(const Eqn_Diffusion1DParams &params)
{
  left = params.left;
  right = params.right;

  if (!_density.init(params.nx, 1)) {
    printf("[ERROR] Eqn_Diffusion1DCoF::set_parameters - failed to initialize density\n");
    return false;
  }

  if (params.initial_values.nx() != params.nx) {
    printf("[ERROR] Eqn_Diffusion1DCoF::set_parameters - initial_values.nx() mismatch: %d != %d", params.initial_values.nx(), params.nx);
    return false;
  }

  if (!_density.copy_interior_data(params.initial_values)) {
    printf("[ERROR] Eqn_Diffusion1DCoF::set_parameters - failed to copy initial_value");
    return false;
  }

  _diffusion_solver.h() = params.h;
  _diffusion_solver.coefficient() = params.diffusion_coefficient;
  _diffusion_solver.initialize_storage(params.nx, &_density);

  if (!check_float(h())) {
    printf("[ERROR] Eqn_Diffusion1DCoF::set_parameters - bad h value %f\n", h()); 
    return false;
  }


  ThreadManager::barrier();

  int tid = ThreadManager::this_image();
  int num_images = ThreadManager::num_images();

  int left_tid = (tid - 1 + num_images) % num_images;
  int right_tid = (tid + 1) % num_images;

  Region1D left_from = _density.co(left_tid)->region(nx()-1);
  Region1D left_to   = _density.region(-1);
  Region1D right_from = _density.co(right_tid)->region(0);
  Region1D right_to   = _density.region(nx());

  _left_handle = CoArrayManager::barrier_allocate(left_to, left_from);
  _right_handle = CoArrayManager::barrier_allocate(right_to, right_from);

  return true;
}

Eqn_Diffusion1DCoF::~Eqn_Diffusion1DCoF()
{
  CoArrayManager::barrier_deallocate(_left_handle);
  CoArrayManager::barrier_deallocate(_right_handle);
}


}  // end namespace

