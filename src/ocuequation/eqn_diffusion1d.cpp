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
#include "ocuequation/eqn_diffusion1d.h"


namespace ocu {


Eqn_Diffusion1D::Eqn_Diffusion1D()
{
}

bool
Eqn_Diffusion1D::set_parameters(const Eqn_Diffusion1DParams &params)
{
  bool ok = true;

  // pre-validate
  if (params.initial_values.nx() != params.nx) {
    printf("[ERROR] Eqn_Diffusion1D::set_parameters - initial_values.nx() mismatch: %d != %d", params.initial_values.nx(), params.nx);
    ok = false;
  }

  if (!ok)
    return false;

  // set the values internally
  _diffusion_solver.h() = params.h;
  _diffusion_solver.left = params.left;
  _diffusion_solver.right = params.right;
  _diffusion_solver.initialize_storage(params.nx);
  _diffusion_solver.coefficient() = params.diffusion_coefficient;
  _diffusion_solver.density.copy_interior_data(params.initial_values);
  _host_density.init(params.nx, 0);

  // post-validate
  if (left_boundary().type != BC_NEUMANN && left_boundary().type != BC_DIRICHELET && left_boundary().type != BC_PERIODIC) {
    printf("[ERROR] Eqn_Diffusion1D::set_parameters - Invalid boundary left condition\n");
    ok = false;
  }
  if (right_boundary().type != BC_NEUMANN && right_boundary().type != BC_DIRICHELET && right_boundary().type != BC_PERIODIC) {
    printf("[ERROR] Eqn_Diffusion1D::set_parameters - Invalid boundary right condition\n");
    ok = false;
  }
  if ((left_boundary().type == BC_PERIODIC && right_boundary().type != BC_PERIODIC) ||
      (left_boundary().type != BC_PERIODIC && right_boundary().type == BC_PERIODIC)) {
    printf("[ERROR] Eqn_Diffusion1D::set_parameters - cannot have only one periodic boundary condition\n"); 
    ok = false;  
  }

  if (!check_float(h())) {
    printf("[ERROR] Eqn_Diffusion1D::set_parameters - bad h value\n"); 
    ok = false;
  }

  return ok;
}

double Eqn_Diffusion1D::get_max_stable_timestep() const
{
  return (h() * h()) / (2 * diffusion_coefficient());
}

bool Eqn_Diffusion1D::advance_one_step(double dt)
{
  clear_error();

  if (!check_float(dt)) {
    printf("[ERROR] Eqn_Diffusion1D::advance_one_step - bad dt value\n");
    return false;
  }

  _diffusion_solver.solve();
  
  // forward euler
  CHECK_OK(_diffusion_solver.density.linear_combination(1.0f, _diffusion_solver.density, dt, _diffusion_solver.deriv_densitydt));

  return !any_error();
}


void Eqn_Diffusion1D::copy_density_to_host()
{
  _host_density.copy_interior_data(_diffusion_solver.density);
}



}

