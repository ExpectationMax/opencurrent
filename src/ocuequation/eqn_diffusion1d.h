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

#ifndef __OCU_EQUATION_EQN_DIFFUSION_1D_H__
#define __OCU_EQUATION_EQN_DIFFUSION_1D_H__

#include "ocuequation/equation.h"
#include "ocuequation/sol_laplaciancent1d.h"
#include "ocuequation/parameters.h"

namespace ocu {

class Eqn_Diffusion1DParams : public Parameters 
{
public:

  int nx;
  double diffusion_coefficient;
  BoundaryCondition left, right;
  double h;
  Grid1DHostF initial_values;
};

class Eqn_Diffusion1D : public Equation {

  //**** MEMBER VARIABLES ****
  Sol_LaplacianCentered1DDevice _diffusion_solver;
  Grid1DHostF               _host_density;

public:

  //**** MANAGERS ****
  Eqn_Diffusion1D();


  //**** OVERRIDES ****
  double get_max_stable_timestep() const;
  bool advance_one_step(double dt);

  //**** PUBLIC INTERFACE ****
  bool set_parameters(const Eqn_Diffusion1DParams &params);
  void copy_density_to_host();

  int                nx()                    const { return _diffusion_solver.nx(); }
  const Grid1DHostF &density()               const { return _host_density; }
  double             diffusion_coefficient() const { return _diffusion_solver.coefficient(); }
  const BoundaryCondition &left_boundary()   const { return _diffusion_solver.left; }
  const BoundaryCondition &right_boundary()  const { return _diffusion_solver.right; }
  double             h()                     const { return _diffusion_solver.h(); }
};






} // end namespace

#endif

