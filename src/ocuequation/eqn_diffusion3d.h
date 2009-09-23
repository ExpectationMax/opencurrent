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

#ifndef __OCU_EQUATION_EQN_DIFFUSION_3D_H__
#define __OCU_EQUATION_EQN_DIFFUSION_3D_H__

#include "ocuequation/equation.h"
#include "ocuequation/sol_laplaciancent3d.h"
#include "ocuequation/parameters.h"

namespace ocu {

template<typename T>
class Eqn_Diffusion3DParams : public Parameters 
{
public:

  Eqn_Diffusion3DParams() {
    nx = ny = nz = 0;
    hx = hy = hz = 0;
    diffusion_coefficient = 0;
  }

  int nx, ny, nz;
  double hx, hy, hz;
  T diffusion_coefficient;
  BoundaryConditionSet bc;
  Grid3DHost<T> initial_values;
};


template<typename T>
class Eqn_Diffusion3D : public Equation {

  //**** MEMBER VARIABLES ****
  Sol_LaplacianCentered3DDevice<T> _diffusion_solver;
  Grid3DDevice<T>              _density;
  Grid3DDevice<T>              _deriv_densitydt;
  double _hx, _hy, _hz;
  int   _nx, _ny, _nz;
  BoundaryConditionSet _bc;

public:

  //**** MANAGERS ****
  Eqn_Diffusion3D();

  //**** OVERRIDES ****
  double get_max_stable_timestep() const;
  bool advance_one_step(double dt);

  //**** PUBLIC INTERFACE ****
  bool set_parameters(const Eqn_Diffusion3DParams<T> &params);

  int nx() const { return _nx; }
  int ny() const { return _ny; }
  int nz() const { return _nz; }
  double hx() const { return _hx; }
  double hy() const { return _hy; }
  double hz() const { return _hz; }

  const Grid3DDevice<T> &density() const { return _density; }
  double   diffusion_coefficient() const { return _diffusion_solver.coefficient; }
};






} // end namespace

#endif

