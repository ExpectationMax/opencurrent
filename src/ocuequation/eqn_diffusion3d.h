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
#include "ocustorage/coarray.h"

namespace ocu {

template<typename T>
class Eqn_Diffusion3DBaseParams : public Parameters 
{
public:

  Eqn_Diffusion3DBaseParams() {
    nx = ny = nz = 0;
    hx = hy = hz = 0;
    diffusion_coefficient = 0;
  }

  int nx, ny, nz;
  double hx, hy, hz;
  T diffusion_coefficient;
};



template<typename T>
class Eqn_Diffusion3DBase : public Equation {
protected:

  Eqn_Diffusion3DBase();

  double _hx, _hy, _hz;
  int   _nx, _ny, _nz;

public:


  bool set_base_parameters(const Eqn_Diffusion3DBaseParams<T> &params);

  int nx() const { return _nx; }
  int ny() const { return _ny; }
  int nz() const { return _nz; }
  double hx() const { return _hx; }
  double hy() const { return _hy; }
  double hz() const { return _hz; }
};


template<typename T>
class Eqn_Diffusion3DParams : public Eqn_Diffusion3DBaseParams<T>
{
public:
  Eqn_Diffusion3DParams() {  }

  BoundaryConditionSet bc;
  Grid3DHost<T> initial_values;
};


template<typename T>
class Eqn_Diffusion3D : public Eqn_Diffusion3DBase<T> {

  //**** MEMBER VARIABLES ****
  Sol_LaplacianCentered3DDevice<T> _diffusion_solver;
  Grid3DDevice<T>              _density;
  Grid3DDevice<T>              _deriv_densitydt;
  BoundaryConditionSet         _bc;

public:

  //**** MANAGERS ****
  Eqn_Diffusion3D();

  //**** OVERRIDES ****
  bool advance_one_step(double dt);
  double get_max_stable_timestep() const;

  //**** PUBLIC INTERFACE ****
  bool set_parameters(const Eqn_Diffusion3DParams<T> &params);

  const Grid3DDevice<T> &density() const { return _density; }
  double   diffusion_coefficient() const { return _diffusion_solver.coefficient; }
};


template<typename T>
class Eqn_Diffusion3DCoParams : public Eqn_Diffusion3DParams<T>
{
public:
  Eqn_Diffusion3DCoParams() {
    negx_image = -1;
    posx_image = -1;
  }

  int negx_image;
  int posx_image;
};

template<typename T>
class Eqn_Diffusion3DCo : public Eqn_Diffusion3DBase<T> {

  //**** MEMBER VARIABLES ****
  Sol_LaplacianCentered3DDevice<T> _diffusion_solver;
  Grid3DDeviceCo<T>                _density;
  Grid3DDevice<T>                  _deriv_densitydt;
  BoundaryConditionSet _ext_bc; // these are external boundary conditions.
  BoundaryConditionSet _local_bc; // local boundary conditions for this image

  int _posx_handle;
  int _negx_handle;

  int _left_image;
  int _right_image;

  // layout across gpus is split over x axis

public:

  //**** MANAGERS ****
  Eqn_Diffusion3DCo(const char *id);
  ~Eqn_Diffusion3DCo();

  //**** OVERRIDES ****
  bool advance_one_step(double dt);
  double get_max_stable_timestep() const;

  //**** PUBLIC INTERFACE ****
  bool set_parameters(const Eqn_Diffusion3DCoParams<T> &params);

  const Grid3DDeviceCo<T> &density() const { return _density; }
  double   diffusion_coefficient() const { return _diffusion_solver.coefficient; }
};




} // end namespace

#endif

