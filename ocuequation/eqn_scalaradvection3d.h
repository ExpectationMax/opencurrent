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

#ifndef __OCU_EQUATION_EQN_SCALAR_ADVECTION_3D_H__
#define __OCU_EQUATION_EQN_SCALAR_ADVECTION_3D_H__


#include "ocuutil/boundary_condition.h"
#include "ocuutil/interpolation.h"
#include "ocuequation/equation.h"
#include "ocuequation/parameters.h"
#include "ocuequation/sol_passiveadvection3d.h"

namespace ocu {



template<typename T>
class Eqn_ScalarAdvection3DParams : public Parameters
{
public:
  Eqn_ScalarAdvection3DParams() {
    nx = ny = nz = 0;
    hx = hy = hz = 0;
    advection_scheme = IT_ERROR;
  }

  int nx, ny, nz;
  double hx, hy, hz;
  Grid3DHost<T> initial_values;
  Grid3DHost<T> u, v, w;
  BoundaryConditionSet bc;
  InterpolationType advection_scheme;

  void init_grids(int nx_val, int ny_val, int nz_val, bool pinned=true) {
    nx = nx_val;
    ny = ny_val;
    nz = nz_val;
    u.init(nx+1,ny,nz,1,1,1,pinned,0,1,1);
    v.init(nx,ny+1,nz,1,1,1,pinned,1,0,1);
    w.init(nx,ny,nz+1,1,1,1,pinned,1,1,0);
    initial_values.init(nx,ny,nz,1,1,1,pinned,1,1,1);
  }
};

template<typename T>
class Eqn_ScalarAdvection3D : public Equation
{
  //**** MEMBER VARIABLES ****
  Sol_PassiveAdvection3DDevice<T> _advection;

public:

  //**** PUBLIC STATE ****
  Grid3DDevice<T> u, v, w;
  Grid3DDevice<T> phi;
  Grid3DDevice<T> deriv_phidt;
  BoundaryConditionSet bc;

  //**** MANAGERS ****
  Eqn_ScalarAdvection3D() { }

  //**** PUBLIC INTERFACE ****
  bool set_parameters(const Eqn_ScalarAdvection3DParams<T> &params);

  double get_max_stable_timestep() const;
  bool advance_one_step(double dt);


  int    nx()  const { return _advection.nx(); }
  int    ny()  const { return _advection.ny(); }
  int    nz()  const { return _advection.nz(); }
  double hx()  const { return _advection.hx(); }
  double hy()  const { return _advection.hy(); }
  double hz()  const { return _advection.hz(); }
};




typedef Eqn_ScalarAdvection3DParams<float> Eqn_ScalarAdvection3DParamsF;
typedef Eqn_ScalarAdvection3DParams<double> Eqn_ScalarAdvection3DParamsD;

typedef Eqn_ScalarAdvection3D<float> Eqn_ScalarAdvection3DF;
typedef Eqn_ScalarAdvection3D<double> Eqn_ScalarAdvection3DD;


} // end namespace


#endif

