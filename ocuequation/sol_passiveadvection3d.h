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

#ifndef __OCU_EQUATION_PASSIVE_ADVECTION_3D_DEV_H__
#define __OCU_EQUATION_PASSIVE_ADVECTION_3D_DEV_H__

#include "ocuutil/interpolation.h"
#include "ocuequation/solver.h"
#include "ocustorage/grid3d.h"
#include "ocuutil/boundary_condition.h"

namespace ocu {


//! \class PassiveAdvection3DDevice 
//! \brief Calculates passive advection of a scalar value by a staggered incompressible vector field.
//! For now, it will use first-order upwinding and I'll experiment with different gpu implementations.
template<typename T>
class Sol_PassiveAdvection3DDevice : public Solver
{

  //**** MEMBER VARIABLES ****
  int _nx, _ny, _nz;
  double _hx, _hy, _hz;

public:

  //**** MANAGERS ****
  Sol_PassiveAdvection3DDevice();

  //**** PUBLIC STATE ****
  Grid3DDevice<T> *u, *v, *w;
  Grid3DDevice<T> *phi;
  Grid3DDevice<T> *deriv_phidt;
  InterpolationType interp_type;

  //**** PUBLIC INTERFACE ****
  bool solve();

  bool initialize_storage(int nx, int ny, int nz, double hx, double hy, double hz, Grid3DDevice<T> *u_val, Grid3DDevice<T> *v_val, Grid3DDevice<T> *w_val, 
    Grid3DDevice<T> *phi_val, Grid3DDevice<T> *deriv_phidt_val);

  int    nx() const { return _nx; }
  int    ny() const { return _ny; }
  int    nz() const { return _nz; }
  double hx() const { return _hx; }
  double hy() const { return _hy; }
  double hz() const { return _hz; }
};





} // end namespace

#endif

