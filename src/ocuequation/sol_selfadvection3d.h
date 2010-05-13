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

#ifndef __OCU_EQUATION_SELF_ADVECTION_3D_DEV_H__
#define __OCU_EQUATION_SELF_ADVECTION_3D_DEV_H__

#include "ocuequation/solver.h"
#include "ocustorage/grid3d.h"
#include "ocuutil/boundary_condition.h"
#include "ocuutil/interpolation.h"

namespace ocu {


//! \class Sol_SelfAdvection3DDevice 
//! \brief Calculates advection of a vector valued staggered incompressible vector field by itself.
//! For now, it will use first-order upwinding and I'll experiment with different gpu implementations.

template<typename T>
class Sol_SelfAdvection3DDevice : public Solver
{

  //**** MEMBER VARIABLES ****
  int _nx, _ny, _nz;
  double _hx, _hy, _hz;

  bool solve_naive();
  bool solve_tex();

  bool bind_textures();
  bool unbind_textures();

public:

  //**** MANAGERS ****
  Sol_SelfAdvection3DDevice();
  ~Sol_SelfAdvection3DDevice();

  //**** PUBLIC STATE ****
  Grid3DDevice<T> *u, *v, *w;
  Grid3DDevice<T> *deriv_udt;
  Grid3DDevice<T> *deriv_vdt;
  Grid3DDevice<T> *deriv_wdt;
  InterpolationType interp_type;

  //**** PUBLIC INTERFACE ****

  // u,v,w must have ghost cells filled in before this is called!
  bool solve();

  bool initialize_storage(int nx, int ny, int nz, double hx, double hy, double hz, Grid3DDevice<T> *u_val, Grid3DDevice<T> *v_val, Grid3DDevice<T> *w_val, 
    Grid3DDevice<T> *deriv_udt_val, Grid3DDevice<T> *deriv_vdt_val, Grid3DDevice<T> *deriv_wdt_val);

  int    nx() const { return _nx; }
  int    ny() const { return _ny; }
  int    nz() const { return _nz; }
  double hx() const { return _hx; }
  double hy() const { return _hy; }
  double hz() const { return _hz; }
};





} // end namespace

#endif

