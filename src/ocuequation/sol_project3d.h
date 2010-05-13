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

#ifndef __OCU_EQUATION_PROJECT_3D_DEV_H__
#define __OCU_EQUATION_PROJECT_3D_DEV_H__

#include "ocuutil/boundary_condition.h"
#include "ocuequation/solver.h"
#include "ocuequation/sol_mgpressure3d.h"
#include "ocuequation/sol_divergence3d.h"
#include "ocuequation/sol_gradient3d.h"

namespace ocu {


class Sol_ProjectDivergence3DBase : public Solver {
protected:

  //*** MEMBER VARIABLES ****
  int _nx, _ny, _nz;
  double _hx, _hy, _hz;

  //**** INTERNAL METHODS ****
  BoundaryCondition convert_bc_to_poisson_eqn(const BoundaryCondition &bc) const;

  bool initialize_base_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DUntyped *u, Grid3DUntyped *v, Grid3DUntyped *w);

public:

  //**** PUBLIC STATE ****
  BoundaryConditionSet bc;


  //**** MANAGERS ****
  Sol_ProjectDivergence3DBase() { }
  ~Sol_ProjectDivergence3DBase() { }
};



template<typename T>
class Sol_ProjectDivergence3DDeviceStorage : public Sol_ProjectDivergence3DBase {

protected:

  bool initialize_device_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDevice<T> *u, Grid3DDevice<T> *v, Grid3DDevice<T> *w);

public:

  //**** PUBLIC STATE ****
  Grid3DDevice<T> *u, *v, *w; // updated in place
  Grid3DDevice<T> divergence;

  //**** MANAGERS ****
  Sol_ProjectDivergence3DDeviceStorage() { }
  ~Sol_ProjectDivergence3DDeviceStorage() { }

};


template<typename T>
class Sol_ProjectDivergence3DDevice : public Sol_ProjectDivergence3DDeviceStorage<T> {

public:

  //**** PUBLIC STATE ****
  Sol_MultigridPressure3DDevice<T> pressure_solver;
  Sol_Divergence3DDevice<T> divergence_solver;
  Sol_Gradient3DDevice<T> gradient_solver;

  //**** MANAGERS ****
  Sol_ProjectDivergence3DDevice() { }
  ~Sol_ProjectDivergence3DDevice() { }

  //**** PUBLIC INTERFACE ****
  bool solve(double tolerance=1e-5);
  bool solve_divergence_only();

  bool initialize_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDevice<T> *u, Grid3DDevice<T> *v, Grid3DDevice<T> *w);
};


typedef Sol_ProjectDivergence3DDevice<float> Sol_ProjectDivergence3DDeviceF;
typedef Sol_ProjectDivergence3DDevice<double> Sol_ProjectDivergence3DDeviceD;





} // end namespace

#endif

