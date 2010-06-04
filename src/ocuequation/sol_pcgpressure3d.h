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


#ifndef __FSSYSTEM_PCG_PRESSURE_3D_DEV_H__
#define __FSSYSTEM_PCG_PRESSURE_3D_DEV_H__

#include <cmath>
#include <vector_types.h>


#include "ocuutil/boundary_condition.h"
#include "ocuutil/convergence.h"
#include "ocustorage/grid3d.h"
#include "ocustorage/grid2d.h"
#include "ocuequation/solver.h"
#include "ocuequation/sol_mgpressure3d.h"

namespace ocu {

enum PreconditionerType {
  PRECOND_NONE,
  PRECOND_JACOBI,
  PRECOND_MULTIGRID,
  PRECOND_BFBT,
};

template<typename T>
class Sol_PCGPressure3DDevice : public Solver
{
protected:
  //**** MEMBER VARIABLES ****

  double _fx, _fy, _fz;
  double _h;
  int _nx, _ny, _nz;
  Grid3DDevice<T> _pressure;
  Grid3DDevice<T> _d_r; // residual in CG iteration, also used as RHS for multigrid
  Grid3DDevice<T> *_rhs;
  Grid3DDevice<T> *_coefficient; // coefficient for varying-coeff Poisson problem.
  Sol_MultigridPressure3DDevice<T> _mg;

  //**** INTERNAL METHODS ****
  // translate between varying grid sizes vs uniform spacing with coefficients
  double hx() const { return _h/sqrt(_fx); }
  double hy() const { return _h/sqrt(_fy); }
  double hz() const { return _h/sqrt(_fz); }

  virtual bool do_pcg(double tolerance, int max_iter, double &result_l2, double &result_linf);
  virtual bool do_cg(double tolerance, int max_iter, double &result_l2, double &result_linf);
  
  // src isn't const because we enforce boundary conditions on it first
  void invoke_kernel_diag_preconditioner(Grid3DDevice<T> &dst, const Grid3DDevice<T> &src);
  void invoke_kernel_apply_laplacian(Grid3DDevice<T> &dst, Grid3DDevice<T> &src);
  T    invoke_kernel_dot_product(const Grid3DDevice<T> &a, const Grid3DDevice<T> &b);

  void apply_preconditioner(Grid3DDevice<T> &dst, const Grid3DDevice<T> &src);

public:

  //**** PUBLIC STATE ****
  BoundaryConditionSet bc;
  ConvergenceType convergence;
  PreconditionerType preconditioner;
  int multigrid_cycles;
  bool multigrid_use_fmg; // false -> use vcycles only

  //**** MANAGERS ****
  Sol_PCGPressure3DDevice();
  ~Sol_PCGPressure3DDevice();

  //**** PUBLIC INTERFACE ****
  bool solve(double &residual, double tolerance = 1e-6, int max_iter = 1000);
  bool initialize_storage(int nx_val, int ny_val, int nz_val, float hx_val, float hy_val, float hz_val, Grid3DDevice<T> *rhs, Grid3DDevice<T> *coefficient);
  
  Grid3DDevice<T>       &pressure()       { return _pressure; }
  const Grid3DDevice<T> &pressure() const { return _pressure; }
  Grid3DDevice<T>       &rhs     ()       { return *_rhs; }
  const Grid3DDevice<T> &rhs     () const { return *_rhs; }
  Grid3DDevice<T>       &coefficient()       { return *_coefficient; }
  const Grid3DDevice<T> &coefficient() const { return *_coefficient; }

};




typedef Sol_PCGPressure3DDevice<float> Sol_PCGPressure3DDeviceF;
typedef Sol_PCGPressure3DDevice<double> Sol_PCGPressure3DDeviceD;



} // end namespace


#endif 

