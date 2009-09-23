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

#ifndef __OCU_EQUATION_MULTIGRID_PRESSURE_3D_DEV_H__
#define __OCU_EQUATION_MULTIGRID_PRESSURE_3D_DEV_H__

#include <cmath>
#include <vector_types.h>


#include "ocuutil/boundary_condition.h"
#include "ocuutil/convergence.h"
#include "ocustorage/grid3d.h"
#include "ocustorage/grid2d.h"
#include "ocuequation/solver.h"

extern int DBG;

namespace ocu {


class Sol_MultigridPressure3DBase : public Solver
{
protected:
  //**** MEMBER VARIABLES ****
  int _num_levels;
  bool _failure;

  double *_h;
  double _omega;
  double _fx, _fy, _fz;

  int3 *_dim;

  //**** INTERNAL METHODS ****
  int   nx(int level) const { return _dim[level].x; }
  int   ny(int level) const { return _dim[level].y; }
  int   nz(int level) const { return _dim[level].z; }
  
  double get_h(int level) const { return _h[level]; }

  bool any_failures() const { return _failure; }
  void clear_failures()     { _failure = false; }
  void add_failure()        { _failure = true; }

  // translate between varying grid sizes vs uniform spacing with coefficients
  double hx(int level) const { return get_h(level)/sqrt(_fx); }
  double hy(int level) const { return get_h(level)/sqrt(_fy); }
  double hz(int level) const { return get_h(level)/sqrt(_fz); }

  int num_levels(int nx_val, int ny_val, int nz_val);
  double optimal_omega(double hx_val, double hy_val, double hz_val);
  double bc_diag_mod(const BoundaryCondition &bc, double factor) const;

  virtual bool do_fmg(double tolerance, int max_iter, double &result_l2, double &result_linf);
  virtual bool do_vcycle(double tolerance, int max_iter, double &result_l2, double &result_linf);
  bool initialize_base_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val);

  //**** OVERRIDES ****
  virtual void clear_zero(int level) = 0;
  virtual void apply_boundary_conditions(int level) = 0;
  virtual void relax(int level, int iterations) = 0;            
  virtual void restrict_residuals(int fine_level, int coarse_level, double *l2, double *linf) = 0;
  virtual void prolong(int coarse_level, int fine_level) = 0;
  //virtual void residual_norm(int level, double &l2, double &linf) = 0;

public:

  //**** PUBLIC STATE ****
  BoundaryConditionSet bc;
  int nu1;  // pre-smoothing steps
  int nu2;  // post-smoothing steps
  ConvergenceType convergence;

  //**** PUBLIC INTERFACE ****
  bool solve(double &residual, double tolerance = 1e-6, int max_iter = 4);

  Sol_MultigridPressure3DBase();
  ~Sol_MultigridPressure3DBase();

};

class Sol_MultigridPressureMixed3DDeviceD;

template<typename T>
class Sol_MultigridPressure3DDevice : public Sol_MultigridPressure3DBase {
  friend class Sol_MultigridPressureMixed3DDeviceD;
  //**** MEMBER VARIABLES ****
  Grid3DDevice<T> ** _r_grid;   // residual grids
  Grid3DDevice<T> ** _u_grid;   // solution grids
  Grid3DDevice<T> ** _b_grid;   // rhs grids
  
  Grid3DHost<T>  ** _hu_grid;   // host buffers for storing u grid for host-side relaxation
  Grid3DHost<T>  ** _hb_grid;   // host buffers for storing b grid for host-side relaxation

  Grid2DDevice<T> ** _diag_grid;   // 2d grid of diagonal coefficients grids
  Grid2DDevice<T> &get_diag(int level) { return *_diag_grid[level]; }
  void initialize_diag_grids();
  Grid3DDevice<T> _pressure; // results stored here, initial guess may be input here.
  bool _has_any_periodic_bc;

  //**** INTERNAL METHODS ****
  Grid3DDevice<T> &get_u(int level) { return *_u_grid[level]; }
  Grid3DDevice<T> &get_r(int level) { return *_r_grid[level]; }
  Grid3DDevice<T> &get_b(int level) { return *_b_grid[level]; }

  // this might be null
  Grid3DHost<T> *get_host_u(int level) { return _hu_grid[level]; }
  Grid3DHost<T> *get_host_b(int level) { return _hb_grid[level]; }

  bool invoke_kernel_relax(Grid3DDevice<T> &u_grid, Grid3DDevice<T> &b_grid, Grid2DDevice<T> &diag_grid, int red_black, double h);
  bool invoke_kernel_enforce_bc(Grid3DDevice<T> &u_grid, BoundaryConditionSet &bc, double hx, double hy, double hz);
  bool invoke_kernel_calculate_residual(Grid3DDevice<T> &u_grid, Grid3DDevice<T> &b_grid, Grid3DDevice<T> &r_grid, double h);
  bool invoke_kernel_restrict(Grid3DDevice<T> &r_grid, Grid3DDevice<T> &b_coarse_grid);
  bool invoke_kernel_prolong(Grid3DDevice<T> &u_coarse_grid, Grid3DDevice<T> &u_fine_grid);

  bool bind_tex_relax(Grid3DDevice<T> &u_grid, Grid3DDevice<T> &b_grid);
  bool unbind_tex_relax();

  bool bind_tex_calculate_residual(Grid3DDevice<T> &u_grid, Grid3DDevice<T> &b_grid);
  bool unbind_tex_calculate_residual();

  bool bind_tex_prolong(Grid3DDevice<T> &u_coarse_grid, Grid3DDevice<T> &u_fine_grid);
  bool unbind_tex_prolong();

  void do_cpu_solve(Grid3DHost<T> &h_u, Grid3DHost<T> &h_b, bool red_black, T h, T xpos_mod, T xneg_mod, T ypos_mod, T yneg_mod, T zpos_mod, T zneg_mod);
  void relax_on_host(int level, int iterations);

  //***** OVERRIDES ****
  virtual void apply_boundary_conditions(int level);
  virtual void relax(int level, int iterations);            
  virtual void restrict_residuals(int fine_level, int coarse_level, double *l2, double *linf);
  virtual void prolong(int coarse_level, int fine_level);
  virtual void clear_zero(int level);
  

public:

  //**** MANAGERS ****
  Sol_MultigridPressure3DDevice();
  ~Sol_MultigridPressure3DDevice();

  //**** PUBLIC INTERFACE ****
  bool initialize_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDevice<T> *rhs);
  
  Grid3DDevice<T>       &pressure()       { return _pressure; }
  const Grid3DDevice<T> &pressure() const { return _pressure; }
  
  // read-only access to internal grids
  const Grid3DDevice<T> &read_u(int level) const { return *_u_grid[level]; }
  const Grid3DDevice<T> &read_r(int level) const { return *_r_grid[level]; }
  const Grid3DDevice<T> &read_b(int level) const { return *_b_grid[level]; }

};



typedef Sol_MultigridPressure3DDevice<float> Sol_MultigridPressure3DDeviceF;
typedef Sol_MultigridPressure3DDevice<double> Sol_MultigridPressure3DDeviceD;



} // end namespace


#endif 

