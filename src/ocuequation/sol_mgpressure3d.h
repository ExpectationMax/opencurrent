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
#include <string>


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

  enum RelaxOrder {
    RO_RED_BLACK,
    RO_BLACK_RED,
    RO_SYMMETRIC
  };

  //**** MEMBER VARIABLES ****
  int _num_levels;

  double *_h;
  double _omega;
  double _fx, _fy, _fz;

  int3 *_dim;

  //**** INTERNAL METHODS ****
  int   nx(int level) const { return _dim[level].x; }
  int   ny(int level) const { return _dim[level].y; }
  int   nz(int level) const { return _dim[level].z; }
  
  double get_h(int level) const { return _h[level]; }

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
  virtual void relax(int level, int iterations, RelaxOrder order) = 0;            
  virtual void restrict_residuals(int fine_level, int coarse_level, double *l2, double *linf) = 0;
  virtual void prolong(int coarse_level, int fine_level) = 0;
  //virtual void residual_norm(int level, double &l2, double &linf) = 0;

public:

  //**** PUBLIC STATE ****
  BoundaryConditionSet bc;
  int nu1;  // pre-smoothing steps
  int nu2;  // post-smoothing steps
  ConvergenceType convergence;
  bool make_symmetric_operator; // make the application of smoothing in the vcycles a symmetric linear operator

  //**** PUBLIC INTERFACE ****
  bool solve(double &residual, double tolerance = 1e-6, int max_iter = 4);
  bool run_vcycles(int count);
  bool run_fmg(int count);

  Sol_MultigridPressure3DBase();
  ~Sol_MultigridPressure3DBase();

};

class Sol_MultigridPressureMixed3DDeviceD;

template<typename T>
class Sol_MultigridPressure3DDevice : public Sol_MultigridPressure3DBase {
protected:
  friend class Sol_MultigridPressureMixed3DDeviceD;
  //**** MEMBER VARIABLES ****
  Grid3DDevice<T> ** _r_grid;   // residual grids
  Grid3DDevice<T> ** _u_grid;   // solution grids
  Grid3DDevice<T> ** _b_grid;   // rhs grids
  
  Grid3DHost<T>  ** _hu_grid;   // host buffers for storing u grid for host-side relaxation
  Grid3DHost<T>  ** _hb_grid;   // host buffers for storing b grid for host-side relaxation

  Grid2DDevice<T> ** _diag_grid;   // 2d grid of diagonal coefficients grids
  Grid3DDevice<T> _pressure; // results stored here, initial guess may be input here.
  bool _update_bc_between_colors;

  //**** INTERNAL METHODS ****
  Grid2DDevice<T> &get_diag(int level) { return *_diag_grid[level]; }
  void initialize_diag_grids();

  Grid3DDevice<T> &get_u(int level) { return *_u_grid[level]; }
  Grid3DDevice<T> &get_r(int level) { return *_r_grid[level]; }
  Grid3DDevice<T> &get_b(int level) { return *_b_grid[level]; }

  // this might be null
  Grid3DHost<T> *get_host_u(int level) { return _hu_grid[level]; }
  Grid3DHost<T> *get_host_b(int level) { return _hb_grid[level]; }

  bool invoke_kernel_relax(Grid3DDevice<T> &u_grid, Grid3DDevice<T> &b_grid, Grid2DDevice<T> &diag_grid, int red_black, double h, const BoundaryConditionSet &this_bc);
  bool invoke_kernel_enforce_bc(Grid3DDevice<T> &u_grid, const BoundaryConditionSet &bc, double hx, double hy, double hz);
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
  void relax_on_host(int level, int iterations, RelaxOrder order);

  virtual BoundaryConditionSet get_bc_at_level(int level) const;
  virtual void apply_host_boundary_conditions(Grid3DHost<T> &h_u, int level);

  //***** OVERRIDES ****
  virtual void apply_boundary_conditions(int level);
  virtual void relax(int level, int iterations, RelaxOrder order); 
  virtual void restrict_residuals(int fine_level, int coarse_level, double *l2, double *linf);
  virtual void prolong(int coarse_level, int fine_level);
  virtual void clear_zero(int level);
  

public:

  bool disable_relax_on_host;

  //**** MANAGERS ****
  Sol_MultigridPressure3DDevice();
  ~Sol_MultigridPressure3DDevice();

  //**** PUBLIC INTERFACE ****
  // TODO: split this out so that subclass can reuse code
  virtual bool initialize_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDevice<T> *rhs);
  
  Grid3DDevice<T>       &pressure()       { return _pressure; }
  const Grid3DDevice<T> &pressure() const { return _pressure; }
  
  // read-only access to internal grids
  const Grid3DDevice<T> &read_u(int level) const { return *_u_grid[level]; }
  const Grid3DDevice<T> &read_r(int level) const { return *_r_grid[level]; }
  const Grid3DDevice<T> &read_b(int level) const { return *_b_grid[level]; }

};

template<typename T>
class Sol_MultigridPressure3DDeviceCo : public Sol_MultigridPressure3DDevice<T>
{
  std::string _id;
  Grid3DDeviceCo<T> _co_pressure;
  Grid3DDevice<T> _u_prolong;  // on thread 0, this is treated as u[cutoff] for purposes of prolong
  Grid3DDevice<T> _b_restrict; // on thread 0, this is treated as b[cutoff] for purposes of restrict
  
  std::vector<BoundaryConditionSet> _local_bc;  // boundary conditions for this image

  std::vector<int> _u_posx_hdl; // send u values to posx nbr
  std::vector<int> _u_negx_hdl; // send u values to negx nbr
  std::vector<int> _hu_posx_hdl; // send hu values to posx nbr
  std::vector<int> _hu_negx_hdl; // send hu values to negx nbr
  int _gather_b_hdl;  // send b grids to image 0
  int _scatter_u_hdl; // pull u grids from image 0

  int _multi_thread_cutoff_level; // level below which only image 0 is active

  Grid3DDeviceCo<T> &get_co_u(int level) { return * ((Grid3DDeviceCo<T> *)this->_u_grid[level]); }
  Grid3DDeviceCo<T> &get_co_r(int level) { return * ((Grid3DDeviceCo<T> *)this->_r_grid[level]); }
  Grid3DDeviceCo<T> &get_co_b(int level) { 
    if (level == 0) 
      printf("[WARNING] Sol_MultigridPressure3DDeviceCo::get_co_b(0) - level 0 b grid is not a co-array\n");
    return * ((Grid3DDeviceCo<T> *)this->_b_grid[level]); 
  }

  Grid3DHostCo<T> *get_host_co_u(int level) { return ((Grid3DHostCo<T> *)this->_hu_grid[level]); }


  // nx is always the resolution at which relax(level) will run
  int   local_nx(int level) const { return level < this->_multi_thread_cutoff_level ? this->_dim[level].x : this->_dim[level].x / ThreadManager::num_images(); }
  virtual BoundaryConditionSet get_bc_at_level(int level) const;
  virtual void apply_host_boundary_conditions(Grid3DHost<T> &h_u, int level);

  int num_active_images(int level) const {
    return level < this->_multi_thread_cutoff_level ? ThreadManager::num_images() : 1;
  }

  //***** OVERRIDES ****
  virtual void apply_boundary_conditions(int level);
  virtual void relax(int level, int iterations, Sol_MultigridPressure3DBase::RelaxOrder order); 
  virtual void restrict_residuals(int fine_level, int coarse_level, double *l2, double *linf);
  virtual void prolong(int coarse_level, int fine_level);
  virtual void clear_zero(int level);
  
public:

  int grid_size_for_cutoff;

  //**** MANAGERS ****
  Sol_MultigridPressure3DDeviceCo(const char *id);
  ~Sol_MultigridPressure3DDeviceCo();

  Grid3DDeviceCo<T>       &co_pressure()       { return _co_pressure; }
  const Grid3DDeviceCo<T> &co_pressure() const { return _co_pressure; }

  virtual bool initialize_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDevice<T> *rhs);

};

typedef Sol_MultigridPressure3DDevice<float> Sol_MultigridPressure3DDeviceF;
typedef Sol_MultigridPressure3DDevice<double> Sol_MultigridPressure3DDeviceD;

typedef Sol_MultigridPressure3DDeviceCo<float> Sol_MultigridPressure3DDeviceCoF;
typedef Sol_MultigridPressure3DDeviceCo<double> Sol_MultigridPressure3DDeviceCoD;


} // end namespace


#endif 

