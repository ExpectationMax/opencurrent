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

#include <cuda.h>
#include <cstdio>

#include "ocuutil/float_routines.h"
#include "ocuutil/timing_pool.h"
#include "ocuutil/timer.h"
#include "ocustorage/grid3dboundary.h"
#include "ocuequation/sol_mgpressure3d.h"



template<typename T>
__global__ void
Sol_MultigridPressure3DDevice_initialize_diag_grids(T *grid, int istride, int nx, int ny, 
      T laplace_diag, T xpos_mod, T xneg_mod, T ypos_mod, T yneg_mod)
{
  int i     = __umul24(blockIdx.x ,blockDim.x) + threadIdx.x;
  int j     = __umul24(blockIdx.y ,blockDim.y) + threadIdx.y;

  if (i < nx && j < ny) {
    T diag = laplace_diag;
    if (i==nx-1) diag += xpos_mod;
    if (i==0   ) diag += xneg_mod;
    if (j==ny-1) diag += ypos_mod;
    if (j==0   ) diag += yneg_mod;
    grid[i * istride + j] = diag;
  }
}





namespace ocu {



template<typename T>
bool 
Sol_MultigridPressure3DDevice<T>::invoke_kernel_enforce_bc(
  Grid3DDevice<T> &u_grid, 
  BoundaryConditionSet &bc_val,
  double hx_val, double hy_val, double hz_val)
{
  return apply_3d_boundary_conditions_level1_nocorners(u_grid, bc_val, hx_val, hy_val, hz_val);
}


template<typename T>
void 
Sol_MultigridPressure3DDevice<T>::clear_zero(int level)
{
  get_u(level).clear_zero();
}

template<typename T>
void 
Sol_MultigridPressure3DDevice<T>::do_cpu_solve(
  Grid3DHost<T> &h_u, Grid3DHost<T> &h_b, bool red_black, 
  T h, T xpos_mod, T xneg_mod, T ypos_mod, T yneg_mod, T zpos_mod, T zneg_mod)
{
  T laplace_diag = 2*_fx + 2*_fy +2*_fz;

  for (int i=0; i < h_u.nx(); i++) {
    T x_mod = (i==0) ? xneg_mod : (i == h_u.nx()-1) ? xpos_mod : 0;

    for (int j=0; j < h_u.ny(); j++) {
      T y_mod = (j==0) ? yneg_mod : (j == h_u.ny()-1) ? ypos_mod : 0;
      
      int k_start = (i+j+red_black)%2;
            
      T *u_ptr = &h_u.at(i,j,k_start);
      const T *b_ptr = &h_b.at(i,j,k_start);
      const T *uip1_ptr = u_ptr + h_u.xstride();
      const T *uim1_ptr = u_ptr - h_u.xstride();
      const T *ujp1_ptr = u_ptr + h_u.ystride();
      const T *ujm1_ptr = u_ptr - h_u.ystride();

      for (int k=k_start; k < h_u.nz(); k+=2) {

        T z_mod = (k==0) ? zneg_mod : (k == h_u.nz()-1) ? zpos_mod : 0;        
        
        T residual = (-h*h* (*b_ptr) - laplace_diag * (*u_ptr));
        residual += _fz * (*(u_ptr-1) + *(u_ptr+1));
        residual += _fy * (*ujp1_ptr  + *ujm1_ptr);
        residual += _fx * (*uip1_ptr  + *uim1_ptr);

        *u_ptr += ((T)_omega) * residual/(laplace_diag + x_mod + y_mod + z_mod);

        u_ptr += 2;
        uip1_ptr += 2;
        uim1_ptr += 2;
        ujp1_ptr += 2;
        ujm1_ptr += 2;
        b_ptr += 2;        
      }
    }
  }
}

template<typename T>
void 
Sol_MultigridPressure3DDevice<T>::relax_on_host(int level, int iterations)
{
  // copy grid & rhs to host
  // then solve

  if (!get_host_u(level) || !get_host_b(level)) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::relax_on_host - host buffers not allocated for level %d\n", level);
    add_failure();
    return;
  }

  Grid3DHost<T> &h_u = *get_host_u(level);
  Grid3DHost<T> &h_b = *get_host_b(level);

  // copy data
  h_u.copy_all_data(get_u(level));
  h_b.copy_all_data(get_b(level));

  BoundaryConditionSet h_bc = bc;
  
  if (level != 0) {
    h_bc.make_homogeneous();
  }

  KernelWrapper wrapper;
  wrapper.ToggleCPUTiming(true);
  wrapper.ToggleGPUTiming(false);
  wrapper.PreKernel();

  // solve here.
  for (int iters = 0; iters < iterations; iters++) {
    for (int red_black = 0; red_black < 2; red_black++) {      
      do_cpu_solve(h_u, h_b, red_black, get_h(level), 
            (T)bc_diag_mod(h_bc.xpos, _fx), (T)bc_diag_mod(h_bc.xneg, _fx), (T)bc_diag_mod(h_bc.ypos, _fy), 
            (T)bc_diag_mod(h_bc.yneg, _fy), (T)bc_diag_mod(h_bc.zpos, _fz), (T)bc_diag_mod(h_bc.zneg, _fz));
      
      apply_3d_boundary_conditions_level1_nocorners(h_u, h_bc, hx(level), hy(level), hz(level));
    }
  }  

  wrapper.PostKernel("Sol_MultigridPressure3DDevice::relax_on_host", nz(level));

  // write results back
  get_u(level).copy_all_data(h_u);

}

template<typename T>
void 
Sol_MultigridPressure3DDevice<T>::relax(int level, int iterations)
{ 
  if (level == _num_levels-1) {
    relax_on_host(level, iterations);
    return;
  }

  check_ok(bind_tex_relax(get_u(level), get_b(level)));
  for (int iters = 0; iters < iterations; iters++) {
    for (int red_black = 0; red_black < 2; red_black++) {      
      check_ok(invoke_kernel_relax(get_u(level), get_b(level), get_diag(level), red_black, get_h(level)));
      //NB: if there are periodic bc's, we must put this in the inner loop, otherwise we can save some work & move it outside
      if (_has_any_periodic_bc)
        apply_boundary_conditions(level);
    }
    
    if (!_has_any_periodic_bc)
      apply_boundary_conditions(level);
  }
  check_ok(unbind_tex_relax());

  if (any_error()) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::relax - failed at level %d\n", level);
  }
}

template<typename T>
void 
Sol_MultigridPressure3DDevice<T>::restrict_residuals(int fine_level, int coarse_level, double *l2, double *linf)
{
  check_ok(bind_tex_calculate_residual(get_u(fine_level), get_b(fine_level)));
  check_ok(invoke_kernel_calculate_residual(get_u(fine_level), get_b(fine_level), get_r(fine_level), get_h(fine_level)));
  check_ok(unbind_tex_calculate_residual());

  if (coarse_level != fine_level) {
    check_ok(invoke_kernel_restrict(get_r(fine_level), get_b(coarse_level)));
  }

  // calculate the norms if requested
  if (l2) {
    T residual_norm = 0;
    check_ok(get_r(fine_level).reduce_sqrsum(residual_norm));
    residual_norm /= (nx(fine_level) * ny(fine_level) * nz(fine_level));
    *l2 = sqrt(residual_norm);
  }

  if (linf) {
    T linf_norm = 0;
    check_ok(get_r(fine_level).reduce_maxabs(linf_norm));
    *linf = (double)linf_norm;
  }

  if (any_error()) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::restrict_residuals - failed at level %d -> %d\n", fine_level, coarse_level);
  }
}




template<typename T>
void 
Sol_MultigridPressure3DDevice<T>::prolong(int coarse_level, int fine_level)
{
  check_ok(bind_tex_prolong(get_u(coarse_level), get_u(fine_level)));
  check_ok(invoke_kernel_prolong(get_u(coarse_level), get_u(fine_level)));
  check_ok(unbind_tex_prolong());

  apply_boundary_conditions(fine_level);

  if (any_error()) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::prolong - failed at level %d -> %d\n", coarse_level, fine_level);
  }
}



template<typename T>
void 
Sol_MultigridPressure3DDevice<T>::apply_boundary_conditions(int level)
{
  if (level == 0) {
    if (!invoke_kernel_enforce_bc(get_u(level), bc, hx(level), hy(level), hz(level))) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::apply_boundary_conditions - failed at level %d \n", level);
      add_failure();
    }
  }
  else {
    // zero out all values

    BoundaryConditionSet bc_zero = bc;
    bc_zero.make_homogeneous();

    if (!invoke_kernel_enforce_bc(get_u(level), bc_zero, hx(level), hy(level), hz(level))) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::apply_boundary_conditions - failed at level %d \n", level);
      add_failure();
    }
  }
}



template<typename T>
Sol_MultigridPressure3DDevice<T>::Sol_MultigridPressure3DDevice()
{
  _r_grid = 0;
  _u_grid = 0;
  _b_grid = 0;
  _has_any_periodic_bc = false;
}

template<typename T>
Sol_MultigridPressure3DDevice<T>::~Sol_MultigridPressure3DDevice()
{
  for (int l=0; l < _num_levels; l++) {
    if (l > 0) {
      delete _u_grid[l];
      delete _b_grid[l];
    }
    delete _r_grid[l];
    delete _diag_grid[l];
    delete _hu_grid[l];
    delete _hb_grid[l];
  }

  delete[] _u_grid;
  delete[] _b_grid;
  delete[] _hu_grid;
  delete[] _hb_grid;
  delete[] _r_grid;
  delete[] _diag_grid;
}

template<typename T>
void
Sol_MultigridPressure3DDevice<T>::initialize_diag_grids()
{
  _diag_grid = new Grid2DDevice<T> *[_num_levels];
  int level;
  for (level=0; level < _num_levels; level++) {
    _diag_grid[level] = new Grid2DDevice<T>();
    _diag_grid[level]->init(nx(level), ny(level),0,0);

    // call kernel to init them
    int tnx = nx(level);
    int tny = ny(level);

    int threadsInX = 16;
    int threadsInY = 16;

    int blocksInX = (tnx+threadsInX-1)/threadsInX;
    int blocksInY = (tny+threadsInY-1)/threadsInY;

    dim3 Dg = dim3(blocksInX, blocksInY);
    dim3 Db = dim3(threadsInX, threadsInY);

    Sol_MultigridPressure3DDevice_initialize_diag_grids<<<Dg, Db>>>(&get_diag(level).at(0,0), get_diag(level).xstride(), nx(level), ny(level), 
      (T)(2*_fx + 2*_fy +2*_fz), (T)bc_diag_mod(bc.xpos, _fx), (T)bc_diag_mod(bc.xneg, _fx), (T)bc_diag_mod(bc.ypos, _fy), (T)bc_diag_mod(bc.yneg, _fy));
  }
}


template<typename T>
bool 
Sol_MultigridPressure3DDevice<T>::initialize_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDevice<T> *rhs)
{
  if (!initialize_base_storage(nx_val, ny_val, nz_val, hx_val, hy_val, hz_val)) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - error in grid calculation\n");
    return false;
  }

  _has_any_periodic_bc = bc.xpos.check_type(BC_PERIODIC) || bc.xneg.check_type(BC_PERIODIC) ||
                         bc.ypos.check_type(BC_PERIODIC) || bc.yneg.check_type(BC_PERIODIC) ||
                         bc.zpos.check_type(BC_PERIODIC) || bc.zneg.check_type(BC_PERIODIC);

  int b_gx = rhs->gx();
  int b_gy = rhs->gy();
  int b_gz = rhs->gz();

  if (b_gx < 1 || b_gy < 1 || b_gz < 1) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - rhs has invalid ghost cells (%d,%d,%d), must be >= 1\n", b_gx, b_gy, b_gz);
    return false;
  }

  if (rhs->nx() != nx_val, rhs->ny() != ny_val, rhs->nz() != nz_val) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - rhs dimension mismatch (%d,%d,%d) != (%d,%d,%d)\n", rhs->nx(), rhs->ny(), rhs->nz(), nx_val, ny_val, nz_val);
    return false;
  }

  // do allocation and initialization
  _r_grid = new Grid3DDevice<T> *[_num_levels];
  _u_grid = new Grid3DDevice<T> *[_num_levels];
  _b_grid = new Grid3DDevice<T> *[_num_levels];
  _hu_grid = new Grid3DHost<T> *[_num_levels];
  _hb_grid = new Grid3DHost<T> *[_num_levels];
  
  // init these ptrs to null
  memset(_hu_grid, 0, sizeof(Grid3DHost<T> *) * _num_levels);
  memset(_hb_grid, 0, sizeof(Grid3DHost<T> *) * _num_levels);

  int b_padx = rhs->paddingx();
  int b_pady = rhs->paddingy();
  int b_padz = rhs->paddingz();

  _b_grid[0] = rhs;
  _r_grid[0] = new Grid3DDevice<T>();
  if (!_r_grid[0]->init(nx_val, ny_val, nz_val,b_gx,b_gy,b_gz, b_padx, b_pady, b_padz)) { 
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _r_grid[0]\n");
    return false;
  }

  _u_grid[0] = &_pressure;
  if (!_u_grid[0]->init(nx_val, ny_val, nz_val,b_gx,b_gy,b_gz, b_padx, b_pady, b_padz)) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _u_grid[0]\n");
    return false;
  }

  _u_grid[0]->clear_zero();

  int level;
  for (level=1; level < _num_levels; level++) {

    _u_grid[level] = new Grid3DDevice<T>();
    _b_grid[level] = new Grid3DDevice<T>();
    _r_grid[level] = new Grid3DDevice<T>();

    if (!_u_grid[level]->init(nx(level), ny(level), nz(level),1,1,1)) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _u_grid[%d]\n", level);
      return false;
    }
    if (!_b_grid[level]->init(nx(level), ny(level), nz(level),1,1,1)) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _b_grid[%d]\n", level);
      return false;
    }
    if (!_r_grid[level]->init(nx(level), ny(level), nz(level),1,1,1)) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _r_grid[%d]\n", level);
      return false;
    }

    if (level == _num_levels-1) {
      _hu_grid[level] = new Grid3DHost<T>();
      _hb_grid[level] = new Grid3DHost<T>();
      _hu_grid[level]->init_congruent(*_u_grid[level]);
      _hb_grid[level]->init_congruent(*_b_grid[level]);
    }

  }

  // post-validation
  for (level=1; level < _num_levels; level++) {
    if (!get_u(level).check_interior_dimension_match(get_r(level)) || !get_u(level).check_layout_match(get_r(level)) ||
        !get_u(level).check_interior_dimension_match(get_b(level)) || !get_u(level).check_layout_match(get_b(level))) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - grid dimension mismatch\n");
      return false;
    }
  }

  initialize_diag_grids();


  return true;
}


template void Sol_MultigridPressure3DDevice<float>::initialize_diag_grids();
template bool Sol_MultigridPressure3DDevice<float>::initialize_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDevice<float> *rhs);
template Sol_MultigridPressure3DDevice<float>::Sol_MultigridPressure3DDevice();
template Sol_MultigridPressure3DDevice<float>::~Sol_MultigridPressure3DDevice();
template void Sol_MultigridPressure3DDevice<float>::apply_boundary_conditions(int level);
template void Sol_MultigridPressure3DDevice<float>::prolong(int coarse_level, int fine_level);
template void Sol_MultigridPressure3DDevice<float>::restrict_residuals(int fine_level, int coarse_level, double *l2, double *linf);
template void Sol_MultigridPressure3DDevice<float>::do_cpu_solve(Grid3DHost<float> &h_u, Grid3DHost<float> &h_b, bool red_black, float h, 
                                                             float xpos_mod, float xneg_mod, float ypos_mod, float yneg_mod, float zpos_mod, float zneg_mod);
template void Sol_MultigridPressure3DDevice<float>::relax_on_host(int level, int iterations);
template void Sol_MultigridPressure3DDevice<float>::clear_zero(int level);
template bool Sol_MultigridPressure3DDevice<float>::invoke_kernel_enforce_bc(Grid3DDevice<float> &u_grid, BoundaryConditionSet &bc_val, double hx_val, double hy_val, double hz_val);


#ifdef OCU_DOUBLESUPPORT

template void Sol_MultigridPressure3DDevice<double>::initialize_diag_grids();
template bool Sol_MultigridPressure3DDevice<double>::initialize_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDevice<double> *rhs);
template Sol_MultigridPressure3DDevice<double>::Sol_MultigridPressure3DDevice();
template Sol_MultigridPressure3DDevice<double>::~Sol_MultigridPressure3DDevice();
template void Sol_MultigridPressure3DDevice<double>::apply_boundary_conditions(int level);
template void Sol_MultigridPressure3DDevice<double>::prolong(int coarse_level, int fine_level);
template void Sol_MultigridPressure3DDevice<double>::restrict_residuals(int fine_level, int coarse_level, double *l2, double *linf);
template void Sol_MultigridPressure3DDevice<double>::do_cpu_solve(Grid3DHost<double> &h_u, Grid3DHost<double> &h_b, bool red_black, double h, 
                                                             double xpos_mod, double xneg_mod, double ypos_mod, double yneg_mod, double zpos_mod, double zneg_mod);
template void Sol_MultigridPressure3DDevice<double>::relax_on_host(int level, int iterations);
template void Sol_MultigridPressure3DDevice<double>::clear_zero(int level);
template bool Sol_MultigridPressure3DDevice<double>::invoke_kernel_enforce_bc(Grid3DDevice<double> &u_grid, BoundaryConditionSet &bc_val, double hx_val, double hy_val, double hz_val);


#endif // OCU_DOUBLESUPPORT

} // end namespace

