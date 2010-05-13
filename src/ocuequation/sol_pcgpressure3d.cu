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


#include "ocuutil/float_routines.h"
#include "ocuutil/thread.h"
#include "ocustorage/grid3dboundary.h"
#include "ocuequation/sol_pcgpressure3d.h"
#include "ocustorage/gridnetcdf.h"

//#define WRITE_ERROR

template<typename T>
__global__ void Sol_PCGPressure3DDevice_ptwise_mult(T *result, const T *a, const T *b,
  int xstride, int ystride, 
  int nx, int ny, int nz, int blocksInY, float invBlocksInY)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  int idx = __mul24(i, xstride) + __mul24(j,ystride) + k;

  if (i < nx && j < ny && k < nz) {
    result[idx] = a[idx] * b[idx];
  }
}

template<typename T>
__global__
void  Sol_PCGPressure3DDevice_diag_preconditioner(
  T *dst, const T *src, const T *coeff, int xstride, int ystride, 
  int nx, int ny, int nz, 
  double fx_div_hsq, double fy_div_hsq, double fz_div_hsq,
  int blocksInY, float invBlocksInY)
{
  unsigned int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;
  
  int idx = __mul24(i,xstride) + __mul24(j,ystride) + k;

  // divide by diagonal element from weighted laplacian
  if (i < nx && j < ny && k < nz) {
    T center_src = src[idx];
    T coeff_ctr = coeff[idx];
    T coeff_km1 = coeff[idx - 1      ];
    T coeff_kp1 = coeff[idx + 1      ];
    T coeff_jm1 = coeff[idx - ystride];
    T coeff_jp1 = coeff[idx + ystride];
    T coeff_im1 = coeff[idx - xstride];
    T coeff_ip1 = coeff[idx + xstride];

    T diagonal = 
        fz_div_hsq*((T)0.5)*(coeff_km1 + coeff_ctr) +
        fz_div_hsq*((T)0.5)*(coeff_kp1 + coeff_ctr) +
        fy_div_hsq*((T)0.5)*(coeff_jm1 + coeff_ctr) +
        fy_div_hsq*((T)0.5)*(coeff_jp1 + coeff_ctr) +
        fx_div_hsq*((T)0.5)*(coeff_im1 + coeff_ctr) +
        fx_div_hsq*((T)0.5)*(coeff_ip1 + coeff_ctr);

    dst[idx] = center_src / diagonal;
  }
}



template<typename T>
__global__
void  Sol_PCGPressure3DDevice_apply_laplacian(
  T *dst, const T *src, const T *coeff, int xstride, int ystride, 
  int nx, int ny, int nz, 
  double fx_div_hsq, double fy_div_hsq, double fz_div_hsq, double diag, 
  int blocksInY, float invBlocksInY)
{
  unsigned int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;
  
  int idx = __mul24(i,xstride) + __mul24(j,ystride) + k;

  // this is actually the inverse of the laplacian
  if (i < nx && j < ny && k < nz) {

    T center_src = src[idx];
    T coeff_ctr = coeff[idx];
    T coeff_km1 = coeff[idx - 1      ];
    T coeff_kp1 = coeff[idx + 1      ];
    T coeff_jm1 = coeff[idx - ystride];
    T coeff_jp1 = coeff[idx + ystride];
    T coeff_im1 = coeff[idx - xstride];
    T coeff_ip1 = coeff[idx + xstride];


    dst[idx] = 
      -(fz_div_hsq*(src[idx - 1      ] - center_src) * ((T)0.5)*(coeff_km1 + coeff_ctr) +
        fz_div_hsq*(src[idx + 1      ] - center_src) * ((T)0.5)*(coeff_kp1 + coeff_ctr) +
        fy_div_hsq*(src[idx - ystride] - center_src) * ((T)0.5)*(coeff_jm1 + coeff_ctr) +
        fy_div_hsq*(src[idx + ystride] - center_src) * ((T)0.5)*(coeff_jp1 + coeff_ctr) +
        fx_div_hsq*(src[idx - xstride] - center_src) * ((T)0.5)*(coeff_im1 + coeff_ctr) +
        fx_div_hsq*(src[idx + xstride] - center_src) * ((T)0.5)*(coeff_ip1 + coeff_ctr));
  }
}




template<typename T>
__global__
void  Sol_PCGPressure3DDevice_apply_laplacian_uniform(
  T *dst, const T *src, int xstride, int ystride, 
  int nx, int ny, int nz, 
  double fx_div_hsq, double fy_div_hsq, double fz_div_hsq, double diag, 
  int blocksInY, float invBlocksInY)
{
  unsigned int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;
  
  int idx = __mul24(i,xstride) + __mul24(j,ystride) + k;

  // this is actually the inverse of the laplacian
  if (i < nx && j < ny && k < nz) {
    //perturb one diagonal element to make matrix non-singular, at index 0,0,0
    T diag_mod = (idx == 0) ? diag : 0;

    dst[idx] = 
     -(fz_div_hsq*(src[idx - 1      ] + src[idx + 1      ]) + 
       fy_div_hsq*(src[idx - ystride] + src[idx + ystride]) +
       fx_div_hsq*(src[idx - xstride] + src[idx + xstride]) - (diag+diag_mod)*src[idx]);
  }
}


namespace ocu {


template<typename T>
Sol_PCGPressure3DDevice<T>::Sol_PCGPressure3DDevice()
{
  _fx = _fy = _fz = 0;
  _h = 1;
  _nx = _ny = _nz = 0;
  convergence = CONVERGENCE_L2;
  preconditioner = PRECOND_BFBT;
  multigrid_use_fmg = true;
  multigrid_cycles = 2;

  _mg.convergence = CONVERGENCE_NONE_CALC_L2LINF;
}

template<typename T>
Sol_PCGPressure3DDevice<T>::~Sol_PCGPressure3DDevice()
{
}




template<typename T>
void 
Sol_PCGPressure3DDevice<T>::invoke_kernel_apply_laplacian(Grid3DDevice<T> &dst, Grid3DDevice<T> &src)
{
  check_ok(apply_3d_boundary_conditions_level1_nocorners(src, bc, hx(), hy(), hz()));

  T fx_div_hsq = _fx / (_h*_h);
  T fy_div_hsq = _fy / (_h*_h);
  T fz_div_hsq = _fz / (_h*_h);
  T diag = 2 * (_fx + _fy + _fz) / (_h*_h);

  // pt-wise laplacian
  int tnx = src.nz();
  int tny = src.ny();
  int tnz = src.nx();

  int threadsInX = 32;
  int threadsInY = 4;
  int threadsInZ = 1;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  PreKernel();
#if 1
  Sol_PCGPressure3DDevice_apply_laplacian<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(&dst.at(0,0,0), &src.at(0,0,0), &_coefficient->at(0,0,0), src.xstride(), src.ystride(), 
    src.nx(), src.ny(), src.nz(), 
    fx_div_hsq, fy_div_hsq, fz_div_hsq, diag,
    blocksInY, 1.0f/(float)blocksInY);
#else
  Sol_PCGPressure3DDevice_apply_laplacian_uniform<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(&dst.at(0,0,0), &src.at(0,0,0), src.xstride(), src.ystride(), 
    src.nx(), src.ny(), src.nz(), 
    fx_div_hsq, fy_div_hsq, fz_div_hsq, diag,
    blocksInY, 1.0f/(float)blocksInY);
#endif
  PostKernel("Sol_PCGPressure3DDevice_apply_laplacian");
}

template<typename T>
void    
Sol_PCGPressure3DDevice<T>::invoke_kernel_diag_preconditioner(Grid3DDevice<T> &dst, const Grid3DDevice<T> &src)
{
  T fx_div_hsq = _fx / (_h*_h);
  T fy_div_hsq = _fy / (_h*_h);
  T fz_div_hsq = _fz / (_h*_h);

  // pt-wise laplacian
  int tnx = src.nz();
  int tny = src.ny();
  int tnz = src.nx();

  int threadsInX = 32;
  int threadsInY = 4;
  int threadsInZ = 1;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  PreKernel();
  Sol_PCGPressure3DDevice_diag_preconditioner<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(&dst.at(0,0,0), &src.at(0,0,0), &_coefficient->at(0,0,0), src.xstride(), src.ystride(), 
    src.nx(), src.ny(), src.nz(), 
    fx_div_hsq, fy_div_hsq, fz_div_hsq,
    blocksInY, 1.0f/(float)blocksInY);
  PostKernel("Sol_PCGPressure3DDevice_diag_preconditioner");
}


template<typename T>
T    
Sol_PCGPressure3DDevice<T>::invoke_kernel_dot_product(const Grid3DDevice<T> &a, const Grid3DDevice<T> &b)
{
  Grid3DDevice<T> temp;
  check_ok(temp.init_congruent(a));

  // pt-wise mult
  int tnx = a.nz();
  int tny = a.ny();
  int tnz = a.nx();

  int threadsInX = 32;
  int threadsInY = 4;
  int threadsInZ = 1;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  PreKernel();
  Sol_PCGPressure3DDevice_ptwise_mult<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(&temp.at(0,0,0), &a.at(0,0,0), &b.at(0,0,0), a.xstride(), a.ystride(), 
    a.nx(), a.ny(), a.nz(), 
    blocksInY, 1.0f/(float)blocksInY);
  PostKernel("Sol_PCGPressure3DDevice_ptwise_mult");

  T result;
  temp.reduce_sum(result);
  return result;
}

template<typename T>
void
Sol_PCGPressure3DDevice<T>::apply_preconditioner(Grid3DDevice<T> &dst, const Grid3DDevice<T> &src)
{

  if (preconditioner == PRECOND_JACOBI) {
    // jacobi
    // if this were uniform coefficient, the preconditioner would just be:
    // T diag = 2 * (_fx + _fy + _fz) / (_h*_h);
    // dst.linear_combination(1.0/diag, src);
    invoke_kernel_diag_preconditioner(dst, src);
  }

  else if (preconditioner == PRECOND_MULTIGRID) {
    // multigrid
    _mg.pressure().clear_zero();

    if (multigrid_use_fmg)
      _mg.run_fmg(multigrid_cycles);
    else
      _mg.run_vcycles(multigrid_cycles);

    // have to switch sign b/c L is negative-definite.
    dst.linear_combination(-1, _mg.pressure());
  }
  else if (preconditioner == PRECOND_BFBT) {

    _mg.pressure().clear_zero();

    if (multigrid_use_fmg)
      _mg.run_fmg(multigrid_cycles);
    else
      _mg.run_vcycles(multigrid_cycles);

    // have to switch sign b/c L is negative-definite.
    dst.linear_combination(-1, _mg.pressure());
    invoke_kernel_diag_preconditioner(dst, dst);
  }
  else {
    // no preconditioner
    // identity
    dst.copy_all_data(src);
  }

}


template<typename T>
bool 
Sol_PCGPressure3DDevice<T>::do_pcg(double tolerance, int max_iter, double &result_l2, double &result_linf)
{
#ifdef WRITE_ERROR
  NetCDFGrid3DWriter  writer;
  writer.open("error.nc", _nx, _ny, _nz);
  writer.define_variable("error", NC_DOUBLE, GS_CENTER_POINT);
  Grid3DHost<T> h_r;
  h_r.init_congruent(_d_r);
#endif //WRITE_ERROR


  // must enforce bc's on coefficient to match bc's on pressure
  check_ok(apply_3d_boundary_conditions_level1_nocorners(*_coefficient, bc, hx(), hy(), hz()));

  // solve L * pressure = rhs
  // where L is the discrete Laplacian with appropriate boundary conditions

  Grid3DDevice<T> d_y, d_d;
  check_ok(d_y.init_congruent(_pressure));
  check_ok(d_d.init_congruent(_pressure));        

  // y <- L * pressure
  invoke_kernel_apply_laplacian(d_y, _pressure); 

  //       r <- rhs - L*pressure
  // (aka) r <- rhs - y
  _d_r.linear_combination((T)1, *_rhs, (T)-1, d_y);

#ifdef WRITE_ERROR
  h_r.copy_all_data(_d_r);
  writer.add_data("error", h_r, 0);
#endif

  //  d = Minv * r
  apply_preconditioner(d_d, _d_r);

  // delta0 = <r,d>
  T delta0 = invoke_kernel_dot_product(_d_r, d_d);
  T deltanew = delta0;

  printf("[INFO] Sol_PCGPressure3DDevice<T>::do_pcg - initial residual norm %g\n", sqrt(delta0));

  int iteration_count = 0;
  bool converged = (iteration_count > max_iter) || (delta0 < tolerance*tolerance);

  while (!converged) {

    // y <- L * d
    invoke_kernel_apply_laplacian(d_y, d_d); 

    //     alpha <- <r,d>/<L * d,d>
    // aka alpha <- <r,d>/<d,y>
    T d_dot_y = invoke_kernel_dot_product(d_d, d_y);

    T alpha = deltanew / d_dot_y;
    
    // pressure <- pressure + alpha * d
    _pressure.linear_combination((T) 1, _pressure, alpha, d_d);
    
    // every 25 iterations, restart residual
    if (iteration_count % 25 == 0) {
      // y <- L * pressure
      invoke_kernel_apply_laplacian(d_y, _pressure);             
      //       r <- rhs - L*pressure
      // (aka) r <- rhs - y
      _d_r.linear_combination((T)1, *_rhs, (T)-1, d_y);
    }
    else {
      //     r <- r - alpha * (L * p) 
      // aka r <- r - alpha * y 
      _d_r.linear_combination((T)1, _d_r, -alpha, d_y);
    }

#ifdef WRITE_ERROR
    h_r.copy_all_data(_d_r);
    writer.add_data("error", h_r, iteration_count+1);
#endif

    //  s = Minv * r (store 's' in d_y here)

    // what are d_r and d_y before and after this?
    apply_preconditioner(d_y, _d_r);


    // beta <- <r_{i+1},s_{i+1}>/<r,s> 
    T deltaold = deltanew;
    deltanew = invoke_kernel_dot_product(_d_r, d_y);
    T beta = deltanew / deltaold;
    
    // d <- s + beta*d
    d_d.linear_combination(beta, d_d, (T)1, d_y);

    printf("%d: %g\n", iteration_count, sqrt(deltanew));

    // check convergence (iter > max_iter) or (sqrt(<r,d>) < tolerance)
    iteration_count++;
    converged = (iteration_count > max_iter) || (deltanew < tolerance*tolerance);
  }

  bool ok;

  if(deltanew <= tolerance*tolerance) {
    printf("[INFO] Sol_PCGPressure3DDevice<T>::do_pcg - Converged to %f residual in %d iterations\n", sqrt(deltanew), iteration_count);
    ok = true;
  }
  else {
    printf("[WARNING] Sol_PCGPressure3DDevice<T>::do_pcg - Failed to converge within %d iterations (achieved %f residual)\n",max_iter,sqrt(deltanew));
    ok = false;
  }

#ifdef WRITE_ERROR
  writer.close();
#endif

  T r_linf;
  _d_r.reduce_maxabs(r_linf);
  result_linf = r_linf;
  result_l2 = sqrt(deltanew);
  return ok;
}

template<typename T>
bool 
Sol_PCGPressure3DDevice<T>::do_cg(double tolerance, int max_iter, double &result_l2, double &result_linf)
{
  // solve L * pressure = rhs
  // where L is the discrete Laplacian with appropriate boundary conditions

  Grid3DDevice<T> d_y, d_d;
  check_ok(d_y.init_congruent(_pressure));
  check_ok(d_d.init_congruent(_pressure));        


  // y <- L * pressure
  invoke_kernel_apply_laplacian(d_y, _pressure); 

  //       r <- rhs - L*pressure
  // (aka) r <- rhs - y
  _d_r.linear_combination((T)1, *_rhs, (T)-1, d_y);

  //  d_d = d_r
  d_d.copy_all_data(_d_r);

	
  // delta0 = <r,r>
  T delta0;
  _d_r.reduce_sqrsum(delta0);
  T deltanew = delta0;

  printf("[INFO] Sol_PCGPressure3DDevice<T>::do_pcg - initial residual norm %g\n", sqrt(delta0));

  int iteration_count = 0;
  bool converged = (iteration_count > max_iter) || (delta0 < tolerance*tolerance);

  while (!converged) {

    // y <- L * d
    invoke_kernel_apply_laplacian(d_y, d_d); 

    //     alpha <- <r,r>/<L * d,d>
    // aka alpha <- <r,r>/<d,y>
    T d_dot_y = invoke_kernel_dot_product(d_d, d_y);
    T alpha = deltanew / d_dot_y;
    
    // pressure <- pressure + alpha * d
    _pressure.linear_combination((T) 1, _pressure, alpha, d_d);
    
    // every 25 iterations, restart residual
    if (iteration_count % 25 == 0) {
      // y <- L * pressure
      invoke_kernel_apply_laplacian(d_y, _pressure);             
      //       r <- rhs - L*pressure
      // (aka) r <- rhs - y
      _d_r.linear_combination((T)1, *_rhs, (T)-1, d_y);
    }
    else {
      //     r <- r - alpha * (L * p) 
      // aka r <- r - alpha * y 
      _d_r.linear_combination((T)1, _d_r, -alpha, d_y);
    }
    
    // beta <- <r_{i+1},r_{i+1}>/<r,r> 
    T deltaold = deltanew;
    _d_r.reduce_sqrsum(deltanew);
    T beta = deltanew / deltaold;
    
    // d <- r + beta*d
    d_d.linear_combination(beta, d_d, (T)1, _d_r);

    printf("%d: %g\n", iteration_count, sqrt(deltanew));

    // check convergence (iter > max_iter) or (sqrt(<r,r>) < tolerance)
    iteration_count++;
    converged = (iteration_count > max_iter) || (deltanew < tolerance*tolerance);
  }

  bool ok;

  if(deltanew <= tolerance*tolerance) {
    printf("[INFO] Sol_PCGPressure3DDevice<T>::do_pcg - Converged to %f residual in %d iterations\n", sqrt(deltanew), iteration_count);
    ok = true;
  }
  else {
    printf("[WARNING] Sol_PCGPressure3DDevice<T>::do_pcg - Failed to converge within %d iterations (achieved %f residual)\n",max_iter,sqrt(deltanew));
    ok = false;
  }

  T r_linf;
  _d_r.reduce_maxabs(r_linf);
  result_linf = r_linf;
  result_l2 = sqrt(deltanew);
  return ok;
}


template<typename T>
bool
Sol_PCGPressure3DDevice<T>::solve(double &residual, double tolerance, int max_iter)
{
  clear_error();

  if (!check_float(tolerance)) {
    printf("[ERROR] Sol_PCGPressure3DDevice::solve - garbage tolerance value %f\n", tolerance);
    return false;
  }

  if (tolerance < 0) {
    printf("[ERROR] Sol_PCGPressure3DDevice::solve - negative tolerance value %f\n", tolerance);
    return false;
  }

  if (max_iter <= 0) {
    printf("[WARNING] Sol_PCGPressure3DDevice::solve - non-positive max_iter %d\n", max_iter);
    return false;
  }

  double l2, linf;
  bool ok = true;

  if (!do_pcg(tolerance, max_iter, l2, linf)) {
    printf("[WARNING] Sol_MultigridPressure3DBase::solve - do_fmg failed\n");
    ok = false;
  }

  residual = (convergence == CONVERGENCE_L2) ? l2 : linf;
  return ok && !any_error();
}

template<typename T>
bool 
Sol_PCGPressure3DDevice<T>::initialize_storage(int nx_val, int ny_val, int nz_val, float hx_val, float hy_val, float hz_val, Grid3DDevice<T> *rhs, Grid3DDevice<T> *coefficient)
{
  if (!check_float(hx_val) || !check_float(hy_val) || !check_float(hz_val)) {
    printf("[ERROR] Sol_PCGPressure3DDevice::initialize_storage - garbage hx,hy,hz value %f %f %f\n", hx_val, hy_val, hz_val);
    return false;
  }

  if (!bc.check_type(BC_PERIODIC, BC_NEUMANN)) {
    printf("[ERROR] Sol_PCGPressure3DDevice::initialize_storage - unsupported boundary type\n");
    return false;
  }

  _h = min3(hx_val, hy_val, hz_val);
  _fx = (_h * _h) / (hx_val * hx_val);
  _fy = (_h * _h) / (hy_val * hy_val);
  _fz = (_h * _h) / (hz_val * hz_val);
  _nx = nx_val;
  _ny = ny_val;
  _nz = nz_val;


  if (rhs->gx() < 1 || rhs->gy() < 1 || rhs->gz() < 1) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - rhs has invalid ghost cells (%d,%d,%d), must be >= 1\n", rhs->gx(), rhs->gy(), rhs->gz());
    return false;
  }

  if (rhs->nx() != nx_val, rhs->ny() != ny_val, rhs->nz() != nz_val) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - rhs dimension mismatch (%d,%d,%d) != (%d,%d,%d)\n", rhs->nx(), rhs->ny(), rhs->nz(), nx_val, ny_val, nz_val);
    return false;
  }
  _rhs = rhs;

  if (!coefficient->check_layout_match(*_rhs)) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - coefficient layout must match rhs\n");
    return false;
  }
  _coefficient = coefficient;

  if (!_pressure.init_congruent(*_rhs)) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize pressure\n");
    return false;
  }

  _pressure.clear_zero();

  if (!_d_r.init_congruent(_pressure)) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _d_r\n");
    return false;
  }

  _mg.nu1 = 2;
  _mg.nu2 = 2;
  _mg.bc = bc;
  if (!_mg.initialize_storage(_nx,_ny,_nz,hx_val,hy_val,hz_val,&_d_r)) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize multigrid solver\n");
    return false;
  }

  return true;
}




template class Sol_PCGPressure3DDevice<float>;

#ifdef OCU_DOUBLESUPPORT
template class Sol_PCGPressure3DDevice<double>;
#endif // FS_DOUBLESUPPORT




} // end namespace
