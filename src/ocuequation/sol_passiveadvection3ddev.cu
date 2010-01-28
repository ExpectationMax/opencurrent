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

#include <cstdio>
#include <cuda.h>

#include "ocuutil/timer.h"
#include "ocuutil/timing_pool.h"
#include "ocuutil/float_routines.h"
#include "ocustorage/grid3dops.h"
#include "ocuequation/sol_passiveadvection3d.h"



//! This routine works because u,v,w,phi, and dphidt must all be padded so that they have the same memory layout,
//! even though they have different dimensions.  Then we can calculate indexing math once, and reuse it for
//! all of the grids.
template<typename T, typename INTERP>
__global__ void Sol_PassiveAdvection3D_apply_interp(
  T *u, T *v, T *w, T *phi, T *dphidt, 
  T invhx, T invhy, T invhz, 
  int xstride, int ystride,
  int nx, int ny, int nz, int blocksInY, float invBlocksInY, INTERP interp)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);

  // transpose for coalescing since k is the fastest changing index 
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  if (i < nx && j < ny && k < nz) {

    // calc phi indexing
    int idx = __mul24(i, xstride) + __mul24(j,ystride) + k;
    T phi_ijk = phi[idx];
  
    int idx_pi = idx + xstride;
    int idx_pj = idx + ystride;
    int idx_pk = idx + 1;

    int idx_mi = idx - xstride;
    int idx_mj = idx - ystride;
    int idx_mk = idx - 1;

    T u_pi  = u[idx_pi];
    T u_ijk = u[idx];

    T duphi = u_pi * interp(u_pi, phi_ijk, phi[idx_pi]) - u_ijk * interp(u_ijk, phi[idx_mi], phi_ijk);

    T v_pj  = v[idx_pj];
    T v_ijk = v[idx];

    T dvphi = v_pj * interp(v_pj, phi_ijk, phi[idx_pj]) - v_ijk * interp(v_ijk, phi[idx_mj], phi_ijk);

    T w_pk  = w[idx_pk];
    T w_ijk = w[idx];

    T dwphi = w_pk * interp(w_pk, phi_ijk, phi[idx_pk]) - w_ijk * interp(w_ijk, phi[idx_mk], phi_ijk);
    
    // write the result
    dphidt[idx] = (-duphi * invhx) - (dvphi * invhy) - (dwphi * invhz);
  }
}




namespace ocu {

template<typename T>
Sol_PassiveAdvection3DDevice<T>::Sol_PassiveAdvection3DDevice()
{
  _nx = _ny = _nz = 0;
  u = 0;
  v = 0;
  w = 0;
  phi = 0;
  deriv_phidt = 0;

  interp_type = IT_FIRST_ORDER_UPWIND;
}

template<typename T>
bool 
Sol_PassiveAdvection3DDevice<T>::solve()
{
  int tnx = _nz; 
  int tny = _ny;
  int tnz = _nx;

  int threadsInX = 32;
  int threadsInY = 2;
  int threadsInZ = 2;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  PreKernel();

  if (interp_type == IT_FIRST_ORDER_UPWIND) {    
    Sol_PassiveAdvection3D_apply_interp<<<Dg, Db>>>(&u->at(0,0,0),&v->at(0,0,0),&w->at(0,0,0),&phi->at(0,0,0),&deriv_phidt->at(0,0,0),
      (T)(1/_hx), (T)(1/_hy), (T)(1/_hz), phi->xstride(), phi->ystride(), 
      _nx, _ny, _nz, blocksInY, 1.0f / (float)blocksInY, InterpolatorFirstOrderUpwind<T>());

  }
  else if (interp_type == IT_SECOND_ORDER_CENTERED) {
    Sol_PassiveAdvection3D_apply_interp<<<Dg, Db>>>(&u->at(0,0,0),&v->at(0,0,0),&w->at(0,0,0),&phi->at(0,0,0),&deriv_phidt->at(0,0,0),
      (T)(1/_hx), (T)(1/_hy), (T)(1/_hz), phi->xstride(), phi->ystride(), 
      _nx, _ny, _nz, blocksInY, 1.0f / (float)blocksInY, InterpolatorSecondOrderCentered<T>());
  }
  else {
    printf("[ERROR] Sol_PassiveAdvection3DDevice::solve() - invalid interpolation type %d\n", interp_type);
    return false;
  }

  return PostKernel("Sol_PassiveAdvection3D_apply_interp");
}

template<typename T>
bool 
Sol_PassiveAdvection3DDevice<T>::initialize_storage(int nx, int ny, int nz, double hx, double hy, double hz, Grid3DDevice<T> *u_val, Grid3DDevice<T> *v_val, Grid3DDevice<T> *w_val,
                                                Grid3DDevice<T> *phi_val, Grid3DDevice<T> *deriv_phidt_val)
{
  // u,v,w must be the proper dimensions, i.e. staggered grid
  if (u_val->nx() != nx+1 || u_val->ny() != ny || u_val->nz() != nz) {
    printf("[ERROR] Sol_PassiveAdvection3DDevice::initialize_storage - u dimensions mismatch\n");
    return false;
  }

  if (v_val->nx() != nx || v_val->ny() != ny+1 || v_val->nz() != nz) {
    printf("[ERROR] Sol_PassiveAdvection3DDevice::initialize_storage - v dimensions mismatch\n");
    return false;
  }

  if (w_val->nx() != nx || w_val->ny() != ny || w_val->nz() != nz+1) {
    printf("[ERROR] Sol_PassiveAdvection3DDevice::initialize_storage - v dimensions mismatch\n");
    return false;
  }

  if (phi_val->nx() != nx || phi_val->ny() != ny || phi_val->nz() != nz) {
    printf("[ERROR] Sol_PassiveAdvection3DDevice::initialize_storage - phi dimensions mismatch\n");
    return false;
  }

  if (deriv_phidt_val->nx() != nx || deriv_phidt_val->ny() != ny || deriv_phidt_val->nz() != nz) {
    printf("[ERROR] Sol_PassiveAdvection3DDevice::initialize_storage - deriv_phidt dimensions mismatch\n");
    return false;
  }

  // only need to check phi & deriv since other grids will have matching layout.
  if (phi_val->gx() < 1 || phi_val->gy() < 1 || phi_val->gz() < 1  ||
      deriv_phidt_val->gx() < 1 || deriv_phidt_val->gy() < 1 || deriv_phidt_val->gz() < 1) {
    printf("[ERROR] Sol_PassiveAdvection3DDevice::initialize_storage - phi must have at least 1 ghost cell in all dimensions\n");
    return false;
  }

  // u,v,w must all share the same memory layout.  This is a cuda optimization to simplify indexing.
  if (!u_val->check_layout_match(*v_val) || !u_val->check_layout_match(*w_val) || !u_val->check_layout_match(*phi_val) || !u_val->check_layout_match(*deriv_phidt_val)) {
    printf("[ERROR] Sol_PassiveAdvection3DDevice::initialize_storage - u,v,w layout mismatch\n");
    return false;
  }

  u = u_val;
  v = v_val;
  w = w_val;
  phi = phi_val;
  deriv_phidt = deriv_phidt_val;


  if (!check_float(hx) || !check_float(hy) || !check_float(hz)) {
    printf("[ERROR] Sol_PassiveAdvection3DDevice::initialize_storage - garbage hx,hy,hz value\n");
    return false;
  }

  _hx = hx;
  _hy = hy;
  _hz = hz;

  _nx = nx;
  _ny = ny;
  _nz = nz;

  return true;
}


template class Sol_PassiveAdvection3DDevice<float>;

#ifdef OCU_DOUBLESUPPORT
template class Sol_PassiveAdvection3DDevice<double>;
#endif


} // end namespace

