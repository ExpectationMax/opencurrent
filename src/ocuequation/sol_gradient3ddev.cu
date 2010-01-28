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
#include "ocustorage/grid3dops.h"
#include "ocuequation/sol_gradient3d.h"



template<typename T>
__global__ void Sol_Gradient3DDevice_subtract_grad(T *u, T *v, T *w, T *phi, T coefficient,
  T invhx, T invhy, T invhz, 
  int xstride, int ystride,
  int nx, int ny, int nz, int blocksInY, float invBlocksInY)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);

  // transpose for coalescing since k is the fastest changing index 
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;


  if (i < nx && j < ny && k < nz) {
    int idx = __mul24(i, xstride) + __mul24(j,ystride) + k;

    T p_ijk = phi[idx];
    u[idx] += coefficient * invhx * (p_ijk - phi[idx - xstride]);
    v[idx] += coefficient * invhy * (p_ijk - phi[idx - ystride]);
    w[idx] += coefficient * invhz * (p_ijk - phi[idx - 1      ]);
  }
}


namespace ocu {

template<typename T>
Sol_Gradient3DDevice<T>::Sol_Gradient3DDevice() 
{
  _hx = _hy = _hz;
  u = v = w = 0;
  phi = 0;
  coefficient = (T)-1;
}

template<typename T>
bool
Sol_Gradient3DDevice<T>::solve() 
{
  int tnx = phi->nz();
  int tny = phi->ny();
  int tnz = phi->nx();

  int threadsInX = 16;
  int threadsInY = 4;
  int threadsInZ = 4;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  PreKernel();

  Sol_Gradient3DDevice_subtract_grad<<<Dg, Db>>>(&u->at(0,0,0),&v->at(0,0,0),&w->at(0,0,0), &phi->at(0,0,0), coefficient,
    (T)(1/_hx), (T)(1/_hy), (T)(1/_hz), 
    phi->xstride(), phi->ystride(), 
    phi->nx(), phi->ny(), phi->nz(), blocksInY, 1.0f / (float)blocksInY);

  return PostKernel("Sol_Gradient3DDevice_subtract_grad");
}

template<typename T>
bool 
Sol_Gradient3DDevice<T>::initialize_storage(int nx, int ny, int nz, double hx, double hy, double hz, 
    Grid3DDevice<T> *u_val, Grid3DDevice<T> *v_val, Grid3DDevice<T> *w_val, Grid3DDevice<T> *phi_val)
{
  u = u_val;
  v = v_val;
  w = w_val;
  phi = phi_val;

  if (!check_valid_mac_dimensions(*u_val, *v_val, *w_val, nx, ny, nz)) {
    printf("[ERROR] Sol_Gradient3DDevice::initialize_storage - u,v,w grid dimensions mismatch\n");
    return false;
  }

  if (phi_val->nx() != nx || phi_val->ny() != ny || phi_val->nz() != nz ||
    !phi_val->check_layout_match(*u_val)) {
    printf("[ERROR] Sol_Gradient3DDevice::initialize_storage - invalid dimensions for phi\n");
    return false;
  }

  // since they all have the same layout, we only need to test u
  if (u_val->gx() < 1 || u_val->gy() < 1 || u_val->gz() < 1) {
    printf("[ERROR] Sol_Gradient3DDevice::initialize_storage - must have at least one ghost cell on all sides\n");
    return false;
  }

  if (!check_float(hx) || !check_float(hy) || !check_float(hz)) {
    printf("[ERROR] Sol_Gradient3DDevice::initialize_storage - garbage hx,hy,hz value\n");
    return false;
  }

  _hx = hx;
  _hy = hy;
  _hz = hz;

  return true;
}


template class Sol_Gradient3DDevice<float>;
#ifdef OCU_DOUBLESUPPORT
template class Sol_Gradient3DDevice<double>;
#endif

} // end namespace

