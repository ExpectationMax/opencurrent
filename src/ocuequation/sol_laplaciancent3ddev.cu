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

#include "ocuequation/sol_laplaciancent3d.h"
#include "ocuutil/timer.h"
#include "ocuutil/timing_pool.h"
#include "ocuutil/float_routines.h"




template<typename T>
__global__ void Sol_LaplacianCentered3DDevice_stencil(
  T *phi, T *dphidt, 
  T invhx2, T invhy2, T invhz2, T coefficient,
  int xstride, int ystride, 
  int nx, int ny, int nz, 
  int blocksInY, float invBlocksInY)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);

  // transpose for coalescing since k is the fastest changing index 
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  // shift so we will get maximum coalescing.  This means that we will need to test if k>0 below.

  if (i < nx && j < ny && k < nz) {

    // calc phi indexing
    int idx = __mul24(i, xstride) + __mul24(j,ystride) + k;
    
    T phi_ijk = phi[idx];
    T dphi_ijk = dphidt[idx];

    T laplacian = 
      invhz2 * (phi[idx + 1      ] + phi[idx - 1      ] - ((T)2) * phi_ijk) +
      invhy2 * (phi[idx + ystride] + phi[idx - ystride] - ((T)2) * phi_ijk) +
      invhx2 * (phi[idx + xstride] + phi[idx - xstride] - ((T)2) * phi_ijk);


    dphidt[idx] = dphi_ijk + coefficient * laplacian;
  }

}


namespace ocu {

template<typename T>
Sol_LaplacianCentered3DDevice<T>::Sol_LaplacianCentered3DDevice()
{
  _hx = 0;
  _hy = 0;
  _hz = 0;
  _nx = 0;
  _ny = 0;
  _nz = 0;

  phi = 0;
  deriv_phidt = 0;

  coefficient = 1;
}

template<typename T>
bool Sol_LaplacianCentered3DDevice<T>::solve()
{
  // launch nz+1 threads since the kernel shifts by -1 for better coalescing.  Also, we must transpose x,y,z into
  // thread ids for beter coalescing behaviors, so that adjacent threads operate on adjacent memory locations.
  int tnx = _nz;
  int tny = _ny;
  int tnz = _nx;

  int threadsInX = 16;
  int threadsInY = 2;
  int threadsInZ = 2;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);
  

  PreKernel();
  Sol_LaplacianCentered3DDevice_stencil<<<Dg, Db>>>(&phi->at(0,0,0),&deriv_phidt->at(0,0,0),
    (T)(1/(_hx*_hx)), (T)(1/(_hy*_hy)), (T)(1/(_hz*_hz)), coefficient,
    phi->xstride(), phi->ystride(), 
    _nx, _ny, _nz, blocksInY, 1.0f / (float)blocksInY);
  return PostKernel("Sol_LaplacianCentered3DDevice_stencil");
}

template<typename T>
bool Sol_LaplacianCentered3DDevice<T>::initialize_storage(int nx, int ny, int nz, double hx, double hy, double hz, Grid3DDevice<T> *phi_val, Grid3DDevice<T> *deriv_phidt_val)
{
  // phi must be the proper dimensions
  if (phi_val->nx() != nx || phi_val->ny() != ny || phi_val->nz() != nz) {
    printf("[ERROR] Sol_LaplacianCentered3DDevice::initialize_storage - phi dimensions mismatch\n");
    return false;
  }

  if (phi_val->gx() < 1 || phi_val->gy() < 1 || phi_val->nz() < 1) {
    printf("[ERROR] Sol_LaplacianCentered3DDevice::initialize_storage - must have >= 1 ghost cell\n");
    return false;
  }

  // phi & deriv_phidt share the same memory layout.  This is a cuda optimization to simplify indexing.
  if (!phi_val->check_layout_match(*deriv_phidt_val)) {
    printf("[ERROR] Sol_LaplacianCentered3DDevice::initialize_storage - phi/deriv_phidt layout mismatch\n");
    return false;
  }

  if (!check_float(hx) || !check_float(hy) || !check_float(hz)) {
    printf("[ERROR] Sol_LaplacianCentered3DDevice::initialize_storage - garbage hx,hy,hz value\n");
    return false;
  }

  _hx = hx;
  _hy = hy;
  _hz = hz;
  _nx = nx;
  _ny = ny;
  _nz = nz;

  phi = phi_val;
  deriv_phidt = deriv_phidt_val;

  return true;
}

template class Sol_LaplacianCentered3DDevice<float>;
#ifdef OCU_DOUBLESUPPORT
template class Sol_LaplacianCentered3DDevice<double>;
#endif

} // end namespace

