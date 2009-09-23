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
#include "ocuequation/sol_divergence3d.h"


template<typename T>
__global__ void Sol_Divergence3DDevice_calculate_divergence(T *u, T *v, T *w, T *divergence,
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

    divergence[idx] = invhx * (u[idx + xstride] - u[idx]) +
                      invhy * (v[idx + ystride] - v[idx]) +
                      invhz * (w[idx + 1      ] - w[idx]);
  }
}

namespace ocu {

template<typename T>
Sol_Divergence3DDevice<T>::Sol_Divergence3DDevice()
{
  _nx = 0;
  _ny = 0;
  _nz = 0;
  _hx = 0;
  _hy = 0;
  _hz = 0;

  u = v = w = 0;
  divergence = 0;
}

template<typename T>
bool
Sol_Divergence3DDevice<T>::solve()
{
  int tnx = _nz;
  int tny = _ny;
  int tnz = _nx;

  int threadsInX = 16;
  int threadsInY = 4;
  int threadsInZ = 4;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  PreKernel();
  Sol_Divergence3DDevice_calculate_divergence<<<Dg, Db>>>(&u->at(0,0,0),&v->at(0,0,0),&w->at(0,0,0), &divergence->at(0,0,0),
    (T)(1/_hx), (T)(1/_hy), (T)(1/_hz), 
    u->xstride(), u->ystride(), 
    _nx, _ny, _nz, blocksInY, 1.0f / (float)blocksInY);
  return PostKernel("Sol_Divergence3DDevice_calculate_divergence::calculate_divergence");
}

template<typename T>
bool 
Sol_Divergence3DDevice<T>::initialize_storage(
  int nx, int ny, int nz, double hx, double hy, double hz, Grid3DDevice<T> *u_val, Grid3DDevice<T> *v_val, Grid3DDevice<T> *w_val, Grid3DDevice<T> *divergence_val)
{
  u = u_val;
  v = v_val;
  w = w_val;
  divergence = divergence_val;

  if (!check_valid_mac_dimensions(*u_val, *v_val, *w_val, nx, ny, nz)) {
    printf("[ERROR] Sol_Divergence3DDevice::initialize_storage - u,v,w grid dimensions mismatch\n");
    return false;
  }

  if (divergence_val->nx() != nx || divergence_val->ny() != ny || divergence_val->nz() != nz ||
    !divergence_val->check_layout_match(*u_val)) {
    printf("[ERROR] Sol_Divergence3DDevice::initialize_storage - invalid dimensions for divergence\n");
    return false;
  }

  // since they all have the same layout, we only need to test u
  if (u_val->gx() < 1 || u_val->gy() < 1 || u_val->gz() < 1) {
    printf("[ERROR] Sol_Divergence3DDevice::initialize_storage - must have at least one ghost cell on all sides\n");
    return false;
  }

  if (!check_float(hx) || !check_float(hy) || !check_float(hz)) {
    printf("[ERROR] Sol_Divergence3DDevice::initialize_storage - garbage hx,hy,hz value\n");
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




template class Sol_Divergence3DDevice<float>;
#ifdef OCU_DOUBLESUPPORT
template class Sol_Divergence3DDevice<double>;
#endif

} // end namespace

