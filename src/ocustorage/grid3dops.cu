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
#include "ocuutil/reduction_op.h"
#include "ocustorage/grid1dops.h"
#include "ocustorage/grid3dops.h"
#include "ocuutil/kernel_wrapper.h"



template<typename T, typename REDUCE>
__global__ 
void reduce_kernel(T* grid2d, int xstride2d, const T* grid, int xstride, int ystride, int nx, int ny, int nz, REDUCE red)
{
  // we are going to organize threads into a CTA of dimensions 2 x 2 x 32, we a CTA
  // will have 4 groups of 32 threads, each group assigned to sequential k values.
  // However, threads are considered adjacent if they are sequential in i, so we have to transpose
  // from CUDA's indices to grid3d indices.  That way the adjacent 32 threads in z will all 
  // be in the same warp.


  int i = blockIdx.x * blockDim.z + threadIdx.z;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  extern __shared__ char smempool[];
  T *shmem = (T *)smempool;

  // offset to a per-warp 32-float segment of the smempool based on the threadIdx.y and threadIdx.z
  shmem += (threadIdx.z * blockDim.y + threadIdx.y) * 32;

  // only continue if we are in-bounds in x and y - all threads in a warp
  // will go the same way through this conditional
  if (i < nx && j < ny) {

    // starting memory location for our z-row
    const T *base_ptr = grid + xstride * i + ystride * j;
    T result = red.identity();

    int k = threadIdx.x;

    // step 1. do a linear scan through through all values (i,j,k+32n)
    if (k < nz) {
      result = red.process(base_ptr[k]);
    }
    
    k += 32;

    while (k < nz) {
      // assumes that gz < 32.  Otherwise, we'd need to add an additional 'if (k >=0)' test here.  But this assumption is reasonable.
      result = red.reduce(red.process(base_ptr[k]), result);

      k += 32;
    }

    // step 2. do a warp-wide reduction on result, which will result a sum over (i,j,k) for all k
    int wid = threadIdx.x;  

    // the out-of-bounds threads are automatically handled by initializing result with an identify element
    shmem[wid] = result;

    // do the reduction - no synthreads needed 'cause this is a warp.
    // NB: this will fail in device emulation mode, or M-CUDA!
    if (wid < 16) shmem[wid] = red.reduce(shmem[wid], shmem[wid+16]);
    if (wid < 8) shmem[wid] = red.reduce(shmem[wid], shmem[wid+8]);
    if (wid < 4) shmem[wid] = red.reduce(shmem[wid], shmem[wid+4]);
    if (wid < 2) shmem[wid] = red.reduce(shmem[wid], shmem[wid+2]);
    if (wid < 1) shmem[wid] = red.reduce(shmem[wid], shmem[wid+1]);

    // step 3. write the final per-warp result into grid2d
    if (wid == 0)
      grid2d[i * xstride2d + j] = shmem[0];
  }
}




namespace ocu {


template<typename T, typename REDUCE>
bool reduce_with_operator(const ocu::Grid3DDevice<T> &grid, T &result, REDUCE reduce)
{
  ocu::Grid1DDevice<T> d_grid1d;

  d_grid1d.init(grid.nx()*grid.ny(), 0);

  int tnx = grid.nx();
  int tny = grid.ny();

  int threadsInX = 2;
  int threadsInY = 2;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;

  // one block per 2x2 section of i,j
  dim3 Dg = dim3(blocksInX, blocksInY);
  // transpose x,y,z so that we have 32 adjancent threads (which will operate on sequential z-addresses in the grid)
  dim3 Db = dim3(32, threadsInY, threadsInX);

  unsigned int smemsize  = threadsInX*threadsInY*32 * sizeof(T);

  KernelWrapper wrapper;

  wrapper.PreKernel();
  reduce_kernel<<<Dg,Db,smemsize>>>(&d_grid1d.at(0), grid.ny(), &grid.at(0,0,0), grid.xstride(), grid.ystride(), grid.nx(), grid.ny(), grid.nz(), reduce);
  if (!wrapper.PostKernel("reduce_kernel"))
    return false;

  if (!reduce_with_operator(d_grid1d, result, reduce, true)) {
    printf("[ERROR] reduce_with_operator - 1d reduction failed\n");
  }
  return true; 
}


template bool reduce_with_operator(const ocu::Grid3DDevice<float> &, float &, ReduceDevMaxF);
template bool reduce_with_operator(const ocu::Grid3DDevice<float> &, float &, ReduceDevMinF);
template bool reduce_with_operator(const ocu::Grid3DDevice<float> &, float &, ReduceDevMaxAbsF);
template bool reduce_with_operator(const ocu::Grid3DDevice<float> &, float &, ReduceDevSum<float>);
template bool reduce_with_operator(const ocu::Grid3DDevice<float> &, float &, ReduceDevSqrSum<float>);
template bool reduce_with_operator(const ocu::Grid3DDevice<float> &, float &, ReduceDevCheckNan<float>);
template bool reduce_with_operator(const ocu::Grid3DDevice<int> &, int &, ReduceDevMaxI);
template bool reduce_with_operator(const ocu::Grid3DDevice<int> &, int &, ReduceDevMinI);
template bool reduce_with_operator(const ocu::Grid3DDevice<int> &, int &, ReduceDevMaxAbsI);
template bool reduce_with_operator(const ocu::Grid3DDevice<int> &, int &, ReduceDevSum<int>);
template bool reduce_with_operator(const ocu::Grid3DDevice<int> &, int &, ReduceDevSqrSum<int>);

#ifdef OCU_DOUBLESUPPORT

template bool reduce_with_operator(const ocu::Grid3DDevice<double> &, double &, ReduceDevMaxD);
template bool reduce_with_operator(const ocu::Grid3DDevice<double> &, double &, ReduceDevMinD);
template bool reduce_with_operator(const ocu::Grid3DDevice<double> &, double &, ReduceDevMaxAbsD);
template bool reduce_with_operator(const ocu::Grid3DDevice<double> &, double &, ReduceDevSum<double>);
template bool reduce_with_operator(const ocu::Grid3DDevice<double> &, double &, ReduceDevSqrSum<double>);
template bool reduce_with_operator(const ocu::Grid3DDevice<double> &, double &, ReduceDevCheckNan<double>);

#endif // OCU_DOUBLESUPPORT

} // end namespace
