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

#include "cuda.h"
#include <cstdio>

#include "ocuutil/reduction_op.h"
#include "ocustorage/grid1dops.h"


//
// This is the most basic reduction strategy.  Each block of B threads
// will reduce B consecutive elements to 1 element.  That 1 element is
// written into the output[] array.  The host code will keep iterating
// until this produces the final result.  Hence, we expect log_B(N)
// kernel launches.
//
template<typename T, typename REDUCE>
__global__ void reduce_per_cta(const T *input, int gx, unsigned int N, T *output, bool first, REDUCE red)
{
  int blocksize = blockDim.x;
  int tid       = threadIdx.x;
  int i         = blockIdx.x*blockDim.x + threadIdx.x - gx;

  extern __shared__ char smempool[];
  T *shmem = (T *)smempool;

  // load one cta's worth of data into shmem (shift by gx for coalescing)
  if (first) {
    shmem[tid] = (i >= 0 && i<N) ? red.process(input[i]) : red.identity();
  }
  else {
    shmem[tid] = (i >= 0 && i<N) ? input[i] : red.identity();
  }

  __syncthreads();
    
  // do a cta-wide reduction
  for(int bit=blocksize/2; bit>0; bit/=2)
  {
      T t = red.reduce(shmem[tid] , shmem[tid ^ bit]);  
      __syncthreads();
      shmem[tid] = t;
      __syncthreads();
  }

  // write the result out if we are thread 0
  if( tid==0 ) {
    output[blockIdx.x] = shmem[tid];
  }
}


namespace ocu {

template<typename T, typename REDUCE>
bool reduce_with_operator(const ocu::Grid1DDevice<T> &grid, T &result, REDUCE reduce, bool suppress_process)
{
  unsigned int ctasize = 256;
  unsigned int num_elems = grid.nx();
  unsigned int nthreads = grid.nx() + grid.gx();
  int numblocks = (nthreads + ctasize - 1) / ctasize;

  ocu::Grid1DDevice<T> temp_grid1, temp_grid2;
  temp_grid1.init(numblocks, 0);
  temp_grid2.init(numblocks, 0);

  T *temp_to = &temp_grid1.at(0);
  T *temp_from = &temp_grid2.at(0);
  T *temp_temp;

  unsigned int smemsize  = ctasize * sizeof(T);
  reduce_per_cta<<<numblocks, ctasize, smemsize>>>(&grid.at(0), grid.gx(), num_elems, temp_to, suppress_process ? false : true, reduce);


  while (numblocks > 1) {
    // swap buffers
    temp_temp = temp_to; temp_to = temp_from; temp_from = temp_temp;

    num_elems = numblocks;
    numblocks = (num_elems + ctasize - 1) / ctasize;
    reduce_per_cta<<<numblocks, ctasize, smemsize>>>(temp_from, 0, num_elems, temp_to, false, reduce);
    
    cudaError_t er = cudaGetLastError();
    if (er != (unsigned int)CUDA_SUCCESS) {
      printf("[ERROR] reduce_with_operator - reduce_per_cta failed with CUDA error \"%s\"\n", cudaGetErrorString(er));
      return false;    
    }

  }

  cudaError_t er = cudaMemcpy(&result, temp_to, sizeof(T), cudaMemcpyDeviceToHost);
  if (er != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] reduce_with_operator - cudaMemcpy failed with CUDA error \"%s\"\n", cudaGetErrorString(er));
    return false;    
  }

  return true; 
}

template bool reduce_with_operator(const ocu::Grid1DDevice<float> &, float &, ReduceDevMaxAbsF, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<float> &, float &, ReduceDevMaxF, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<float> &, float &, ReduceDevMinF, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<float> &, float &, ReduceDevSum<float>, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<float> &, float &, ReduceDevSqrSum<float>, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<float> &, float &, ReduceDevCheckNan<float>, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<int> &, int &, ReduceDevMaxAbsI, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<int> &, int &, ReduceDevSum<int>, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<int> &, int &, ReduceDevSqrSum<int>, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<int> &, int &, ReduceDevMaxI, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<int> &, int &, ReduceDevMinI, bool);

#ifdef OCU_DOUBLESUPPORT

template bool reduce_with_operator(const ocu::Grid1DDevice<double> &, double &, ReduceDevMaxD, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<double> &, double &, ReduceDevMinD, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<double> &, double &, ReduceDevMaxAbsD, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<double> &, double &, ReduceDevSum<double>, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<double> &, double &, ReduceDevSqrSum<double>, bool);
template bool reduce_with_operator(const ocu::Grid1DDevice<double> &, double &, ReduceDevCheckNan<double>, bool);

#endif // OCU_DOUBLESUPPORT

} // end namespace

