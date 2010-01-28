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

#include "ocuequation/eqn_incompressns3d.h"
#include "ocuutil/kernel_wrapper.h"

template<typename T>
__global__ void Eqn_IncompressibleNS3D_add_thermal_force(T *dvdt, T coefficient, const T *temperature,
  int xstride, int ystride, int nbr_stride, 
  int nx, int ny, int nz, int blocksInY, float invBlocksInY)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);

  // transpose for coalescing since k is the fastest changing index 
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  // shift so we will get maximum coalescing.  This means that we will need to test if k>0 below.
  int idx = __mul24(i, xstride) + __mul24(j,ystride) + k;

  if (i < nx && j < ny && k < nz) {
    dvdt[idx] += ((T).5) * coefficient * (temperature[idx] + temperature[idx-nbr_stride]);
  }
}

namespace ocu {



template<typename T>
void 
Eqn_IncompressibleNS3D<T>::add_thermal_force()
{
  // apply thermal force by adding -gkT to dvdt (let g = -1, k = 1, so this is just dvdt += T)
  //_advection_solver.deriv_vdt.linear_combination((T)1.0, _advection_solver.deriv_vdt, (T)1.0, _thermal_solver.phi);

  int tnx = nz();
  int tny = ny();
  int tnz = nx();

  int threadsInX = 16;
  int threadsInY = 2;
  int threadsInZ = 2;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  T direction_mult = _vertical_direction & DIR_NEGATIVE_FLAG ? 1 : -1;
  T *uvw = (_vertical_direction & DIR_XAXIS_FLAG) ? &_deriv_udt.at(0,0,0) :
           (_vertical_direction & DIR_YAXIS_FLAG) ? &_deriv_vdt.at(0,0,0) : &_deriv_wdt.at(0,0,0);

  KernelWrapper wrapper;
  wrapper.PreKernel();

  Eqn_IncompressibleNS3D_add_thermal_force<<<Dg, Db>>>(uvw, direction_mult * _gravity * _bouyancy, &_temp.at(0,0,0),
    _temp.xstride(), _temp.ystride(), _temp.stride(_vertical_direction), nx(), ny(), nz(), 
    blocksInY, 1.0f / (float)blocksInY);

  if (!wrapper.PostKernel("Eqn_IncompressibleNS3D_add_thermal_force"))
    add_error();

}

template void Eqn_IncompressibleNS3D<float>::add_thermal_force();

#ifdef OCU_DOUBLESUPPORT
template void Eqn_IncompressibleNS3D<double>::add_thermal_force();
#endif


} // end namespace

