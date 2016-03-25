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
#include "ocuutil/thread.h"
#include "ocuequation/sol_laplaciancent1d.h"

//***********************************************************************************
// Kernels (currently must be outside of namespaces)
//***********************************************************************************


__global__ void Sol_Laplacian1DCentered_apply_stencil(float inv_h2, float *deriv_densitydt, float *density, int nx)
{
  int i = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
  
  i--;

  // note that density & deriv_densitydt are both shifted so that they point to the "0" element, even though the arrays start
  // at element -1.  hence by offsetting i as above, we will get better coalescing for the cost of an added test if i>=0  

  if (i>=0 && i < nx)
    deriv_densitydt[i] = inv_h2 * (density[i-1] - 2.0f * density[i] + density[i+1]);
}


__global__ void Sol_Laplacian1DCentered_apply_boundary_conditions(float *density, ocu::BoundaryCondition left, ocu::BoundaryCondition right, int nx, float h)
{   
  if (left.type == ocu::BC_PERIODIC) {
    density[-1] = density[nx-1];
  }
  else if (left.type == ocu::BC_DIRICHELET) {
    density[-1] = 2 * left.value - density[0];
  }
  else { // (left.type == ocu::BC_NEUMANN)
    density[-1] = density[0] + h * left.value;
  }


  if (right.type == ocu::BC_PERIODIC) {
    density[nx] = density[1];
  }
  else if (right.type == ocu::BC_DIRICHELET) {
    density[nx] = 2 * right.value - density[nx-1];
  }
  else { // (right.type == ocu::BC_NEUMANN)
    density[nx] = density[nx-1] + h * right.value;
  }
}



namespace ocu {


bool 
Sol_LaplacianCentered1DDevice::initialize_storage(int nx)
{
  density.init(nx, 1);
  deriv_densitydt.init(nx, 1); // pad so that memory accesses will be better coalesced
  _nx = nx;

  return true;
}

void 
Sol_LaplacianCentered1DDevice::apply_boundary_conditions()
{
  dim3 Dg(1);
  dim3 Db(1);
  
  Sol_Laplacian1DCentered_apply_boundary_conditions<<<Db, Db, 0, ThreadManager::get_compute_stream()>>>(&density.at(0), left, right, nx(), h());
  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_LaplacianCentered1DDevice::apply_boundary_conditions - CUDA error \"%s\"\n", cudaGetErrorString(er));
  }  
}

bool 
Sol_LaplacianCentered1DDevice::solve()
{
  // centered differencing
  float inv_h2 = coefficient() / (h() * h());

  apply_boundary_conditions();

  // launch nx+1 threads
  dim3 Dg((nx()+1+255) / 256);
  dim3 Db(256);

  PreKernel();
  Sol_Laplacian1DCentered_apply_stencil<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(inv_h2, &deriv_densitydt.at(0), &density.at(0), nx());
  PostKernelDim("Sol_LaplacianCentered1DDevice::solve", Dg, Db);

  return !any_error();
}



bool 
Sol_LaplacianCentered1DDeviceNew::initialize_storage(int nx, Grid1DDeviceF *density_val)
{
  density = density_val;
  if (density->nx() != nx) {
    printf("[ERROR] Sol_LaplacianCentered1DDeviceNew::initialize_storage - density width %d != %d\n", density->nx(), nx);
    return false;
  }

  deriv_densitydt.init(nx, 1); 
  _nx = nx;

  return true;
}


bool 
Sol_LaplacianCentered1DDeviceNew::solve()
{
  // centered differencing
  float inv_h2 = coefficient() / (h() * h());

  // launch nx+1 threads
  dim3 Dg((nx()+1+255) / 256);
  dim3 Db(256);

  PreKernel();
  Sol_Laplacian1DCentered_apply_stencil<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(inv_h2, &deriv_densitydt.at(0), &density->at(0), nx());
//  Sol_Laplacian1DCentered_apply_stencil<<<Dg, Db>>>(inv_h2, &deriv_densitydt.at(0), &density->at(0), nx());
  PostKernelDim("Sol_LaplacianCentered1DDevice::solve", Dg, Db);

  return !any_error();
}


}

