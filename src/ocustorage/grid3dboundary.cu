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
#include <algorithm>
#include "ocustorage/grid3dboundary.h"
#include "ocuutil/kernel_wrapper.h"


//! Launch enough threads to have threads of dimension (max(nx,ny), max(ny,nz))
//! This does *not* fill in corner element, but that's ok since our finite difference
//! stencil has no corner points.
//! If we needed corner elements, it would probably be fastest to launch a 1-thread kernel
//! that fills them all in, since it's only 8 values = 8 ld and 8 st operations.
template<typename T>
__global__ void kernel_apply_3d_boundary_conditions_level1_nocorners(
    T *phi, 
    ocu::BoundaryConditionSet bc,
    T hx, T hy, T hz, int xstride, int ystride, int nx, int ny, int nz)
{
  int X = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int Y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;

  // boundaries in +-z direction on the xy face
  if (X < nx && Y < ny) {
      int i=X;
      int j=Y;

      if (bc.zneg.type == ocu::BC_PERIODIC) {
        phi[i * xstride + j * ystride + -1] = phi[i * xstride + j * ystride + (nz-1)];
      }
      else if (bc.zneg.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + j * ystride + -1] =  -phi[i * xstride + j * ystride + 0] + 2.0 * (T)bc.zneg.value;
      }
      else { // (bc.zneg.type == ocu::BC_NEUMANN)
        phi[i * xstride + j * ystride + -1] = phi[i * xstride + j * ystride + 0] - hz * (T)bc.zneg.value;
      }


      if (bc.zpos.type == ocu::BC_PERIODIC) {
        phi[i * xstride + j * ystride + nz] = phi[i * xstride + j * ystride + 0];
      }
      else if (bc.zpos.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + j * ystride + nz] =  -phi[i * xstride + j * ystride + (nz-1)] + 2.0 * (T)bc.zpos.value;
      }
      else { // (bc.zpos.type == ocu::BC_NEUMANN)
        phi[i * xstride + j * ystride + nz] =  phi[i * xstride + j * ystride + (nz-1)] + hz * (T)bc.zpos.value;
      }
  }

  // boundaries in +-y direction in the xz face 
  if (X < nx && Y < nz) {
      int i=X;
      int k=Y;

      if (bc.yneg.type == ocu::BC_PERIODIC) {
        phi[i * xstride + -1 * ystride + k] = phi[i * xstride + (ny-1) * ystride + k];
      }
      else if (bc.yneg.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + -1 * ystride + k] =  -phi[i * xstride + 0 * ystride + k] + 2.0 * (T)bc.yneg.value;
      }
      else { // (bc.yneg.type == ocu::BC_NEUMANN)
        phi[i * xstride + -1 * ystride + k] = phi[i * xstride + 0 * ystride + k] - hy * (T)bc.yneg.value;
      }


      if (bc.ypos.type == ocu::BC_PERIODIC) {
        phi[i * xstride + ny * ystride + k] = phi[i * xstride + 0 * ystride + k];
      }
      else if (bc.ypos.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + ny * ystride + k] =  -phi[i * xstride + (ny-1) * ystride + k] + 2.0 * (T)bc.ypos.value;
      }
      else { // (bc.ypos.type == ocu::BC_NEUMANN)
        phi[i * xstride + ny * ystride + k] =  phi[i * xstride + (ny-1) * ystride + k] + hy * (T)bc.ypos.value;
      }

  }

  // boundaries on the +-x direction in the yz face
  if (X < ny && Y < nz) {
      int j=X;
      int k=Y;

      if (bc.xneg.type == ocu::BC_PERIODIC) {
        phi[-1 * xstride + j * ystride + k] = phi[(nx-1) * xstride + j * ystride + k];
      }
      else if (bc.xneg.type == ocu::BC_DIRICHELET) {
        phi[-1 * xstride + j * ystride + k] =  -phi[0 * xstride + j * ystride + k] + 2.0 * (T)bc.xneg.value;
      }
      else { // (bc.xneg.type == ocu::BC_NEUMANN)
        phi[-1 * xstride + j * ystride + k] = phi[0 * xstride + j * ystride + k] - hx * (T)bc.xneg.value;
      }


      if (bc.xpos.type == ocu::BC_PERIODIC) {
        phi[nx * xstride + j * ystride + k] = phi[0 * xstride + j * ystride + k];
      }
      else if (bc.xpos.type == ocu::BC_DIRICHELET) {
        phi[nx * xstride + j * ystride + k] =  -phi[(nx-1) * xstride + j * ystride + k] + 2.0 * (T)bc.xpos.value;
      }
      else { // (bc.xpos.type == ocu::BC_NEUMANN)
        phi[nx * xstride + j * ystride + k] =  phi[(nx-1) * xstride  + j * ystride + k] + hx * (T)bc.xpos.value;
      }
  }

}


template<typename T>
__global__ void kernel_apply_3d_boundary_conditions_level2_nocorners(
    T *phi, 
    ocu::BoundaryConditionSet bc,
    T hx, T hy, T hz, int xstride, int ystride, int nx, int ny, int nz)
{
  int X = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int Y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;

  // boundaries in +-z direction on the xy face
  if (X < nx && Y < ny) {
      int i=X;
      int j=Y;

      if (bc.zneg.type == ocu::BC_PERIODIC) {
        phi[i * xstride + j * ystride + -1] = phi[i * xstride + j * ystride + (nz-1)];
        phi[i * xstride + j * ystride + -2] = phi[i * xstride + j * ystride + (nz-2)];
      }
      else if (bc.zneg.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + j * ystride + -1] =  -phi[i * xstride + j * ystride + 0] + 2 * bc.zneg.value;
        // not sure what the right answer is for this:
        phi[i * xstride + j * ystride + -2] =  -phi[i * xstride + j * ystride + 0] + 2 * bc.zneg.value;
      }
      else { // (bc.zneg.type == ocu::BC_NEUMANN)
        phi[i * xstride + j * ystride + -1] = phi[i * xstride + j * ystride + 0] - hz * bc.zneg.value;
        phi[i * xstride + j * ystride + -2] = phi[i * xstride + j * ystride + 0] - 2 * hz * bc.zneg.value;
      }


      if (bc.zpos.type == ocu::BC_PERIODIC) {
        phi[i * xstride + j * ystride + nz  ] = phi[i * xstride + j * ystride + 0];
        phi[i * xstride + j * ystride + nz+1] = phi[i * xstride + j * ystride + 1];
      }
      else if (bc.zpos.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + j * ystride + nz  ] =  -phi[i * xstride + j * ystride + (nz-1)] + 2 * bc.zpos.value;
        phi[i * xstride + j * ystride + nz+1] =  -phi[i * xstride + j * ystride + (nz-1)] + 2 * bc.zpos.value;
      }
      else { // (bc.zpos.type == ocu::BC_NEUMANN)
        phi[i * xstride + j * ystride + nz  ] =  phi[i * xstride + j * ystride + (nz-1)] + hz * bc.zpos.value;
        phi[i * xstride + j * ystride + nz+1] =  phi[i * xstride + j * ystride + (nz-1)] + 2 * hz * bc.zpos.value;
      }
  }

  // boundaries in +-y direction in the xz face 
  if (X < nx && Y < nz) {
      int i=X;
      int k=Y;

      if (bc.yneg.type == ocu::BC_PERIODIC) {
        phi[i * xstride + -1 * ystride + k] = phi[i * xstride + (ny-1) * ystride + k];
        phi[i * xstride + -2 * ystride + k] = phi[i * xstride + (ny-2) * ystride + k];
      }
      else if (bc.yneg.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + -1 * ystride + k] =  -phi[i * xstride + 0 * ystride + k] + 2 * bc.yneg.value;
        phi[i * xstride + -2 * ystride + k] =  -phi[i * xstride + 0 * ystride + k] + 2 * bc.yneg.value;
      }
      else { // (bc.yneg.type == ocu::BC_NEUMANN)
        phi[i * xstride + -1 * ystride + k] = phi[i * xstride + 0 * ystride + k] - hy * bc.yneg.value;
        phi[i * xstride + -2 * ystride + k] = phi[i * xstride + 0 * ystride + k] - 2 * hy * bc.yneg.value;
      }


      if (bc.ypos.type == ocu::BC_PERIODIC) {
        phi[i * xstride + (ny  ) * ystride + k] = phi[i * xstride + 0 * ystride + k];
        phi[i * xstride + (ny+1) * ystride + k] = phi[i * xstride + 1 * ystride + k];
      }
      else if (bc.ypos.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + (ny  ) * ystride + k] =  -phi[i * xstride + (ny-1) * ystride + k] + 2 * bc.ypos.value;
        phi[i * xstride + (ny+1) * ystride + k] =  -phi[i * xstride + (ny-1) * ystride + k] + 2 * bc.ypos.value;
      }
      else { // (bc.ypos.type == ocu::BC_NEUMANN)
        phi[i * xstride + (ny  ) * ystride + k] =  phi[i * xstride + (ny-1) * ystride + k] + hy * bc.ypos.value;
        phi[i * xstride + (ny+1) * ystride + k] =  phi[i * xstride + (ny-1) * ystride + k] + 2 * hy * bc.ypos.value;
      }

  }

  // boundaries on the +-x direction in the yz face
  if (X < ny && Y < nz) {
      int j=X;
      int k=Y;

      if (bc.xneg.type == ocu::BC_PERIODIC) {
        phi[-1 * xstride + j * ystride + k] = phi[(nx-1) * xstride + j * ystride + k];
        phi[-2 * xstride + j * ystride + k] = phi[(nx-2) * xstride + j * ystride + k];
      }
      else if (bc.xneg.type == ocu::BC_DIRICHELET) {
        phi[-1 * xstride + j * ystride + k] =  -phi[0 * xstride + j * ystride + k] + 2 * bc.xneg.value;
        phi[-2 * xstride + j * ystride + k] =  -phi[0 * xstride + j * ystride + k] + 2 * bc.xneg.value;
      }
      else { // (bc.xneg.type == ocu::BC_NEUMANN)
        phi[-1 * xstride + j * ystride + k] = phi[0 * xstride + j * ystride + k] - hx * bc.xneg.value;
        phi[-2 * xstride + j * ystride + k] = phi[0 * xstride + j * ystride + k] - 2 * hx * bc.xneg.value;
      }


      if (bc.xpos.type == ocu::BC_PERIODIC) {
        phi[(nx  ) * xstride + j * ystride + k] = phi[0 * xstride + j * ystride + k];
        phi[(nx+1) * xstride + j * ystride + k] = phi[1 * xstride + j * ystride + k];
      }
      else if (bc.xpos.type == ocu::BC_DIRICHELET) {
        phi[(nx  ) * xstride + j * ystride + k] =  -phi[(nx-1) * xstride + j * ystride + k] + 2 * bc.xpos.value;
        phi[(nx+1) * xstride + j * ystride + k] =  -phi[(nx-1) * xstride + j * ystride + k] + 2 * bc.xpos.value;
      }
      else { // (bc.xpos.type == ocu::BC_NEUMANN)
        phi[(nx  ) * xstride + j * ystride + k] =  phi[(nx-1) * xstride  + j * ystride + k] + hx * bc.xpos.value;
        phi[(nx+1) * xstride + j * ystride + k] =  phi[(nx-1) * xstride  + j * ystride + k] + 2 * hx * bc.xpos.value;
      }
  }

}




template<typename T>
__global__ void kernel_apply_3d_boundary_conditions_level1_z(
    T *phi, 
    ocu::BoundaryConditionSet bc, 
    T hx, T hy, T hz, int xstride, int ystride, int gx, int gy, int gz, int nx, int ny, int nz)
{
  int i = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;

  i -= gx;
  j -= gy;

  // boundaries in +-z direction on the xy face
  if (i >= -1 && i <= nx && j >= -1 && j <= ny) {
      if (bc.zneg.type == ocu::BC_PERIODIC) {
        phi[i * xstride + j * ystride + -1] = phi[i * xstride + j * ystride + (nz-1)];
      }
      else if (bc.zneg.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + j * ystride + -1] =  -phi[i * xstride + j * ystride + 0] + 2 * bc.zneg.value;
      }
      else { // (bc.zneg.type == ocu::BC_NEUMANN)
        phi[i * xstride + j * ystride + -1] = phi[i * xstride + j * ystride + 0] - hz * bc.zneg.value;
      }


      if (bc.zpos.type == ocu::BC_PERIODIC) {
        phi[i * xstride + j * ystride + nz] = phi[i * xstride + j * ystride + 0];
      }
      else if (bc.zpos.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + j * ystride + nz] =  -phi[i * xstride + j * ystride + (nz-1)] + 2 * bc.zpos.value;
      }
      else { // (bc.zpos.type == ocu::BC_NEUMANN)
        phi[i * xstride + j * ystride + nz] =  phi[i * xstride + j * ystride + (nz-1)] + hz * bc.zpos.value;
      }
  }
}

template<typename T>
__global__ void kernel_apply_3d_boundary_conditions_level1_y(
    T *phi, 
    ocu::BoundaryConditionSet bc,
    T hx, T hy, T hz, int xstride, int ystride, int gx, int gy, int gz, int nx, int ny, int nz)
{
  int i = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int k = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;

  i -= gx;
  k -= gz;

  if (i >= -1 && i <= nx && k >= 0 && k < nz) {

      if (bc.yneg.type == ocu::BC_PERIODIC) {
        phi[i * xstride + -1 * ystride + k] = phi[i * xstride + (ny-1) * ystride + k];
      }
      else if (bc.yneg.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + -1 * ystride + k] =  -phi[i * xstride + 0 * ystride + k] + 2 * bc.yneg.value;
      }
      else { // (bc.yneg.type == ocu::BC_NEUMANN)
        phi[i * xstride + -1 * ystride + k] = phi[i * xstride + 0 * ystride + k] - hy * bc.yneg.value;
      }


      if (bc.ypos.type == ocu::BC_PERIODIC) {
        phi[i * xstride + ny * ystride + k] = phi[i * xstride + 0 * ystride + k];
      }
      else if (bc.ypos.type == ocu::BC_DIRICHELET) {
        phi[i * xstride + ny * ystride + k] =  -phi[i * xstride + (ny-1) * ystride + k] + 2 * bc.ypos.value;
      }
      else { // (bc.ypos.type == ocu::BC_NEUMANN)
        phi[i * xstride + ny * ystride + k] =  phi[i * xstride + (ny-1) * ystride + k] + hy * bc.ypos.value;
      }
  }
}


template<typename T>
__global__ void kernel_apply_3d_boundary_conditions_level1_x(
    T *phi, 
    ocu::BoundaryConditionSet bc,
    T hx, T hy, T hz, int xstride, int ystride, int gx, int gy, int gz, int nx, int ny, int nz)
{
  int j = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int k = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;

  j -= gy;
  k -= gz;


  if (j >= 0 && j < ny && k >= 0 && k < nz) {
      if (bc.xneg.type == ocu::BC_PERIODIC) {
        phi[-1 * xstride + j * ystride + k] = phi[(nx-1) * xstride + j * ystride + k];
      }
      else if (bc.xneg.type == ocu::BC_DIRICHELET) {
        phi[-1 * xstride + j * ystride + k] =  -phi[0 * xstride + j * ystride + k] + 2 * bc.xneg.value;
      }
      else { // (bc.xneg.type == ocu::BC_NEUMANN)
        phi[-1 * xstride + j * ystride + k] = phi[0 * xstride + j * ystride + k] - hx * bc.xneg.value;
      }


      if (bc.xpos.type == ocu::BC_PERIODIC) {
        phi[nx * xstride + j * ystride + k] = phi[0 * xstride + j * ystride + k];
      }
      else if (bc.xpos.type == ocu::BC_DIRICHELET) {
        phi[nx * xstride + j * ystride + k] =  -phi[(nx-1) * xstride + j * ystride + k] + 2 * bc.xpos.value;
      }
      else { // (bc.xpos.type == ocu::BC_NEUMANN)
        phi[nx * xstride + j * ystride + k] =  phi[(nx-1) * xstride  + j * ystride + k] + hx * bc.xpos.value;
      }
  }
}


template<typename T>
__global__ void kernel_apply_3d_mac_boundary_conditions_level2_z(
    T *u, T *v, T *w, 
    ocu::BoundaryConditionSet bc,
    T hx, T hy, T hz, int xstride, int ystride, int gx, int gy, int gz, int nx, int ny, int nz)
{
  int i = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
  i -= gx;
  j -= gy;

  if (i >= -1 && i <= nx && j >= -1 && j <= ny) {

    if (bc.zneg.type == ocu::BC_PERIODIC) {
      // normal
      w[ i * xstride + j * ystride + -1] = w[i * xstride + j * ystride + (nz-1)];
      w[ i * xstride + j * ystride + -2] = w[i * xstride + j * ystride + (nz-2)];
      // face at (i,j,0) must be calculated by the outside application

      // tangential
      u[i * xstride + j * ystride + -1] = u[i * xstride + j * ystride + (nz-1)];
      u[i * xstride + j * ystride + -2] = u[i * xstride + j * ystride + (nz-2)];
      v[i * xstride + j * ystride + -1] = v[i * xstride + j * ystride + (nz-1)];
      v[i * xstride + j * ystride + -2] = v[i * xstride + j * ystride + (nz-2)];
    }
    else if (bc.zneg.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      w[ i * xstride + j * ystride + -2] = bc.zneg.value;
      w[ i * xstride + j * ystride + -1] = bc.zneg.value;
      w[ i * xstride + j * ystride +  0] = bc.zneg.value;

      // tangential
      u[i * xstride + j * ystride + -1] = u[i * xstride + j * ystride + 0] * (1 - 2 * bc.zneg.aux_value);
      u[i * xstride + j * ystride + -2] = u[i * xstride + j * ystride + 0] * (1 - 2 * bc.zneg.aux_value); // don't know about this?
      v[i * xstride + j * ystride + -1] = v[i * xstride + j * ystride + 0] * (1 - 2 * bc.zneg.aux_value);
      v[i * xstride + j * ystride + -2] = v[i * xstride + j * ystride + 0] * (1 - 2 * bc.zneg.aux_value);
    }

    if (bc.zpos.type == ocu::BC_PERIODIC) {
      // normal
      w[ i * xstride + j * ystride + (nz  )] = w[i * xstride + j * ystride + 0];
      w[ i * xstride + j * ystride + (nz+1)] = w[i * xstride + j * ystride + 1];
      w[ i * xstride + j * ystride + (nz+2)] = w[i * xstride + j * ystride + 2];

      // tangential
      u[i * xstride + j * ystride + (nz  )] = u[i * xstride + j * ystride + 0];
      u[i * xstride + j * ystride + (nz+1)] = u[i * xstride + j * ystride + 1];
      v[i * xstride + j * ystride + (nz  )] = v[i * xstride + j * ystride + 0];
      v[i * xstride + j * ystride + (nz+1)] = v[i * xstride + j * ystride + 1];
    }
    else if (bc.zpos.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      w[ i * xstride + j * ystride + (nz  )] = bc.zpos.value;
      w[ i * xstride + j * ystride + (nz+1)] = bc.zpos.value;
      w[ i * xstride + j * ystride + (nz+2)] = bc.zpos.value;

      // tangential
      u[i * xstride + j * ystride + (nz  )] = u[i * xstride + j * ystride + (nz-1)] * (1 - 2 * bc.zpos.aux_value);
      u[i * xstride + j * ystride + (nz+1)] = u[i * xstride + j * ystride + (nz-1)] * (1 - 2 * bc.zpos.aux_value);
      v[i * xstride + j * ystride + (nz  )] = v[i * xstride + j * ystride + (nz-1)] * (1 - 2 * bc.zpos.aux_value);
      v[i * xstride + j * ystride + (nz+1)] = v[i * xstride + j * ystride + (nz-1)] * (1 - 2 * bc.zpos.aux_value);
    }
  }
}


template<typename T>
__global__ void kernel_apply_3d_mac_boundary_conditions_level2_y(
    T *u, T *v, T *w, 
    ocu::BoundaryConditionSet bc,
    T hx, T hy, T hz, int xstride, int ystride, int gx, int gy, int gz, int nx, int ny, int nz)
{
  int i = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int k = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
  i -= gx;
  k -= gz;


  // boundaries in +-y direction in the xz face 
  if (i >= -1 && i <= nx && k >= 0 && k < nz) {

    if (bc.yneg.type == ocu::BC_PERIODIC) {
      // normal
      v[i * xstride + -1 * ystride + k] = v[i * xstride + (ny-1) * ystride + k];
      v[i * xstride + -2 * ystride + k] = v[i * xstride + (ny-2) * ystride + k];
      // face at (i,0,k) must be calculated by the outside application

      // tangential
      u[i * xstride + -1 * ystride + k] = u[i * xstride + (ny-1) * ystride + k];
      u[i * xstride + -2 * ystride + k] = u[i * xstride + (ny-2) * ystride + k];
      w[i * xstride + -1 * ystride + k] = w[i * xstride + (ny-1) * ystride + k];
      w[i * xstride + -2 * ystride + k] = w[i * xstride + (ny-2) * ystride + k];
    }
    else if (bc.yneg.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      v[i * xstride + -2 * ystride + k] = bc.yneg.value;
      v[i * xstride + -1 * ystride + k] = bc.yneg.value;
      v[i * xstride +  0 * ystride + k] = bc.yneg.value;

      // tangential
      u[i * xstride + -1 * ystride + k] = u[ i * xstride + 0 * ystride + k] * (1 - 2 * bc.yneg.aux_value);
      u[i * xstride + -2 * ystride + k] = u[ i * xstride + 0 * ystride + k] * (1 - 2 * bc.yneg.aux_value);
      w[i * xstride + -1 * ystride + k] = w[ i * xstride + 0 * ystride + k] * (1 - 2 * bc.yneg.aux_value);
      w[i * xstride + -2 * ystride + k] = w[ i * xstride + 0 * ystride + k] * (1 - 2 * bc.yneg.aux_value);
    }

    if (bc.ypos.type == ocu::BC_PERIODIC) {
      // normal
      v[i * xstride + (ny  ) * ystride + k] = v[i * xstride + 0 * ystride + k];
      v[i * xstride + (ny+1) * ystride + k] = v[i * xstride + 1 * ystride + k];
      v[i * xstride + (ny+2) * ystride + k] = v[i * xstride + 2 * ystride + k];

      // tangential
      u[i * xstride + (ny  ) * ystride + k] = u[i * xstride + 0 * ystride + k];
      u[i * xstride + (ny+1) * ystride + k] = u[i * xstride + 1 * ystride + k];
      w[i * xstride + (ny  ) * ystride + k] = w[i * xstride + 0 * ystride + k];
      w[i * xstride + (ny+1) * ystride + k] = w[i * xstride + 1 * ystride + k];
    }
    else if (bc.ypos.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      v[i * xstride + (ny  ) * ystride + k] = bc.ypos.value;
      v[i * xstride + (ny+1) * ystride + k] = bc.ypos.value;
      v[i * xstride + (ny+2) * ystride + k] = bc.ypos.value;

      // tangential
      u[i * xstride + (ny  ) * ystride + k] = u[i * xstride + (ny-1) * ystride + k] * (1 - 2 * bc.ypos.aux_value);
      u[i * xstride + (ny+1) * ystride + k] = u[i * xstride + (ny-1) * ystride + k] * (1 - 2 * bc.ypos.aux_value);
      w[i * xstride + (ny  ) * ystride + k] = w[i * xstride + (ny-1) * ystride + k] * (1 - 2 * bc.ypos.aux_value);
      w[i * xstride + (ny+1) * ystride + k] = w[i * xstride + (ny-1) * ystride + k] * (1 - 2 * bc.ypos.aux_value);
    }
  }
}

template<typename T>
__global__ void kernel_apply_3d_mac_boundary_conditions_level2_x(
    T *u, T *v, T *w, 
    ocu::BoundaryConditionSet bc,
    T hx, T hy, T hz, int xstride, int ystride, int gx, int gy, int gz, int nx, int ny, int nz)
{
  int j = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int k = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
  j -= gy;
  k -= gz;


  // boundaries on the +-x direction in the yz face
  if (j >= 0 && j < ny && k >= 0 && k < nz) {

    if (bc.xneg.type == ocu::BC_PERIODIC) {
      // normal
      u[-1 * xstride + j * ystride + k] = u[(nx-1) * xstride + j * ystride + k];
      u[-2 * xstride + j * ystride + k] = u[(nx-2) * xstride + j * ystride + k];
      // face at (0,j,k) must be calculated by the outside application

      // tangential
      v[-1 * xstride + j * ystride + k] = v[(nx-1) * xstride + j * ystride + k];
      v[-2 * xstride + j * ystride + k] = v[(nx-2) * xstride + j * ystride + k];
      w[-1 * xstride + j * ystride + k] = w[(nx-1) * xstride + j * ystride + k];
      w[-2 * xstride + j * ystride + k] = w[(nx-2) * xstride + j * ystride + k];

    }
    else if (bc.xneg.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      u[-2 * xstride + j * ystride + k] = bc.xneg.value;
      u[-1 * xstride + j * ystride + k] = bc.xneg.value;
      u[ 0 * xstride + j * ystride + k] = bc.xneg.value;

      // tangential
      v[-1 * xstride + j * ystride + k] = v[ 0 * xstride + j * ystride + k] * (1 - 2 * bc.xneg.aux_value);
      v[-2 * xstride + j * ystride + k] = v[ 0 * xstride + j * ystride + k] * (1 - 2 * bc.xneg.aux_value);
      w[-1 * xstride + j * ystride + k] = w[ 0 * xstride + j * ystride + k] * (1 - 2 * bc.xneg.aux_value);
      w[-2 * xstride + j * ystride + k] = w[ 0 * xstride + j * ystride + k] * (1 - 2 * bc.xneg.aux_value);
    }

    if (bc.xpos.type == ocu::BC_PERIODIC) {
      // normal
      u[(nx  ) * xstride + j * ystride + k] = u[0 * xstride + j * ystride + k];
      u[(nx+1) * xstride + j * ystride + k] = u[1 * xstride + j * ystride + k];
      u[(nx+2) * xstride + j * ystride + k] = u[2 * xstride + j * ystride + k];

      // tangential
      v[(nx  ) * xstride + j * ystride + k] = v[0 * xstride + j * ystride + k];
      v[(nx+1) * xstride + j * ystride + k] = v[1 * xstride + j * ystride + k];
      w[(nx  ) * xstride + j * ystride + k] = w[0 * xstride + j * ystride + k];
      w[(nx+1) * xstride + j * ystride + k] = w[1 * xstride + j * ystride + k];
    }
    else if (bc.xpos.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      u[(nx  ) * xstride + j * ystride + k] = bc.xpos.value;
      u[(nx+1) * xstride + j * ystride + k] = bc.xpos.value;
      u[(nx+2) * xstride + j * ystride + k] = bc.xpos.value;

      // tangential
      v[(nx  ) * xstride + j * ystride + k] = v[(nx-1) * xstride + j * ystride + k] * (1 - 2 * bc.xpos.aux_value);
      v[(nx+1) * xstride + j * ystride + k] = v[(nx-1) * xstride + j * ystride + k] * (1 - 2 * bc.xpos.aux_value);
      w[(nx  ) * xstride + j * ystride + k] = w[(nx-1) * xstride + j * ystride + k] * (1 - 2 * bc.xpos.aux_value);
      w[(nx+1) * xstride + j * ystride + k] = w[(nx-1) * xstride + j * ystride + k] * (1 - 2 * bc.xpos.aux_value);
    }
  }
}

template<typename T>
__global__ void kernel_apply_3d_mac_boundary_conditions_level1_z(
    T *u, T *v, T *w, 
    ocu::BoundaryConditionSet bc,
    T hx, T hy, T hz, int xstride, int ystride, int gx, int gy, int gz, int nx, int ny, int nz)
{
  int i = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
  i -= gx;
  j -= gy;

  if (i >= -1 && i <= nx && j >= -1 && j <= ny) {

    if (bc.zneg.type == ocu::BC_PERIODIC) {
      // normal
      w[ i * xstride + j * ystride + -1] = w[i * xstride + j * ystride + (nz-1)];
      // face at (i,j,0) must be calculated by the outside application

      // tangential
      u[i * xstride + j * ystride + -1] = u[i * xstride + j * ystride + (nz-1)];
      v[i * xstride + j * ystride + -1] = v[i * xstride + j * ystride + (nz-1)];
    }
    else if (bc.zneg.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      w[ i * xstride + j * ystride + -1] = bc.zneg.value;
      w[ i * xstride + j * ystride +  0] = bc.zneg.value;

      // tangential
      u[i * xstride + j * ystride + -1] = u[i * xstride + j * ystride + 0] * (1.0f - 2.0f * bc.zneg.aux_value);
      v[i * xstride + j * ystride + -1] = v[i * xstride + j * ystride + 0] * (1.0f - 2.0f * bc.zneg.aux_value);
    }

    if (bc.zpos.type == ocu::BC_PERIODIC) {
      // normal
      w[ i * xstride + j * ystride + (nz  )] = w[i * xstride + j * ystride + 0];
      w[ i * xstride + j * ystride + (nz+1)] = w[i * xstride + j * ystride + 1];

      // tangential
      u[i * xstride + j * ystride + nz] = u[i * xstride + j * ystride + 0];
      v[i * xstride + j * ystride + nz] = v[i * xstride + j * ystride + 0];
    }
    else if (bc.zpos.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      w[ i * xstride + j * ystride + (nz  )] = bc.zpos.value;
      w[ i * xstride + j * ystride + (nz+1)] = bc.zpos.value;

      // tangential
      u[i * xstride + j * ystride + nz] = u[i * xstride + j * ystride + (nz-1)] * (1.0f - 2.0f * bc.zpos.aux_value);
      v[i * xstride + j * ystride + nz] = v[i * xstride + j * ystride + (nz-1)] * (1.0f - 2.0f * bc.zpos.aux_value);
    }
  }
}

template<typename T>
__global__ void kernel_apply_3d_mac_boundary_conditions_level1_y(
    T *u, T *v, T *w, 
    ocu::BoundaryConditionSet bc,
    T hx, T hy, T hz, int xstride, int ystride, int gx, int gy, int gz, int nx, int ny, int nz)
{
  int i = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int k = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
  i -= gx;
  k -= gz;


  // boundaries in +-y direction in the xz face 
  if (i >= -1 && i <= nx && k >= 0 && k < nz) {

    if (bc.yneg.type == ocu::BC_PERIODIC) {
      // normal
      v[i * xstride + -1 * ystride + k] = v[i * xstride + (ny-1) * ystride + k];
      // face at (i,0,k) must be calculated by the outside application


      // tangential
      u[i * xstride + -1 * ystride + k] = u[i * xstride + (ny-1) * ystride + k];
      w[i * xstride + -1 * ystride + k] = w[i * xstride + (ny-1) * ystride + k];
    }
    else if (bc.yneg.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      v[i * xstride + -1 * ystride + k] = bc.yneg.value;
      v[i * xstride +  0 * ystride + k] = bc.yneg.value;

      // tangential
      u[i * xstride + -1 * ystride + k] = u[ i * xstride + 0 * ystride + k] * (1.0f - 2.0f * bc.yneg.aux_value);
      w[i * xstride + -1 * ystride + k] = w[ i * xstride + 0 * ystride + k] * (1.0f - 2.0f * bc.yneg.aux_value);
    }

    if (bc.ypos.type == ocu::BC_PERIODIC) {
      // normal
      v[i * xstride + (ny  ) * ystride + k] = v[i * xstride + 0 * ystride + k];
      v[i * xstride + (ny+1) * ystride + k] = v[i * xstride + 1 * ystride + k];

      // tangential
      u[i * xstride + ny * ystride + k] = u[i * xstride + 0 * ystride + k];
      w[i * xstride + ny * ystride + k] = w[i * xstride + 0 * ystride + k];
    }
    else if (bc.ypos.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      v[i * xstride + (ny  ) * ystride + k] = bc.ypos.value;
      v[i * xstride + (ny+1) * ystride + k] = bc.ypos.value;

      // tangential
      u[i * xstride + ny * ystride + k] = u[i * xstride + (ny-1) * ystride + k] * (1.0f - 2.0f * bc.ypos.aux_value);
      w[i * xstride + ny * ystride + k] = w[i * xstride + (ny-1) * ystride + k] * (1.0f - 2.0f * bc.ypos.aux_value);
    }
  }
}

template<typename T>
__global__ void kernel_apply_3d_mac_boundary_conditions_level1_x(
    T *u, T *v, T *w, 
    ocu::BoundaryConditionSet bc,
    T hx, T hy, T hz, int xstride, int ystride, int gx, int gy, int gz, int nx, int ny, int nz)
{
  int j = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int k = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
  j -= gy;
  k -= gz;


  // boundaries on the +-x direction in the yz face
  if (j >= 0 && j < ny && k >= 0 && k < nz) {

    if (bc.xneg.type == ocu::BC_PERIODIC) {
      // normal
      u[-1 * xstride + j * ystride + k] = u[(nx-1) * xstride + j * ystride + k];
      // face at (0,j,k) must be calculated by the outside application

      // tangential
      v[-1 * xstride + j * ystride + k] = v[(nx-1) * xstride + j * ystride + k];
      w[-1 * xstride + j * ystride + k] = w[(nx-1) * xstride + j * ystride + k];

    }
    else if (bc.xneg.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      u[-1 * xstride + j * ystride + k] = bc.xneg.value;
      u[ 0 * xstride + j * ystride + k] = bc.xneg.value;

      // tangential
      v[-1 * xstride + j * ystride + k] = v[ 0 * xstride + j * ystride + k] * (1.0f - 2.0f * bc.xneg.aux_value);
      w[-1 * xstride + j * ystride + k] = w[ 0 * xstride + j * ystride + k] * (1.0f - 2.0f * bc.xneg.aux_value);
    }

    if (bc.xpos.type == ocu::BC_PERIODIC) {
      // normal
      u[(nx  ) * xstride + j * ystride + k] = u[0 * xstride + j * ystride + k];
      u[(nx+1) * xstride + j * ystride + k] = u[1 * xstride + j * ystride + k];

      // tangential
      v[nx * xstride + j * ystride + k] = v[0 * xstride + j * ystride + k];
      w[nx * xstride + j * ystride + k] = w[0 * xstride + j * ystride + k];
    }
    else if (bc.xpos.type == ocu::BC_FORCED_INFLOW_VARIABLE_SLIP) {
      // normal
      u[(nx  ) * xstride + j * ystride + k] = bc.xpos.value;
      u[(nx+1) * xstride + j * ystride + k] = bc.xpos.value;

      // tangential
      v[nx * xstride + j * ystride + k] = v[(nx-1) * xstride + j * ystride + k] * (1.0f - 2.0f * bc.xpos.aux_value);
      w[nx * xstride + j * ystride + k] = w[(nx-1) * xstride + j * ystride + k] * (1.0f - 2.0f * bc.xpos.aux_value);
    }
  }
}

namespace ocu {


template<typename T>
bool 
apply_3d_boundary_conditions_level1_nocorners(
  Grid3DDevice<T> &grid,  
  const BoundaryConditionSet &bc, 
  double hx, double hy, double hz)
{
  if (!bc.check_type(BC_PERIODIC, BC_DIRICHELET, BC_NEUMANN)) {
    printf("[ERROR] apply_3d_boundary_conditions_level1_nocorners - invalid boundary condition types %d, %d, %d, %d, %d,%d\n", 
      bc.xpos.type, bc.xneg.type, bc.ypos.type, bc.yneg.type, bc.zpos.type, bc.zneg.type);
    return false;
  }

  if (grid.gx() < 1 || grid.gy() < 1 || grid.gz() < 1) {
    printf("[ERROR] apply_3d_boundary_conditions_level1_nocorners - must have at least 1 ghost point\n");
    return false;
  }

  int nx = grid.nx();
  int ny = grid.ny();
  int nz = grid.nz();

  int max_nxny = nx > ny ? nx : ny;
  int max_nynz = ny > nz ? ny : nz;

  dim3 Dg((max_nxny+15) / 16, (max_nynz+15) / 16);
  dim3 Db(16, 16);
  
  KernelWrapper wrapper;
  wrapper.PreKernel();

  kernel_apply_3d_boundary_conditions_level1_nocorners<<<Dg, Db>>>(&grid.at(0,0,0), bc,
    (T)hx, (T)hy, (T)hz, grid.xstride(), grid.ystride(), nx, ny, nz);
  return wrapper.PostKernel("kernel_apply_3d_boundary_conditions_level1_nocorners", nz);
}


template<typename T>
bool 
apply_3d_boundary_conditions_level1(
  Grid3DDevice<T> &grid,  
  const BoundaryConditionSet &bc, 
  double hx, double hy, double hz)
{
  if (!bc.check_type(BC_PERIODIC, BC_DIRICHELET, BC_NEUMANN)) {
    printf("[ERROR] apply_3d_boundary_conditions_level1 - invalid boundary condition types %d, %d, %d, %d, %d,%d\n", 
      bc.xpos.type, bc.xneg.type, bc.ypos.type, bc.yneg.type, bc.zpos.type, bc.zneg.type);
    return false;
  }
 
  if (grid.gx() < 1 || grid.gy() < 1 || grid.gz() < 1) {
    printf("[ERROR] apply_3d_boundary_conditions_level1 - must have at least 1 ghost point\n");
    return false;
  }

  int nx = grid.nx();
  int ny = grid.ny();
  int nz = grid.nz();
  int gx = grid.gx();
  int gy = grid.gy();
  int gz = grid.gz();

  int max_nxny = std::max(nx+gx, ny+gy);
  int max_nynz = std::max(ny+gy, nz+gz);

  KernelWrapper wrapper;

  {
    dim3 Dg((ny+2*gy+15) / 16, (nz+2*gz+15) / 16);
    dim3 Db(16, 16);

    wrapper.PreKernel();
    kernel_apply_3d_boundary_conditions_level1_x<<<Dg, Db>>>(&grid.at(0,0,0), bc,
      (T)hx, (T)hy, (T)hz, grid.xstride(), grid.ystride(), gx, gy, gz, nx, ny, nz);
    if (!wrapper.PostKernel("kernel_apply_3d_boundary_conditions_level1_x", nz))
      return false;
  }

  {
    dim3 Dg((nx+2*gx+15) / 16, (nz+2*gz+15) / 16);
    dim3 Db(16, 16);
    
    wrapper.PreKernel();
    kernel_apply_3d_boundary_conditions_level1_y<<<Dg, Db>>>(&grid.at(0,0,0), bc,
      (T)hx, (T)hy, (T)hz, grid.xstride(), grid.ystride(), gx, gy, gz, nx, ny, nz);
    if (!wrapper.PostKernel("kernel_apply_3d_boundary_conditions_level1_y", nz))
      return false;
  }

  {
    dim3 Dg((nx+2*gx+15) / 16, (ny+2*gy+15) / 16);
    dim3 Db(16, 16);

    wrapper.PreKernel();
    kernel_apply_3d_boundary_conditions_level1_z<<<Dg, Db>>>(&grid.at(0,0,0), bc,
      (T)hx, (T)hy, (T)hz, grid.xstride(), grid.ystride(), gx, gy, gz, nx, ny, nz);
    if (!wrapper.PostKernel("kernel_apply_3d_boundary_conditions_level1_z", nz))
      return false;
  }

  return true;
}


template<typename T>
bool 
apply_3d_mac_boundary_conditions_level1(
  Grid3DDevice<T> &u_grid, Grid3DDevice<T> &v_grid, Grid3DDevice<T> &w_grid,  
  const BoundaryConditionSet &bc, 
  double hx, double hy, double hz)
{
  if (!bc.check_type(BC_PERIODIC, BC_FORCED_INFLOW_VARIABLE_SLIP)) {
    printf("[ERROR] apply_3d_mac_boundary_conditions_level1 - invalid boundary condition types %d, %d, %d, %d, %d,%d\n", 
      bc.xpos.type, bc.xneg.type, bc.ypos.type, bc.yneg.type, bc.zpos.type, bc.zneg.type);
    return false;
  }

  if (!u_grid.check_layout_match(v_grid) || !u_grid.check_layout_match(w_grid)) {
    printf("[ERROR] apply_3d_mac_boundary_conditions_level1 - layout mismatch\n");
    return false;
  }

  if (u_grid.gx() < 1 || u_grid.gy() < 1 || u_grid.gz() < 1) {
    printf("[ERROR] apply_3d_mac_boundary_conditions_level1 - must have at least 1 ghost point\n");
    return false;
  }


  int nx = u_grid.nx()-1;
  int ny = u_grid.ny();
  int nz = u_grid.nz();
  int gx = u_grid.gx();
  int gy = u_grid.gy();
  int gz = u_grid.gz();

  if (v_grid.nx() != nx || v_grid.ny() != ny+1 || v_grid.nz() != nz) {
    printf("[ERROR] apply_3d_mac_boundary_conditions_level1 - v_grid dimension mismatch (%d, %d, %d) != (%d, %d, %d)\n", v_grid.nx(), v_grid.ny(), v_grid.nz(), nx, ny+1, nz);
    return false;
  }

  if (w_grid.nx() != nx || w_grid.ny() != ny || w_grid.nz() != nz+1) {
    printf("[ERROR] apply_3d_mac_boundary_conditions_level1 - w_grid dimension mismatch (%d, %d, %d) != (%d, %d, %d)\n", v_grid.nx(), v_grid.ny(), v_grid.nz(), nx, ny, nz+1);
    return false;
  }

  KernelWrapper wrapper;

  {
    dim3 Dg((ny+2*gy+15) / 16, (nz+2*gz+15) / 16);
    dim3 Db(16, 16);

    wrapper.PreKernel();
    kernel_apply_3d_mac_boundary_conditions_level1_x<<<Dg, Db>>>(&u_grid.at(0,0,0), &v_grid.at(0,0,0), &w_grid.at(0,0,0), bc,
      (T)hx, (T)hy, (T)hz, u_grid.xstride(), u_grid.ystride(), gx, gy, gz, nx, ny, nz);
    if (!wrapper.PostKernel("kernel_apply_3d_mac_boundary_conditions_level1_x", nz))
      return false;
  }

  {
    dim3 Dg((nx+2*gx+15) / 16, (nz+2*gz+15) / 16);
    dim3 Db(16, 16);

    wrapper.PreKernel();
    kernel_apply_3d_mac_boundary_conditions_level1_y<<<Dg, Db>>>(&u_grid.at(0,0,0), &v_grid.at(0,0,0), &w_grid.at(0,0,0), bc,
      (T)hx, (T)hy, (T)hz, u_grid.xstride(), u_grid.ystride(), gx, gy, gz, nx, ny, nz);
    if (!wrapper.PostKernel("kernel_apply_3d_mac_boundary_conditions_level1_y", nz))
      return false;
  }

  {
    dim3 Dg((nx+2*gx+15) / 16, (ny+2*gy+15) / 16);
    dim3 Db(16, 16);

    wrapper.PreKernel();
    kernel_apply_3d_mac_boundary_conditions_level1_z<<<Dg, Db>>>(&u_grid.at(0,0,0), &v_grid.at(0,0,0), &w_grid.at(0,0,0), bc,
      (T)hx, (T)hy, (T)hz, u_grid.xstride(), u_grid.ystride(), gx, gy, gz, nx, ny, nz);
    if (!wrapper.PostKernel("kernel_apply_3d_mac_boundary_conditions_level1_z", nz))
      return false;
  }

  return true;
}



template<typename T>
bool 
apply_3d_boundary_conditions_level2_nocorners(
  Grid3DDevice<T> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz)
{
  if (!bc.check_type(BC_PERIODIC, BC_DIRICHELET, BC_NEUMANN)) {
    printf("[ERROR] apply_3d_boundary_conditions_level2_nocorners - invalid boundary condition types %d, %d, %d, %d, %d,%d\n", 
      bc.xpos.type, bc.xneg.type, bc.ypos.type, bc.yneg.type, bc.zpos.type, bc.zneg.type);
    return false;
  }

  if (grid.gx() < 2 || grid.gy() < 2 || grid.gz() < 2) {
    printf("[ERROR] apply_3d_boundary_conditions_level2_nocorners - must have at least 2 ghost points\n");
    return false;
  }

  int nx = grid.nx();
  int ny = grid.ny();
  int nz = grid.nz();

  int max_nxny = nx > ny ? nx : ny;
  int max_nynz = ny > nz ? ny : nz;

  dim3 Dg((max_nxny+15) / 16, (max_nynz+15) / 16);
  dim3 Db(16, 16);

  KernelWrapper wrapper;
  wrapper.PreKernel();
  kernel_apply_3d_boundary_conditions_level2_nocorners<<<Dg, Db>>>(&grid.at(0,0,0), bc,
    (T)hx, (T)hy, (T)hz, grid.xstride(), grid.ystride(), nx, ny, nz);
  return wrapper.PostKernel("kernel_apply_3d_boundary_conditions_level2_nocorners", nz);
}


template<typename T>
bool 
apply_3d_mac_boundary_conditions_level2(
  Grid3DDevice<T> &u_grid, Grid3DDevice<T> &v_grid, Grid3DDevice<T> &w_grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz)
{
  if (!bc.check_type(BC_PERIODIC, BC_FORCED_INFLOW_VARIABLE_SLIP)) {
    printf("[ERROR] apply_3d_mac_boundary_conditions_level2 - invalid boundary condition types %d, %d, %d, %d, %d,%d\n", 
      bc.xpos.type, bc.xneg.type, bc.ypos.type, bc.yneg.type, bc.zpos.type, bc.zneg.type);
    return false;
  }

  if (!u_grid.check_layout_match(v_grid) || !u_grid.check_layout_match(w_grid)) {
    printf("[ERROR] apply_3d_mac_boundary_conditions_level2 - layout mismatch\n");
    return false;
  }

  if (u_grid.gx() < 2 || u_grid.gy() < 2 || u_grid.gz() < 2) {
    printf("[ERROR] apply_3d_mac_boundary_conditions_level2 - must have at least 2 ghost points\n");
    return false;
  }

  int nx = u_grid.nx()-u_grid.gz();
  int ny = u_grid.ny();
  int nz = u_grid.nz();
  int gx = u_grid.gx();
  int gy = u_grid.gy();
  int gz = u_grid.gz();

  if (v_grid.nx() != nx || v_grid.ny() != ny+1 || v_grid.nz() != nz) {
    printf("[ERROR] apply_3d_mac_boundary_conditions_level2 - v_grid dimension mismatch (%d, %d, %d) != (%d, %d, %d)\n", v_grid.nx(), v_grid.ny(), v_grid.nz(), nx, ny+1, nz);
    return false;
  }

  if (w_grid.nx() != nx || w_grid.ny() != ny || w_grid.nz() != nz+1) {
    printf("[ERROR] apply_3d_mac_boundary_conditions_level2 - w_grid dimension mismatch (%d, %d, %d) != (%d, %d, %d)\n", v_grid.nx(), v_grid.ny(), v_grid.nz(), nx, ny, nz+1);
    return false;
  }

  KernelWrapper wrapper;

  {
    dim3 Dg((ny+2*gy+15) / 16, (nz+2*gz+15) / 16);
    dim3 Db(16, 16);

    wrapper.PreKernel();
    kernel_apply_3d_mac_boundary_conditions_level2_x<<<Dg, Db>>>(&u_grid.at(0,0,0), &v_grid.at(0,0,0), &w_grid.at(0,0,0), bc,
      (T)hx, (T)hy, (T)hz, u_grid.xstride(), u_grid.ystride(), gx, gy, gz, nx, ny, nz);
    if (!wrapper.PostKernel("kernel_apply_3d_mac_boundary_conditions_level2_x", nz))
      return false;
  }

  {
    dim3 Dg((nx+2*gx+15) / 16, (nz+2*gz+15) / 16);
    dim3 Db(16, 16);

    wrapper.PreKernel();
    kernel_apply_3d_mac_boundary_conditions_level2_y<<<Dg, Db>>>(&u_grid.at(0,0,0), &v_grid.at(0,0,0), &w_grid.at(0,0,0), bc,
      (T)hx, (T)hy, (T)hz, u_grid.xstride(), u_grid.ystride(), gx, gy, gz, nx, ny, nz);
    if (!wrapper.PostKernel("kernel_apply_3d_mac_boundary_conditions_level2_y", nz))
      return false;
  }

  {
    dim3 Dg((nx+2*gx+15) / 16, (ny+2*gy+15) / 16);
    dim3 Db(16, 16);

    wrapper.PreKernel();
    kernel_apply_3d_mac_boundary_conditions_level2_z<<<Dg, Db>>>(&u_grid.at(0,0,0), &v_grid.at(0,0,0), &w_grid.at(0,0,0), bc,
      (T)hx, (T)hy, (T)hz, u_grid.xstride(), u_grid.ystride(), gx, gy, gz, nx, ny, nz);
    if (!wrapper.PostKernel("kernel_apply_3d_mac_boundary_conditions_level2_z", nz))
      return false;
  }

  return true;
}

template<typename T>
bool 
apply_3d_boundary_conditions_level1_nocorners(
  Grid3DHost<T> &phi,  
  const BoundaryConditionSet &bc, 
  double hx, double hy, double hz)
{
  if (!bc.check_type(BC_PERIODIC, BC_DIRICHELET, BC_NEUMANN)) {
    printf("[ERROR] apply_3d_boundary_conditions_level1_nocorners - invalid boundary condition types %d, %d, %d, %d, %d,%d\n", 
      bc.xpos.type, bc.xneg.type, bc.ypos.type, bc.yneg.type, bc.zpos.type, bc.zneg.type);
    return false;
  }

  int nx = phi.nx();
  int ny = phi.ny();
  int nz = phi.nz();

  int i,j,k;

  for (i=0; i < nx; i++)
    for (j=0; j < ny; j++) {
      if (bc.zneg.type == ocu::BC_PERIODIC) {
        phi.at(i,j,-1) = phi.at(i,j,nz-1);
      }
      else if (bc.zneg.type == ocu::BC_DIRICHELET) {
        phi.at(i,j,-1) = -phi.at(i,j,0) + (T)2 * (T)bc.zneg.value;
      }
      else { // (bc.zneg.type == ocu::BC_NEUMANN)
        phi.at(i,j,-1) = phi.at(i,j,0) - (T)hz * (T)bc.zneg.value;
      }


      if (bc.zpos.type == ocu::BC_PERIODIC) {
        phi.at(i,j,nz) = phi.at(i,j,0);
      }
      else if (bc.zpos.type == ocu::BC_DIRICHELET) {
        phi.at(i,j,nz) = -phi.at(i,j,nz-1) + (T)2 * (T)bc.zpos.value;
      }
      else { // (bc.zpos.type == ocu::BC_NEUMANN)
        phi.at(i,j,nz) = phi.at(i,j,nz-1) + (T)hz * (T)bc.zpos.value;
      }
    }


  for (i=0; i < nx; i++)
    for (k=0; k < nz; k++) {
      
      if (bc.yneg.type == ocu::BC_PERIODIC) {
        phi.at(i,-1,k) = phi.at(i,ny-1,k);
      }
      else if (bc.yneg.type == ocu::BC_DIRICHELET) {
        phi.at(i,-1,k) = -phi.at(i,0,k) + (T)2 * (T)bc.yneg.value;
      }
      else { // (bc.yneg.type == ocu::BC_NEUMANN)
        phi.at(i,-1,k) = phi.at(i,0,k) - (T)hy * (T)bc.yneg.value;
      }


      if (bc.ypos.type == ocu::BC_PERIODIC) {
        phi.at(i,ny,k) = phi.at(i,0,k);
      }
      else if (bc.ypos.type == ocu::BC_DIRICHELET) {
        phi.at(i,ny,k) = -phi.at(i,ny-1,k) + (T)2 * (T)bc.ypos.value;
      }
      else { // (bc.ypos.type == ocu::BC_NEUMANN)
        phi.at(i,ny,k) = phi.at(i,ny-1,k) + (T)hy * (T)bc.ypos.value;
      }
    }

  for (j=0; j < ny; j++)
    for (k=0; k < nz; k++) {
      if (bc.xneg.type == ocu::BC_PERIODIC) {
        phi.at(-1,j,k) = phi.at(nx-1,j,k);
      }
      else if (bc.xneg.type == ocu::BC_DIRICHELET) {
        phi.at(-1,j,k) = -phi.at(0,j,k) + (T)2 * (T)bc.xneg.value;
      }
      else { // (bc.xneg.type == ocu::BC_NEUMANN)
        phi.at(-1,j,k) = phi.at(0,j,k) - (T)hx * (T)bc.xneg.value;
      }

      if (bc.xpos.type == ocu::BC_PERIODIC) {
        phi.at(nx,j,k) = phi.at(0,j,k);
      }
      else if (bc.xpos.type == ocu::BC_DIRICHELET) {
        phi.at(nx,j,k) = -phi.at(nx-1,j,k) + (T)2 * (T)bc.xpos.value;
      }
      else { // (bc.xpos.type == ocu::BC_NEUMANN)
        phi.at(nx,j,k) =  phi.at(nx-1,j,k) + (T)hx * (T)bc.xpos.value;
      }
    }

  return true;
}


template bool apply_3d_mac_boundary_conditions_level1(Grid3DDevice<float> &u_grid, Grid3DDevice<float> &v_grid, Grid3DDevice<float> &w_grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template bool apply_3d_mac_boundary_conditions_level2(Grid3DDevice<float> &u_grid, Grid3DDevice<float> &v_grid, Grid3DDevice<float> &w_grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template bool apply_3d_boundary_conditions_level1(Grid3DDevice<float> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template bool apply_3d_boundary_conditions_level1_nocorners(Grid3DDevice<float> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template bool apply_3d_boundary_conditions_level2_nocorners(Grid3DDevice<float> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template bool apply_3d_boundary_conditions_level1_nocorners(Grid3DHost<float> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template bool apply_3d_boundary_conditions_level1_nocorners(Grid3DHost<double> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

#ifdef OCU_DOUBLESUPPORT

template bool apply_3d_mac_boundary_conditions_level1(Grid3DDevice<double> &u_grid, Grid3DDevice<double> &v_grid, Grid3DDevice<double> &w_grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template bool apply_3d_mac_boundary_conditions_level2(Grid3DDevice<double> &u_grid, Grid3DDevice<double> &v_grid, Grid3DDevice<double> &w_grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template bool apply_3d_boundary_conditions_level1(Grid3DDevice<double> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template bool apply_3d_boundary_conditions_level1_nocorners(Grid3DDevice<double> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template bool apply_3d_boundary_conditions_level2_nocorners(Grid3DDevice<double> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

#endif // OCU_DOUBLESUPPORT

} // end namespace 

