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

#include "ocuutil/float_routines.h"
#include "ocuequation/sol_mgpressure3d.h"



///////////////////////////////////////////////////////////////////////////////////////////////////
//
// float routines
//
///////////////////////////////////////////////////////////////////////////////////////////////////


texture<float, 1, cudaReadModeElementType> tex_U_C;
texture<float, 1, cudaReadModeElementType> tex_U_F;
texture<float, 1, cudaReadModeElementType> tex_U;
texture<float, 1, cudaReadModeElementType> tex_B;


__global__ void Sol_MultigridPressure3DDeviceF_restrict(
  float *r_buffer, int r_xstride, int r_ystride, float *b_buffer, int b_xstride, int b_ystride, int bnx, int bny, int bnz, unsigned int blocksInY, float invBlocksInY)
{
  // the buffers start at (0,0,0)

  unsigned int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
  int i     = __umul24(blockIdxz ,blockDim.z) + threadIdx.z;
  int j     = __umul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int k     = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;


  int i_F = 2*i;
  int j_F = 2*j;
  int k_F = 2*k;

  // is it risky to use mul24 here?
  int r_idx = __mul24(i_F,r_xstride) + __mul24(j_F,r_ystride) + (k_F);

  // we only need to test if (i,j,k) is in bounds since r is guaranteed to have 2x the dimensions of b.
  if (i < bnx && j < bny && k < bnz) {
    b_buffer[__mul24(i,b_xstride) + __mul24(j,b_ystride) + (k)] = .125 * ( 
      r_buffer[r_idx                            ] + r_buffer[r_idx + 1                        ] +
      r_buffer[r_idx     + r_ystride            ] + r_buffer[r_idx + 1 + r_ystride            ] +
      r_buffer[r_idx     +             r_xstride] + r_buffer[r_idx + 1             + r_xstride] +
      r_buffer[r_idx     + r_ystride + r_xstride] + r_buffer[r_idx + 1 + r_ystride + r_xstride]);
  }
}


__global__ void Sol_MultigridPressure3DDeviceF_calculate_residual(
  float *b_buffer, float *r_buffer, int xstride, int ystride, int nx, int ny, int nz, int tex_offset, 
  float fx_div_hsq, float fy_div_hsq, float fz_div_hsq, float diag, unsigned int blocksInY, float invBlocksInY)
{
  unsigned int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  // coalescing shift

  int idx = __mul24(i,xstride) + __mul24(j,ystride) + (k);
  int tidx = idx + tex_offset;

  if (i < nx && j < ny && k < nz) {
    r_buffer[idx] = b_buffer[idx] -
      (fz_div_hsq*(tex1Dfetch(tex_U,tidx - 1      ) + tex1Dfetch(tex_U,tidx + 1      )) + 
       fy_div_hsq*(tex1Dfetch(tex_U,tidx - ystride) + tex1Dfetch(tex_U,tidx + ystride)) +
       fx_div_hsq*(tex1Dfetch(tex_U,tidx - xstride) + tex1Dfetch(tex_U,tidx + xstride)) 
        -diag*tex1Dfetch(tex_U,tidx));
  }
}



__global__ void Sol_MultigridPressure3DDeviceF_prolong(
  float *uf_buffer, int uf_xstride, int uf_ystride, int uf_tex_offset, 
  int uc_xstride, int uc_ystride, int ucnx, int ucny, int ucnz, int uc_tex_offset, 
  unsigned int blocksInY, float invBlocksInY)
{
  unsigned int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
  int k_C     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j_C     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i_C     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  if (i_C < ucnx && j_C < ucny && k_C < ucnz) {
    int i = 2*i_C+1;
    int j = 2*j_C+1;
    int k = 2*k_C+1;

    // indices used for accessing the texture must be offset, and are named with the "t" prefix
    int tindex_C = __mul24(i_C,uc_xstride) + __mul24(j_C,uc_ystride) + k_C + uc_tex_offset;
    int index_F  = __mul24(i  ,uf_xstride) + __mul24(j  ,uf_ystride) + k;
    int tindex_F = index_F + uf_tex_offset;

    float U_C_i_j_k_ = tex1Dfetch(tex_U_C,tindex_C                              );
    float U_C_ipj_k_ = tex1Dfetch(tex_U_C,tindex_C + uc_xstride                 );
    float U_C_i_jpk_ = tex1Dfetch(tex_U_C,tindex_C              + uc_ystride    );
    float U_C_ipjpk_ = tex1Dfetch(tex_U_C,tindex_C + uc_xstride + uc_ystride    );
    float U_C_i_j_kp = tex1Dfetch(tex_U_C,tindex_C                           + 1);
    float U_C_ipj_kp = tex1Dfetch(tex_U_C,tindex_C + uc_xstride              + 1);
    float U_C_i_jpkp = tex1Dfetch(tex_U_C,tindex_C              + uc_ystride + 1);
    float U_C_ipjpkp = tex1Dfetch(tex_U_C,tindex_C + uc_xstride + uc_ystride + 1);

    float U_C_imj_k_ = tex1Dfetch(tex_U_C,tindex_C - uc_xstride                 );
    float U_C_i_jmk_ = tex1Dfetch(tex_U_C,tindex_C              - uc_ystride    );
    float U_C_imjmk_ = tex1Dfetch(tex_U_C,tindex_C - uc_xstride - uc_ystride    );
    float U_C_i_j_km = tex1Dfetch(tex_U_C,tindex_C                           - 1);
    float U_C_imj_km = tex1Dfetch(tex_U_C,tindex_C - uc_xstride              - 1);
    float U_C_i_jmkm = tex1Dfetch(tex_U_C,tindex_C              - uc_ystride - 1);   
    float U_C_imjmkm = tex1Dfetch(tex_U_C,tindex_C - uc_xstride - uc_ystride - 1);

    float U_C_imjpk_ = tex1Dfetch(tex_U_C,tindex_C - uc_xstride + uc_ystride    );
    float U_C_imj_kp = tex1Dfetch(tex_U_C,tindex_C - uc_xstride              + 1);

    float U_C_ipjmk_ = tex1Dfetch(tex_U_C,tindex_C + uc_xstride - uc_ystride    );
    float U_C_i_jmkp = tex1Dfetch(tex_U_C,tindex_C              - uc_ystride + 1);

    float U_C_ipj_km = tex1Dfetch(tex_U_C,tindex_C + uc_xstride              - 1);
    float U_C_i_jpkm = tex1Dfetch(tex_U_C,tindex_C              + uc_ystride - 1);

    float U_C_ipjmkm = tex1Dfetch(tex_U_C,tindex_C + uc_xstride - uc_ystride - 1);
    float U_C_imjpkm = tex1Dfetch(tex_U_C,tindex_C - uc_xstride + uc_ystride - 1);
    float U_C_imjmkp = tex1Dfetch(tex_U_C,tindex_C - uc_xstride - uc_ystride + 1);
 
    float U_C_ipjpkm = tex1Dfetch(tex_U_C,tindex_C + uc_xstride + uc_ystride - 1);
    float U_C_ipjmkp = tex1Dfetch(tex_U_C,tindex_C + uc_xstride - uc_ystride + 1);


    uf_buffer[index_F] = tex1Dfetch(tex_U_F, tindex_F) +
          0.421875f*U_C_i_j_k_ + 
          0.140625f*(U_C_ipj_k_ + U_C_i_jpk_  + U_C_i_j_kp) +       
          0.046875f*(U_C_ipjpk_ + U_C_ipj_kp  + U_C_i_jpkp) +
          0.015625f*U_C_ipjpkp;
                  
    uf_buffer[index_F - uf_xstride] = tex1Dfetch(tex_U_F, tindex_F - uf_xstride) +  
          0.421875f*U_C_i_j_k_ + 
          0.140625f*(U_C_imj_k_ + U_C_i_jpk_ + U_C_i_j_kp) + 
          0.046875f*(U_C_imjpk_ + U_C_imj_kp + U_C_i_jpkp) +
          0.015625f*U_C_ipjpkp;

    uf_buffer[index_F - uf_ystride] = tex1Dfetch(tex_U_F, tindex_F - uf_ystride) +  
          0.421875f*U_C_i_j_k_ + 
          0.140625f*(U_C_ipj_k_ + U_C_i_jmk_ + U_C_i_j_kp) + 
          0.046875f*(U_C_ipjmk_ + U_C_ipj_kp + U_C_i_jmkp) +
          0.015625f*U_C_ipjmkp;

    uf_buffer[index_F - 1] = tex1Dfetch(tex_U_F, tindex_F - 1) +  
          0.421875f*U_C_i_j_k_ + 
          0.140625f*(U_C_ipj_k_ + U_C_i_jpk_ + U_C_i_j_km) + 
          0.046875f*(U_C_ipjpk_ + U_C_ipj_km + U_C_i_jpkm) +
          0.015625f*U_C_ipjpkm;

    uf_buffer[index_F - uf_xstride - uf_ystride] = tex1Dfetch(tex_U_F, tindex_F - uf_xstride - uf_ystride) +  
          0.421875f*U_C_i_j_k_ + 
          0.140625f*(U_C_imj_k_ + U_C_i_jmk_ + U_C_i_j_kp) + 
          0.046875f*(U_C_imjmk_ + U_C_imj_kp + U_C_i_jmkp) +
          0.015625f*U_C_imjmkp;

    uf_buffer[index_F - uf_xstride - 1] = tex1Dfetch(tex_U_F, tindex_F - uf_xstride - 1) + 
          0.421875f*U_C_i_j_k_ + 
          0.140625f*(U_C_imj_k_ + U_C_i_jpk_ + U_C_i_j_km) + 
          0.046875f*(U_C_imjpk_ + U_C_imj_km + U_C_i_jpkm) +
          0.015625f*U_C_imjpkm;

    uf_buffer[index_F - uf_ystride - 1] = tex1Dfetch(tex_U_F, tindex_F - uf_ystride - 1) + 
          0.421875f*U_C_i_j_k_ + 
          0.140625f*(U_C_ipj_k_ + U_C_i_jmk_ + U_C_i_j_km) + 
          0.046875f*(U_C_ipjmk_ + U_C_ipj_km + U_C_i_jmkm) +
          0.015625f*U_C_ipjmkm;

    uf_buffer[index_F - uf_xstride - uf_ystride - 1] = tex1Dfetch(tex_U_F, tindex_F - uf_xstride - uf_ystride - 1) +
          0.421875f*U_C_i_j_k_ + 
          0.140625f*(U_C_imj_k_ + U_C_i_jmk_ + U_C_i_j_km) + 
          0.046875f*(U_C_imjmk_ + U_C_imj_km + U_C_i_jmkm) +
          0.015625f*U_C_imjmkm;
    }
}


__global__ void Sol_MultigridPressure3DDeviceF_relax(
  float *u_buffer, int xstride, int ystride, int nx, int ny, int nz, int tex_offset, int red_black, float omega, float hsq, float fx, float fy, float fz, float laplace_diag,
  float xpos_mod, float xneg_mod, float ypos_mod, float yneg_mod, float zpos_mod, float zneg_mod, 
  int blocksInY, float invBlocksInY)
{
  unsigned int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  // Since we are reading from tex mem, we don't have to worry about alignment & coalescing.  hence,
  // this routine looks slightly different from the non-texmem version because it doesn't
  // bother with the index shifting for 16-word alignment.  Skipping these steps
  // is slightly faster.

  k*=2;
  if ((i+j+red_black) % 2)
      k++;
  
  int idx = __mul24(i,xstride) + __mul24(j,ystride) + k;
  int tidx = idx + tex_offset;

  if (i < nx && j < ny && k < nz) {
      
    float bval = tex1Dfetch(tex_B, tidx);
    float ucent = tex1Dfetch(tex_U, tidx);

    float residual = (-hsq * bval - laplace_diag * ucent); 

    residual += fz * (tex1Dfetch(tex_U, tidx - 1      ) + tex1Dfetch(tex_U, tidx + 1      ));
    residual += fy * (tex1Dfetch(tex_U, tidx - ystride) + tex1Dfetch(tex_U, tidx + ystride));        
    residual += fx * (tex1Dfetch(tex_U, tidx - xstride) + tex1Dfetch(tex_U, tidx + xstride));

    float diag = laplace_diag;
    if (i==nx-1) diag += xpos_mod;
    if (i==0   ) diag += xneg_mod;
    if (j==ny-1) diag += ypos_mod;
    if (j==0   ) diag += yneg_mod;
    if (k==nz-1) diag += zpos_mod;
    if (k==0   ) diag += zneg_mod;

    // this is all very clever because it handles all boundary conditions with just slight modifications to diag.
    // I need to write up how this works, since it is both clever and obscure.
    float new_val = ucent + omega * residual/diag;

    // This is Gauss-Seidel relaxation
    // Since tex cache isn't coherent with global memory, if we were to read this value immediately, we would get the wrong
    // result.  However, since this is red-black G.S.,we won't be reading this value during this kernel launch,
    // and hence we're ok.  All caches are flushed upon kernel completion.  
    u_buffer[idx] = new_val;
  }
}


namespace ocu {


  template<>
bool 
Sol_MultigridPressure3DDevice<float>::bind_tex_relax(Grid3DDevice<float> &u_grid, Grid3DDevice<float> &b_grid)
{
  //How can these be made to work with templates?
  cudaChannelFormatDesc channelDesc_U = cudaCreateChannelDesc<float>();
  cudaChannelFormatDesc channelDesc_B = cudaCreateChannelDesc<float>();

  // set up texture
  tex_U.filterMode = cudaFilterModePoint;
  tex_U.normalized = false;
  tex_U.channelDesc = channelDesc_U;	

  tex_B.filterMode = cudaFilterModePoint;
  tex_B.normalized = false;
  tex_B.channelDesc = channelDesc_B;	

  if (cudaBindTexture(NULL, &tex_U, u_grid.buffer(), &channelDesc_U, u_grid.num_allocated_elements() * sizeof(float)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceD::bind_tex_relax - Could not bind texture U\n");
    return false;
  }

  if (cudaBindTexture(NULL, &tex_B, b_grid.buffer(), &channelDesc_B, b_grid.num_allocated_elements() * sizeof(float)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceD::bind_tex_relax - Could not bind texture B\n");
    return false;
  }

  return true;
}

template<>
bool 
Sol_MultigridPressure3DDevice<float>::unbind_tex_relax()
{
  cudaUnbindTexture(&tex_U);
  cudaUnbindTexture(&tex_B);
  return true;
}


template<>
bool 
Sol_MultigridPressure3DDevice<float>::bind_tex_calculate_residual(Grid3DDevice<float> &u_grid, Grid3DDevice<float> &b_grid)
{
  //How can these be made to work with templates?
  cudaChannelFormatDesc channelDesc_U = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  // set up texture
  tex_U.filterMode = cudaFilterModePoint;
  tex_U.normalized = false;
  tex_U.channelDesc = channelDesc_U;	

  if (cudaBindTexture(NULL, &tex_U, u_grid.buffer(), &channelDesc_U, u_grid.num_allocated_elements() * sizeof(float)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceF::bind_tex_calculate_residual - Could not bind texture U\n");
    return false;
  }

  return true;
}

template<>
bool 
Sol_MultigridPressure3DDevice<float>::unbind_tex_calculate_residual()
{
  cudaUnbindTexture(&tex_U);
  return true;
}

template<>
bool 
Sol_MultigridPressure3DDevice<float>::bind_tex_prolong(Grid3DDevice<float> &u_coarse_grid, Grid3DDevice<float> &u_fine_grid)
{
  //How can these be made to work with templates?
  cudaChannelFormatDesc channelDesc_U_C = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaChannelFormatDesc channelDesc_U_F = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  // set up texture
  tex_U_C.filterMode = cudaFilterModePoint;
  tex_U_C.normalized = false;
  tex_U_C.channelDesc = channelDesc_U_C;	

  tex_U_F.filterMode = cudaFilterModePoint;
  tex_U_F.normalized = false;
  tex_U_F.channelDesc = channelDesc_U_F;	


  if (cudaBindTexture(NULL, &tex_U_C, u_coarse_grid.buffer(), &channelDesc_U_C, u_coarse_grid.num_allocated_elements() * sizeof(float)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceF::bind_tex_prolong - Could not bind texture U_C\n");
    return false;
  }
  
  if (cudaBindTexture(NULL, &tex_U_F, u_fine_grid.buffer(), &channelDesc_U_F, u_fine_grid.num_allocated_elements() * sizeof(float)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceF::bind_tex_prolong - Could not bind texture U_F\n");
    return false;
  }

  return true;
}

template<>
bool 
Sol_MultigridPressure3DDevice<float>::unbind_tex_prolong()
{
  cudaUnbindTexture(&tex_U_C);
  cudaUnbindTexture(&tex_U_F);
  return true;
}


template<>
bool 
Sol_MultigridPressure3DDevice<float>::invoke_kernel_relax(Grid3DDevice<float> &u_grid, Grid3DDevice<float> &b_grid, Grid2DDevice<float> &diag_grid, int red_black, double h)
{
  int tnx = (u_grid.nz())/2;
  int tny = u_grid.ny();
  int tnz = u_grid.nx();

  //int threadsInX = 32;
  //int threadsInY = 4;
  //int threadsInZ = 1;
  int threadsInX = 16;
  int threadsInY = 4;
  int threadsInZ = 2;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  PreKernel();
  Sol_MultigridPressure3DDeviceF_relax<<<Dg,Db>>>(&u_grid.at(0,0,0), u_grid.xstride(), u_grid.ystride(), u_grid.nx(), u_grid.ny(), u_grid.nz(), u_grid.shift_amount(), red_black, 
    (float)_omega, (float)(h*h), (float)_fx, (float)_fy, (float)_fz,  (float)(2*_fx + 2*_fy +2*_fz),
    (float)bc_diag_mod(bc.xpos, _fx), (float)bc_diag_mod(bc.xneg, _fx), (float)bc_diag_mod(bc.ypos, _fy), (float)bc_diag_mod(bc.yneg, _fy), (float)bc_diag_mod(bc.zpos, _fz), (float)bc_diag_mod(bc.zneg, _fz), 
    blocksInY, 1.0f/(float)blocksInY);
  return PostKernel("Sol_MultigridPressure3DDeviceF_relax", tnz);
}


template<>
bool 
Sol_MultigridPressure3DDevice<float>::invoke_kernel_calculate_residual(Grid3DDevice<float> &u_grid, Grid3DDevice<float> &b_grid, Grid3DDevice<float> &r_grid, double h)
{

  // launch nz+1 threads since the kernel shifts by -1 for better coalescing.  Also, we must transpose x,y,z into
  // thread ids for beter coalescing behaviors, so that adjacent threads operate on adjacent memory locations.
  int tnx = u_grid.nz();
  int tny = u_grid.ny();
  int tnz = u_grid.nx();

  int threadsInX = 16;
  int threadsInY = 2;
  int threadsInZ = 2;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  double fx_div_hsq = _fx / (h*h);
  double fy_div_hsq = _fy / (h*h);
  double fz_div_hsq = _fz / (h*h);
  double diag = 2 * (_fx + _fy + _fz) / (h*h);

  PreKernel();
  Sol_MultigridPressure3DDeviceF_calculate_residual<<<Dg, Db>>>(&b_grid.at(0,0,0), &r_grid.at(0,0,0), r_grid.xstride(), r_grid.ystride(), r_grid.nx(), r_grid.ny(), r_grid.nz(), r_grid.shift_amount(),
    (float)fx_div_hsq, (float)fy_div_hsq, (float)fz_div_hsq, (float)diag, blocksInY, 1.0f/(float)blocksInY);
  return PostKernel("Sol_MultigridPressure3DDeviceF_calculate_residual", tnz);
}

template<>
bool 
Sol_MultigridPressure3DDevice<float>::invoke_kernel_restrict(Grid3DDevice<float> &r_grid, Grid3DDevice<float> &b_coarse_grid)
{
  // launch nz+1 threads since the kernel shifts by -1 for better coalescing.  Also, we must transpose x,y,z into
  // thread ids for beter coalescing behaviors, so that adjacent threads operate on adjacent memory locations.
  int tnx = b_coarse_grid.nz();
  int tny = b_coarse_grid.ny();
  int tnz = b_coarse_grid.nx();

  int threadsInX = 32;
  int threadsInY = 4;
  int threadsInZ = 1;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  PreKernel();
  Sol_MultigridPressure3DDeviceF_restrict<<<Dg, Db>>>(&r_grid.at(0,0,0), r_grid.xstride(), r_grid.ystride(), 
    &b_coarse_grid.at(0,0,0), b_coarse_grid.xstride(), b_coarse_grid.ystride(), b_coarse_grid.nx(), b_coarse_grid.ny(), b_coarse_grid.nz(), 
    blocksInY, 1.0f/(float)blocksInY);
  return PostKernel("Sol_MultigridPressure3DDeviceF_restrict", tnz);
}

template<>
bool 
Sol_MultigridPressure3DDevice<float>::invoke_kernel_prolong(Grid3DDevice<float> &u_coarse_grid, Grid3DDevice<float> &u_fine_grid)
{
  // launch nz+1 threads since the kernel shifts by -1 for better coalescing.  Also, we must transpose x,y,z into
  // thread ids for beter coalescing behaviors, so that adjacent threads operate on adjacent memory locations.
  int tnx = u_coarse_grid.nz();
  int tny = u_coarse_grid.ny();
  int tnz = u_coarse_grid.nx();

  int threadsInX = 32;
  int threadsInY = 4;
  int threadsInZ = 1;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  PreKernel();
  Sol_MultigridPressure3DDeviceF_prolong<<<Dg, Db>>>(&u_fine_grid.at(0,0,0), u_fine_grid.xstride(), u_fine_grid.ystride(), u_fine_grid.shift_amount(),
    u_coarse_grid.xstride(), u_coarse_grid.ystride(), u_coarse_grid.nx(), u_coarse_grid.ny(), u_coarse_grid.nz(),  u_coarse_grid.shift_amount(),
    blocksInY, 1.0f/(float)blocksInY);
  return PostKernel("Sol_MultigridPressure3DDeviceF_prolong", tnz);
}


} // end namespace

