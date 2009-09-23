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
// double routines
//
///////////////////////////////////////////////////////////////////////////////////////////////////


#ifdef OCU_DOUBLESUPPORT

texture<int2, 1, cudaReadModeElementType> dtex_U_C;
texture<int2, 1, cudaReadModeElementType> dtex_U_F;
texture<int2, 1, cudaReadModeElementType> dtex_U;
texture<int2, 1, cudaReadModeElementType> dtex_B;

__inline__ __device__ double tex1Dfetchd_U(const int& i)
{
  int2 v = tex1Dfetch(dtex_U, i);
  return __hiloint2double(v.y, v.x);
}

__inline__ __device__ double tex1Dfetchd_U_C(const int& i)
{
  int2 v = tex1Dfetch(dtex_U_C, i);
  return __hiloint2double(v.y, v.x);
}

__inline__ __device__ double tex1Dfetchd_U_F(const int& i)
{
  int2 v = tex1Dfetch(dtex_U_F, i);
  return __hiloint2double(v.y, v.x);
}

__inline__ __device__ double tex1Dfetchd_B(const int& i)
{
  int2 v = tex1Dfetch(dtex_B, i);
  return __hiloint2double(v.y, v.x);
}


__global__ void Sol_MultigridPressure3DDeviceD_restrict(
  double *r_buffer, int r_xstride, int r_ystride, double *b_buffer, int b_xstride, int b_ystride, int bnx, int bny, int bnz, unsigned int blocksInY, float invBlocksInY)
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


__global__ void Sol_MultigridPressure3DDeviceD_calculate_residual(
  double *b_buffer, double *r_buffer, int xstride, int ystride, int nx, int ny, int nz, int tex_offset, 
  double fx_div_hsq, double fy_div_hsq, double fz_div_hsq, double diag, unsigned int blocksInY, float invBlocksInY)
{
  unsigned int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  int idx = __mul24(i,xstride) + __mul24(j,ystride) + (k);
  int tidx = idx + tex_offset;

  if (i < nx && j < ny && k < nz) {
    r_buffer[idx] = b_buffer[idx] -
      (fz_div_hsq*(tex1Dfetchd_U(tidx - 1      ) + tex1Dfetchd_U(tidx + 1      )) + 
       fy_div_hsq*(tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride)) +
       fx_div_hsq*(tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride)) 
        -diag*tex1Dfetchd_U(tidx));
  }
}



__global__ void Sol_MultigridPressure3DDeviceD_prolong(
  double *uf_buffer, int uf_xstride, int uf_ystride, int uf_tex_offset, 
  int uc_xstride, int uc_ystride, int ucnx, int ucny, int ucnz, int uc_tex_offset, 
  unsigned int blocksInY, float invBlocksInY)
{
  unsigned int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
  int k_C     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j_C     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i_C     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  // coalescing shift

  if (i_C < ucnx && j_C < ucny && k_C < ucnz) {
    int i = 2*i_C+1;
    int j = 2*j_C+1;
    int k = 2*k_C+1;

    // indices used for accessing the texture must be offset, and are named with the "t" prefix
    int tindex_C = __mul24(i_C,uc_xstride) + __mul24(j_C,uc_ystride) + k_C + uc_tex_offset;
    int index_F  = __mul24(i  ,uf_xstride) + __mul24(j  ,uf_ystride) + k;
    int tindex_F = index_F + uf_tex_offset;

    double U_C_i_j_k_ = tex1Dfetchd_U_C(tindex_C                              );
    double U_C_ipj_k_ = tex1Dfetchd_U_C(tindex_C + uc_xstride                 );
    double U_C_i_jpk_ = tex1Dfetchd_U_C(tindex_C              + uc_ystride    );
    double U_C_ipjpk_ = tex1Dfetchd_U_C(tindex_C + uc_xstride + uc_ystride    );
    double U_C_i_j_kp = tex1Dfetchd_U_C(tindex_C                           + 1);
    double U_C_ipj_kp = tex1Dfetchd_U_C(tindex_C + uc_xstride              + 1);
    double U_C_i_jpkp = tex1Dfetchd_U_C(tindex_C              + uc_ystride + 1);
    double U_C_ipjpkp = tex1Dfetchd_U_C(tindex_C + uc_xstride + uc_ystride + 1);

    double U_C_imj_k_ = tex1Dfetchd_U_C(tindex_C - uc_xstride                 );
    double U_C_i_jmk_ = tex1Dfetchd_U_C(tindex_C              - uc_ystride    );
    double U_C_imjmk_ = tex1Dfetchd_U_C(tindex_C - uc_xstride - uc_ystride    );
    double U_C_i_j_km = tex1Dfetchd_U_C(tindex_C                           - 1);
    double U_C_imj_km = tex1Dfetchd_U_C(tindex_C - uc_xstride              - 1);
    double U_C_i_jmkm = tex1Dfetchd_U_C(tindex_C              - uc_ystride - 1);   
    double U_C_imjmkm = tex1Dfetchd_U_C(tindex_C - uc_xstride - uc_ystride - 1);

    double U_C_imjpk_ = tex1Dfetchd_U_C(tindex_C - uc_xstride + uc_ystride    );
    double U_C_imj_kp = tex1Dfetchd_U_C(tindex_C - uc_xstride              + 1);

    double U_C_ipjmk_ = tex1Dfetchd_U_C(tindex_C + uc_xstride - uc_ystride    );
    double U_C_i_jmkp = tex1Dfetchd_U_C(tindex_C              - uc_ystride + 1);

    double U_C_ipj_km = tex1Dfetchd_U_C(tindex_C + uc_xstride              - 1);
    double U_C_i_jpkm = tex1Dfetchd_U_C(tindex_C              + uc_ystride - 1);

    double U_C_ipjmkm = tex1Dfetchd_U_C(tindex_C + uc_xstride - uc_ystride - 1);
    double U_C_imjpkm = tex1Dfetchd_U_C(tindex_C - uc_xstride + uc_ystride - 1);
    double U_C_imjmkp = tex1Dfetchd_U_C(tindex_C - uc_xstride - uc_ystride + 1);
 
    double U_C_ipjpkm = tex1Dfetchd_U_C(tindex_C + uc_xstride + uc_ystride - 1);
    double U_C_ipjmkp = tex1Dfetchd_U_C(tindex_C + uc_xstride - uc_ystride + 1);


    uf_buffer[index_F] = tex1Dfetchd_U_F(tindex_F) +
          0.421875*U_C_i_j_k_ + 
          0.140625*(U_C_ipj_k_ + U_C_i_jpk_  + U_C_i_j_kp) +       
          0.046875*(U_C_ipjpk_ + U_C_ipj_kp  + U_C_i_jpkp) +
          0.015625*U_C_ipjpkp;
                  
    uf_buffer[index_F - uf_xstride] = tex1Dfetchd_U_F(tindex_F - uf_xstride) +  
          0.421875*U_C_i_j_k_ + 
          0.140625*(U_C_imj_k_ + U_C_i_jpk_ + U_C_i_j_kp) + 
          0.046875*(U_C_imjpk_ + U_C_imj_kp + U_C_i_jpkp) +
          0.015625*U_C_ipjpkp;

    uf_buffer[index_F - uf_ystride] = tex1Dfetchd_U_F(tindex_F - uf_ystride) +  
          0.421875*U_C_i_j_k_ + 
          0.140625*(U_C_ipj_k_ + U_C_i_jmk_ + U_C_i_j_kp) + 
          0.046875*(U_C_ipjmk_ + U_C_ipj_kp + U_C_i_jmkp) +
          0.015625*U_C_ipjmkp;

    uf_buffer[index_F - 1] = tex1Dfetchd_U_F(tindex_F - 1) +  
          0.421875*U_C_i_j_k_ + 
          0.140625*(U_C_ipj_k_ + U_C_i_jpk_ + U_C_i_j_km) + 
          0.046875*(U_C_ipjpk_ + U_C_ipj_km + U_C_i_jpkm) +
          0.015625*U_C_ipjpkm;

    uf_buffer[index_F - uf_xstride - uf_ystride] = tex1Dfetchd_U_F(tindex_F - uf_xstride - uf_ystride) +  
          0.421875*U_C_i_j_k_ + 
          0.140625*(U_C_imj_k_ + U_C_i_jmk_ + U_C_i_j_kp) + 
          0.046875*(U_C_imjmk_ + U_C_imj_kp + U_C_i_jmkp) +
          0.015625*U_C_imjmkp;

    uf_buffer[index_F - uf_xstride - 1] = tex1Dfetchd_U_F(tindex_F - uf_xstride - 1) + 
          0.421875*U_C_i_j_k_ + 
          0.140625*(U_C_imj_k_ + U_C_i_jpk_ + U_C_i_j_km) + 
          0.046875*(U_C_imjpk_ + U_C_imj_km + U_C_i_jpkm) +
          0.015625*U_C_imjpkm;

    uf_buffer[index_F - uf_ystride - 1] = tex1Dfetchd_U_F(tindex_F - uf_ystride - 1) + 
          0.421875*U_C_i_j_k_ + 
          0.140625*(U_C_ipj_k_ + U_C_i_jmk_ + U_C_i_j_km) + 
          0.046875*(U_C_ipjmk_ + U_C_ipj_km + U_C_i_jmkm) +
          0.015625*U_C_ipjmkm;

    uf_buffer[index_F - uf_xstride - uf_ystride - 1] = tex1Dfetchd_U_F(tindex_F - uf_xstride - uf_ystride - 1) +
          0.421875*U_C_i_j_k_ + 
          0.140625*(U_C_imj_k_ + U_C_i_jmk_ + U_C_i_j_km) + 
          0.046875*(U_C_imjmk_ + U_C_imj_km + U_C_i_jmkm) +
          0.015625*U_C_imjmkm;
    }
}

__global__ void Sol_MultigridPressure3DDeviceD_relax_CACHE2(
  double *u_buffer, int xstride, int ystride, int nx, int ny, int nz, int tex_offset, int red_black, double omega, double hsq, double fx, double fy, double fz, double laplace_diag,
  double xpos_mod, double xneg_mod, double ypos_mod, double yneg_mod, double zpos_mod, double zneg_mod, 
  int blocksInY, float invBlocksInY,unsigned int local_idx_jstride, unsigned int local_idx_kstride)
{

  unsigned int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  // shift by -1 here to account for gz offset
  int k_start = __mul24(blockIdx.x,blockDim.x) * 2 - 1;
  
  int idx = __mul24(i,xstride) + __mul24(j,ystride) + k_start;
  int tidx = idx + tex_offset;

  // get shmem ptr
  extern __shared__ char smempool[];
  double *local_U = (double *)smempool;

  if (i < nx && j < ny) {

    // offset to a unique region per i,j coordinate
    local_U += __mul24(threadIdx.y, local_idx_jstride) + __mul24(threadIdx.z, local_idx_kstride); 

    // read entire row into local_U[k] - For n threads in z, we will read 2n+2 elements, or to the end of the row, whichever comes first.
    //if (k_start + threadIdx.x <= nz)
      local_U[threadIdx.x] = tex1Dfetchd_U(tidx+threadIdx.x);
    //if (k_start + threadIdx.x + blockDim.x<= nz)
      local_U[threadIdx.x+blockDim.x] = tex1Dfetchd_U(tidx+threadIdx.x+blockDim.x);
    //if (k_start + threadIdx.x + 2*blockDim.x<= nz && threadIdx.x < 2)
      local_U[threadIdx.x+2*blockDim.x] = tex1Dfetchd_U(tidx+threadIdx.x+2*blockDim.x);
  }

  __syncthreads();
  
  local_U++; // to account for ghost node ? can we do the per-color shift here to hide it later?

  int k = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  k*=2;
  if ((i+j+red_black) % 2)
      k++;

  idx = __mul24(i,xstride) + __mul24(j,ystride) + k;
  tidx = idx + tex_offset;

  int local_k = k - k_start;

  if (k < nz) {
    double bval = tex1Dfetchd_B(tidx);
    double ucent = local_U[local_k];

    double residual = (-hsq * bval - laplace_diag * ucent); 

    residual += fz * (local_U[local_k-1]            + local_U[local_k+1]         );
    residual += fy * (tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride));        
    residual += fx * (tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride));

    double diag = laplace_diag;
    if (i==nx-1) diag += xpos_mod;
    if (i==0   ) diag += xneg_mod;
    if (j==ny-1) diag += ypos_mod;
    if (j==0   ) diag += yneg_mod;
    if (k==nz-1) diag += zpos_mod;
    if (k==0   ) diag += zneg_mod;

    // this is all very clever because it handles all boundary conditions with just slight modifications to diag.
    // I need to write up how this works, since it is both clever and obscure.
    double new_val = ucent + omega * residual/diag;

    // This is Gauss-Seidel relaxation
    // Since tex cache isn't coherent with global memory, if we were to read this value immediately, we would get the wrong
    // result.  However, since this is red-black G.S.,we won't be reading this value during this kernel launch,
    // and hence we're ok.  All caches are flushed upon kernel completion.  
    u_buffer[idx] = new_val;
  }
}


__global__ void Sol_MultigridPressure3DDeviceD_relax_CACHE(
  double *u_buffer, int xstride, int ystride, int nx, int ny, int nz, int tex_offset, int red_black, double omega, double hsq, double fx, double fy, double fz, double laplace_diag,
  double xpos_mod, double xneg_mod, double ypos_mod, double yneg_mod, double zpos_mod, double zneg_mod, 
  int blocksInY, float invBlocksInY,unsigned int local_idx_jstride, unsigned int local_idx_kstride)
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

  // cache U[i,j,k+1] in local_U[local_idx+1]
  // for each thread, U[i,j,k-1] will be in local_U[local_idx] (since local_idx's of adjacent k values will also be adjacent)
  extern __shared__ char smempool[];
  double *local_U = (double *)smempool;

  int local_idx = threadIdx.x + __mul24(threadIdx.y, local_idx_jstride) + __mul24(threadIdx.z, local_idx_kstride);

  if (threadIdx.x == 0) {
    local_U[local_idx] = tex1Dfetchd_U(tidx - 1);
  }
  local_U[local_idx+1] = tex1Dfetchd_U(tidx + 1);
  __syncthreads();

  if (i < nx && j < ny && k < nz) {
    double bval = tex1Dfetchd_B(tidx);
    double ucent = tex1Dfetchd_U(tidx);

    double residual = (-hsq * bval - laplace_diag * ucent); 

    residual += fz * (local_U[local_idx]            + local_U[local_idx+1]         );
    residual += fy * (tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride));        
    residual += fx * (tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride));

    double diag = laplace_diag;
    if (i==nx-1) diag += xpos_mod;
    if (i==0   ) diag += xneg_mod;
    if (j==ny-1) diag += ypos_mod;
    if (j==0   ) diag += yneg_mod;
    if (k==nz-1) diag += zpos_mod;
    if (k==0   ) diag += zneg_mod;

    // this is all very clever because it handles all boundary conditions with just slight modifications to diag.
    // I need to write up how this works, since it is both clever and obscure.
    double new_val = ucent + omega * residual/diag;

    // This is Gauss-Seidel relaxation
    // Since tex cache isn't coherent with global memory, if we were to read this value immediately, we would get the wrong
    // result.  However, since this is red-black G.S.,we won't be reading this value during this kernel launch,
    // and hence we're ok.  All caches are flushed upon kernel completion.  
    u_buffer[idx] = new_val;
  }
}

__global__ void Sol_MultigridPressure3DDeviceD_relax_VECTOR(
  double *u_buffer, int xstride, int ystride, int nx, int ny, int nz, int nxm, int nym, int nzm, int tex_offset, int red_black, 
  double omega, double neg_hsq, double fx, double fy, double fz, double laplace_diag,
  double xpos_mod, double xneg_mod, double ypos_mod, double yneg_mod, double zpos_mod, double zneg_mod)
{
  // mapping here is a bit weird - we want the cta's to be arranged in a 2d grid in dimensions x,y, but threads within each
  // cta are arranged in a 3d grid with the threadIdx.x -> k coordinate, so that adjacent threads will operate on
  // adjacent k-coordinates.
  int k     = threadIdx.x;
  int j     = __mul24(blockIdx.y ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdx.x ,blockDim.z) + threadIdx.z;

  // Since we are reading from tex mem, we don't have to worry about alignment & coalescing.  hence,
  // this routine looks slightly different from the non-texmem version because it doesn't
  // bother with the index shifting for 16-word alignment.  Skipping these steps
  // is slightly faster.

  k*=2;
  if ((i+j+red_black) % 2)
    k++;
  
  double diag = laplace_diag;
  if (i==nxm) diag += xpos_mod;
  if (!i    ) diag += xneg_mod;
  if (j==nym) diag += ypos_mod;
  if (!j    ) diag += yneg_mod;
  if (!k    ) diag += zneg_mod;

  if (i >= nx || j >= ny) return;

  // each thread will operate over several locations, separated by blockDim.x*2 positions in the array.
  for(; k < nz; k+=blockDim.x * 2) {

    int idx = __mul24(i,xstride) + __mul24(j,ystride) + k;    
    int tidx = idx + tex_offset;
    double bval = tex1Dfetchd_B(tidx);
    double ucent = tex1Dfetchd_U(tidx);

    double residual = (neg_hsq * bval - laplace_diag * ucent);
    residual += fz * (tex1Dfetchd_U(tidx - 1      ) + tex1Dfetchd_U(tidx + 1      ));
    residual += fy * (tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride));        
    residual += fx * (tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride));

    // this will only be true the last time through a loop, so we can overwrite the diag value since
    // we will not read it again.
    if (k==nzm) diag += zpos_mod;

    // this is all very clever because it handles all boundary conditions with just slight modifications to diag.
    double new_val = ucent + omega * residual/diag;

    // This is Gauss-Seidel relaxation
    // Since tex cache isn't coherent with global memory, if we were to read this value immediately, we would get the wrong
    // result.  However, since this is red-black G.S.,we won't be reading this value during this kernel launch,
    // and hence we're ok.  All caches are flushed upon kernel completion.  
    u_buffer[idx] = new_val;
  }
}

__global__ void Sol_MultigridPressure3DDeviceD_relax_VECTORDIAG(
  double *u_buffer, int xstride, int ystride, int nx, int ny, int nz, int nxm, int nym, int nzm, int tex_offset, int red_black, 
  double omega, double neg_hsq, double fx, double fy, double fz, 
  double laplace_diag,
  double *diags, int diag_istride,
  double zpos_mod, double zneg_mod)
{
  // mapping here is a bit weird - we want the cta's to be arranged in a 2d grid in dimensions x,y, but threads within each
  // cta are arranged in a 3d grid with the threadIdx.x -> k coordinate, so that adjacent threads will operate on
  // adjacent k-coordinates.
  int k     = threadIdx.x;
  int j     = __mul24(blockIdx.y ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdx.x ,blockDim.z) + threadIdx.z;

  // Since we are reading from tex mem, we don't have to worry about alignment & coalescing.  hence,
  // this routine looks slightly different from the non-texmem version because it doesn't
  // bother with the index shifting for 16-word alignment.  Skipping these steps
  // is slightly faster.

  k*=2;
  if ((i+j+red_black) % 2)
      k++;
  
  // reading from a texture is actually ~5% slower, maybe due to cache pollution?
  //double diag = tex1Dfetchd_diag(__mul24(i,diag_istride) + j);
  double diag = diags[__mul24(i,diag_istride) + j];
  if (!k) diag += zneg_mod;

  if (i < nx && j < ny) {

    // each thread will operate over several locations, separated by blockDim.x*2 positions in the array.
    while (k < nz) {

      int idx = __mul24(i,xstride) + __mul24(j,ystride) + k;
      int tidx = idx + tex_offset;

      double bval = tex1Dfetchd_B(tidx);
      double ucent = tex1Dfetchd_U(tidx);

      double residual = (neg_hsq * bval - laplace_diag * ucent);
      residual += fz * (tex1Dfetchd_U(tidx - 1      ) + tex1Dfetchd_U(tidx + 1      ));
      residual += fy * (tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride));        
      residual += fx * (tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride));
 
      // we could pull some of this out of the inner-most loop:
      if (k==nzm) diag += zpos_mod;

      // this is all very clever because it handles all boundary conditions with just slight modifications to diag.
      double new_val = ucent + omega * residual/diag;

      // This is Gauss-Seidel relaxation
      // Since tex cache isn't coherent with global memory, if we were to read this value immediately, we would get the wrong
      // result.  However, since this is red-black G.S.,we won't be reading this value during this kernel launch,
      // and hence we're ok.  All caches are flushed upon kernel completion.  
      u_buffer[idx] = new_val;

      k += (blockDim.x * 2);
    }
  }
}

__global__ void Sol_MultigridPressure3DDeviceD_relax_TEX(
  double *b_buffer, double *u_buffer, int xstride, int ystride, int nx, int ny, int nz, int nxm, int nym, int nzm, int tex_offset, int red_black, 
  double omega, double neg_hsq, double fx, double fy, double fz,
  double laplace_diag,
  double xpos_mod, double xneg_mod, double ypos_mod, double yneg_mod, double zpos_mod, double zneg_mod, 
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
       
    double bval = b_buffer[idx];
    double ucent = tex1Dfetchd_U(tidx);

    double residual = neg_hsq * bval - laplace_diag * ucent;
    residual += fz * (tex1Dfetchd_U(tidx - 1      ) + tex1Dfetchd_U(tidx + 1      ));
    residual += fy * (tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride));        
    residual += fx * (tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride));
        
    // reading from a cache is about 1% slower.  don't know why.    
    double diag = laplace_diag;
    if (i==nxm) diag += xpos_mod;
    if (!i    ) diag += xneg_mod;
    if (j==nym) diag += ypos_mod;
    if (!j    ) diag += yneg_mod;
    if (k==nzm) diag += zpos_mod;
    if (!k    ) diag += zneg_mod;

    // this is all very clever because it handles all boundary conditions with just slight modifications to diag.
    // I need to write up how this works, since it is both clever and obscure.
    double new_val = ucent + omega * residual/diag;
    u_buffer[idx] = new_val;       
  }
}

__global__ void Sol_MultigridPressure3DDeviceD_relax_DIAG(
  double *b_buffer, double *u_buffer, int xstride, int ystride, int nx, int ny, int nz, int nxm, int nym, int nzm, int tex_offset, int red_black, 
  double omega, double neg_hsq, double fx, double fy, double fz,
  double laplace_diag,
  double *diags, int diag_istride,
  double zpos_mod, double zneg_mod,
  int blocksInY, float invBlocksInY)
{
//  unsigned int blockIdxz = truncf(blockIdx.y * invBlocksInY);
//  unsigned int blockIdxy = blockIdx.y - __umul24(blockIdxz,blocksInY);
  unsigned int blockIdxz = blockIdx.y/blocksInY;
  unsigned int blockIdxy = blockIdx.y%blocksInY;
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
       
    double ucent = tex1Dfetchd_U(tidx);

    double residual;
    residual = -laplace_diag * ucent;
    residual += fz * (tex1Dfetchd_U(tidx - 1      ) + tex1Dfetchd_U(tidx + 1      ));
    residual += fy * (tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride));        
    residual += fx * (tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride));
    
    //double bval = tex1Dfetchd_B(tidx);
    double bval = b_buffer[idx];
    residual += neg_hsq * bval;
    
    // reading from a cache is about 1% slower.  don't know why.    
    double diag = diags[__mul24(i,diag_istride) + j];
    if (k==nzm) diag += zpos_mod;
    if (!k    ) diag += zneg_mod;

    // this is all very clever because it handles all boundary conditions with just slight modifications to diag.
    // I need to write up how this works, since it is both clever and obscure.
    double new_val = ucent + omega * residual/diag;

    // This is Gauss-Seidel relaxation
    // Since tex cache isn't coherent with global memory, if we were to read this value immediately, we would get the wrong
    // result.  However, since this is red-black G.S.,we won't be reading this value during this kernel launch,
    // and hence we're ok.  All caches are flushed upon kernel completion.  
    u_buffer[idx] = new_val;
  }
}


__global__ void Sol_MultigridPressure3DDeviceD_relax_ALTDIAG(
  double *b_buffer, double *u_buffer, int xstride, int ystride, int nx, int ny, int nz, int nxm, int nym, int nzm, int tex_offset, int red_black, 
  double omega, double neg_hsq, double fx, double fy, double fz,
  double laplace_diag,
  double *diags, int diag_istride,
  double zpos_mod, double zneg_mod,
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

  j*=2;
  if ((i+k+red_black) % 2)
      j++;
  
  int idx = __mul24(i,xstride) + __mul24(j,ystride) + k;
  int tidx = idx + tex_offset;

  if (i < nx && j < ny && k < nz) {
      
    double ucent = tex1Dfetchd_U(tidx);

    double residual;
    residual = - laplace_diag * ucent;
    residual += fz * (tex1Dfetchd_U(tidx - 1      ) + tex1Dfetchd_U(tidx + 1      ));
    residual += fy * (tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride));        
    residual += fx * (tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride));
    
    double bval = b_buffer[idx];
    residual += neg_hsq * bval;

    // reading from a cache is about 1% slower.  don't know why.
    //double diag = tex1Dfetchd_diag(__mul24(i,diag_istride) + j);
    double diag = diags[__mul24(i,diag_istride) + j];
    if (k==nzm) diag += zpos_mod;
    if (!k    ) diag += zneg_mod;

    // this is all very clever because it handles all boundary conditions with just slight modifications to diag.
    // I need to write up how this works, since it is both clever and obscure.
    double new_val = ucent + omega * residual/diag;

    // This is Gauss-Seidel relaxation
    // Since tex cache isn't coherent with global memory, if we were to read this value immediately, we would get the wrong
    // result.  However, since this is red-black G.S.,we won't be reading this value during this kernel launch,
    // and hence we're ok.  All caches are flushed upon kernel completion.  
    u_buffer[idx] = new_val;
  }
}



__global__ void Sol_MultigridPressure3DDeviceD_relax_LINERB(
  double *b_buffer, double *u_buffer, int xstride, int ystride, int nx, int ny, int nz, int nxm, int nym, int nzm, int tex_offset, int red_black, 
  double omega, double neg_hsq, double fx, double fy, double fz,
  double laplace_diag,
  double *diags, int diag_istride,
  double zpos_mod, double zneg_mod)
{
  int i = blockIdx.x * blockDim.z + threadIdx.z;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  // do red-black staggering in i
  i*=2;
  if ((j+red_black)%2)
    i++;
  
  double diag = diags[__mul24(i,diag_istride) + j];

  // load entire line into shmem
  int local_stride = blockDim.x;
  
  extern __shared__ char smempool[];
  double *shmem = (double *)smempool;
  shmem += (threadIdx.z * blockDim.y + threadIdx.y) * nz *2;
  double *local_U = shmem;
  double *local_resid = local_U + nz;

  bool in_bounds_ij = (i < nx && j < ny);

  int k = threadIdx.x-1; //coalescing shift
  int idx = __mul24(i,xstride) + __mul24(j,ystride) + k;
  int tidx = idx + tex_offset;

  // only continue if we are in-bounds in x and y - all threads in a warp
  // will go the same way through this conditional
  if (in_bounds_ij) {    
    if (k >= 0 && k < nz) {
      local_U[k] = u_buffer[idx];
      local_resid[k] = neg_hsq * b_buffer[idx];
    }

    k+=local_stride;
    tidx += local_stride;
    idx += local_stride;

    if (k < nz) {
      local_U[k] = u_buffer[idx];
      local_resid[k] = neg_hsq * b_buffer[idx];
    }
  }

  // GS update of even elements
  // calc idx
  k = threadIdx.x*2;
  idx = __mul24(i,xstride) + __mul24(j,ystride) + k;
  tidx = idx + tex_offset;

  __syncthreads();

  if (in_bounds_ij && k < nz) {

    double residual = fy * (tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride));        
    residual += fx * (tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride));

    double local_diag = diag;
    if (k == 0) {
      local_diag += zneg_mod;
      residual += fz * tex1Dfetchd_U(tidx-1);
    }
    else {
      residual += fz * local_U[k-1];
    }

    // since k must be even, we know that k != nzm, assuming nz is always even (it is, right?)
    residual += fz * local_U[k+1];    
    residual += local_resid[k];
    residual += -laplace_diag * local_U[k];
    local_U[k] += omega * residual/local_diag;
  }


  // GS update of odd elements
  k++;
  idx++;
  tidx++;

  __syncthreads();

  if (in_bounds_ij && k < nz) {
    
    double residual = fy * (tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride));        
    residual += fx * (tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride));

    double local_diag = diag;
    if (k == nzm) {
      local_diag += zpos_mod;
      residual += fz * tex1Dfetchd_U(tidx+1);
    }
    else {
      residual += fz * local_U[k+1];
    }
    // since k must be odd, we know that k != 0
    residual += fz * local_U[k-1];
    residual += local_resid[k];
    residual += -laplace_diag * local_U[k];

    local_U[k] += omega * residual/local_diag;
  }
  
  
  // recalc idx.  write back
  k = threadIdx.x-1;
  idx = __mul24(i,xstride) + __mul24(j,ystride) + k;

  __syncthreads();

  if (in_bounds_ij) {
    if (k >=0 && k < nz)
      u_buffer[idx] = local_U[k];
    k += local_stride;
    idx += local_stride;
    if (k < nz)
      u_buffer[idx] = local_U[k];
  }
}


// this is a very slow version of this!
__device__ unsigned int bit_reverse(unsigned int v, int n) 
{
  int num_bits = 0;
  while (n>1) {
    num_bits++;
    n /= 2;
  }

  // reverse num_bits left-most bits
  unsigned int r = (v >> num_bits); // r will be reversed bits of v; first get non-reversed part of v
  while (num_bits) {
    r <<= 1;
    r |= (v & 1);
    v >>= 1;
    num_bits--;
  }
  
  return r; 
}

__global__ void Sol_MultigridPressure3DDeviceD_relax_LINERELAX(
  double *b_buffer, double *u_buffer, int xstride, int ystride, int nx, int ny, int nz, int nxm, int nym, int nzm, int tex_offset, int red_black, 
  double omega, double hsq, double fx, double fy, double fz,
  double laplace_diag,
  double zpos_mod, double zneg_mod)
{
  int i = blockIdx.x * blockDim.z + threadIdx.z;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  // do red-black staggering in i
  i*=2;
  if ((j+red_black)%2)
    i++;

  // nz should equal blockDim.x

  // offset to a per-warp group_width size segment of the smempool based on the threadIdx.y and threadIdx.z
  // Could get by with only 4 arrays.  local_U is only needed at the end, at which piont
  // we could reuse either local_Mip or local_Mim.
  int local_stride = blockDim.x;
  extern __shared__ char smempool[];
  double *shmem = (double *)smempool;
  shmem += (threadIdx.z * blockDim.y + threadIdx.y) * local_stride * 4;

  double *local_RHS = shmem;
  double *local_Mii = local_RHS + local_stride;
  double *local_Mip = local_Mii + local_stride;
  double *local_Mim = local_Mip + local_stride;
  double *local_U = local_Mim;

  bool in_bounds_ij = (i < nx && j < ny);
  double diag = 0;

  // where we are in k depends on our position in the group
  int k = threadIdx.x;

  // do not shift k by -1.
  int idx = __mul24(i,xstride) + __mul24(j,ystride) + k;
  int tidx = idx + tex_offset;

  // only continue if we are in-bounds in x and y - all threads in a warp
  // will go the same way through this conditional
  if (in_bounds_ij && k < nz) {
    // read 4 u neighbors & b to calc RHS.
    // then do a cyclic reduction with other threads in our group
    // write result to u

    // load/stores will be perfectly coalesced

    double bval = b_buffer[idx];

    // no need for this now
    //double ucent = tex1Dfetchd_U(tidx);
    //local_U[k] = ucent;

    double RHS = -hsq * bval;
    RHS += fx * (tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride));
    RHS += fy * (tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride));
//    RHS += fx * (u_buffer[idx - xstride] + u_buffer[idx + xstride]);
//    RHS += fy * (u_buffer[idx - ystride] + u_buffer[idx + ystride]);
    
    local_RHS[k] = RHS;

    diag = laplace_diag;
    if (k==nzm) diag += zpos_mod;
    if (!k    ) diag += zneg_mod;

    local_Mii[k] = diag;
    // no need for out of bounds checking in z here since they will be ignored below if oob
    local_Mip[k] = -fz;
    local_Mim[k] = -fz;

  }

  __syncthreads();

  // do cyclic reduction with other threads in group based on u array & rhs array stored in shmem.
  // ignore in_bounds_ij here since we will check that before we do any writes.

  for(int stride = nz/2; stride > 0; stride /= 2) {

    // copying data from a submatrix of size double_stride, writing
    // to a submatrix of size stride
    int double_stride = stride*2;

    // matrix block we are in starts here:
    int k_start = double_stride * (k / double_stride);
    // and our row within this submatrix is this:
    int k_offset = k - k_start;

    // we will get shuffled to this row (depends on whether we are odd or even)
    int k_target = (k_offset/2) + ((k%2==0) ? k_start : k_start + stride);

    // store results in 3 sets of "registers"
    double reg_m_ii = local_Mii[k];
    double reg_r_ne = local_RHS[k];
    double reg_m_im = 0;
    double reg_m_ip = 0;

    if (k_offset > 0) {
      double alpha_minus = -local_Mim[k] / local_Mii[k-1];
      reg_m_ii += alpha_minus * local_Mip[k-1];
      reg_r_ne += alpha_minus * local_RHS[k-1];
      if (k_offset > 1) {
        reg_m_im = alpha_minus * local_Mim[k-1];
      }
    }

    if (k_offset < double_stride-1) {
      double alpha_plus = -local_Mip[k] / local_Mii[k+1];
      reg_m_ii += alpha_plus * local_Mim[k+1];
      reg_r_ne += alpha_plus * local_RHS[k+1];
      if (k_offset < double_stride-2) {
        reg_m_ip = alpha_plus  * local_Mip[k+1];
      }
    }

    __syncthreads();
  
    // copy registers back into M and RHS
    local_Mim[k_target] = reg_m_im;
    local_Mip[k_target] = reg_m_ip;
    local_Mii[k_target] = reg_m_ii;
    local_RHS[k_target] = reg_r_ne;

    __syncthreads();
  }

  local_U[bit_reverse(k, nz)] = local_RHS[k] / local_Mii[k];

  __syncthreads();

  if (in_bounds_ij && k < nz) {    
    u_buffer[idx] = local_U[k];
  }
}



__global__ void Sol_MultigridPressure3DDeviceD_relax(
  double *u_buffer, int xstride, int ystride, int nx, int ny, int nz, int nxm, int nym, int nzm, int tex_offset, int red_black, 
  double omega, double neg_hsq, double fx, double fy, double fz, double laplace_diag,
  double xpos_mod, double xneg_mod, double ypos_mod, double yneg_mod, double zpos_mod, double zneg_mod, 
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
      
    double bval = tex1Dfetchd_B(tidx);
    double ucent = tex1Dfetchd_U(tidx);

    double residual = (neg_hsq * bval - laplace_diag * ucent);
    residual += fz * (tex1Dfetchd_U(tidx - 1      ) + tex1Dfetchd_U(tidx + 1      ));
    residual += fy * (tex1Dfetchd_U(tidx - ystride) + tex1Dfetchd_U(tidx + ystride));        
    residual += fx * (tex1Dfetchd_U(tidx - xstride) + tex1Dfetchd_U(tidx + xstride));

    double diag = laplace_diag;
    if (i==nxm) diag += xpos_mod;
    if (!i    ) diag += xneg_mod;
    if (j==nym) diag += ypos_mod;
    if (!j    ) diag += yneg_mod;
    if (k==nzm) diag += zpos_mod;
    if (!k    ) diag += zneg_mod;

    // this is all very clever because it handles all boundary conditions with just slight modifications to diag.
    // I need to write up how this works, since it is both clever and obscure.
    double new_val = ucent + omega * residual/diag;

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
Sol_MultigridPressure3DDevice<double>::bind_tex_relax(Grid3DDevice<double> &u_grid, Grid3DDevice<double> &b_grid)
{
  //How can these be made to work with templates?
  cudaChannelFormatDesc channelDesc_U = cudaCreateChannelDesc<int2>();
  cudaChannelFormatDesc channelDesc_B = cudaCreateChannelDesc<int2>();

  // set up texture
  dtex_U.filterMode = cudaFilterModePoint;
  dtex_U.normalized = false;
  dtex_U.channelDesc = channelDesc_U;	

  dtex_B.filterMode = cudaFilterModePoint;
  dtex_B.normalized = false;
  dtex_B.channelDesc = channelDesc_B;	

  if (cudaBindTexture(NULL, &dtex_U, u_grid.buffer(), &channelDesc_U, u_grid.num_allocated_elements() * sizeof(double)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceD::bind_tex_relax - Could not bind texture U\n");
    return false;
  }

  if (cudaBindTexture(NULL, &dtex_B, b_grid.buffer(), &channelDesc_B, b_grid.num_allocated_elements() * sizeof(double)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceD::bind_tex_relax - Could not bind texture B\n");
    return false;
  }

  return true;
}

template<>
bool 
Sol_MultigridPressure3DDevice<double>::unbind_tex_relax()
{
  cudaUnbindTexture(&dtex_U);
  cudaUnbindTexture(&dtex_B);
  return true;
}

template<>
bool 
Sol_MultigridPressure3DDevice<double>::bind_tex_calculate_residual(Grid3DDevice<double> &u_grid, Grid3DDevice<double> &b_grid)
{
  //How can these be made to work with templates?
  cudaChannelFormatDesc channelDesc_U = cudaCreateChannelDesc<int2>();
  cudaChannelFormatDesc channelDesc_B = cudaCreateChannelDesc<int2>();

  // set up texture
  dtex_U.filterMode = cudaFilterModePoint;
  dtex_U.normalized = false;
  dtex_U.channelDesc = channelDesc_U;	

  dtex_B.filterMode = cudaFilterModePoint;
  dtex_B.normalized = false;
  dtex_B.channelDesc = channelDesc_B;	

  if (cudaBindTexture(NULL, &dtex_U, u_grid.buffer(), &channelDesc_U, u_grid.num_allocated_elements() * sizeof(double)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceD::bind_tex_calculate_residual - Could not bind texture U\n");
    return false;
  }

  if (cudaBindTexture(NULL, &dtex_B, b_grid.buffer(), &channelDesc_B, b_grid.num_allocated_elements() * sizeof(double)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceD::bind_tex_calculate_residual - Could not bind texture B\n");
    return false;
  }

  return true;
}

template<>
bool 
Sol_MultigridPressure3DDevice<double>::unbind_tex_calculate_residual()
{
  cudaUnbindTexture(&dtex_U);
  cudaUnbindTexture(&dtex_B);
  return true;
}

template<>
bool 
Sol_MultigridPressure3DDevice<double>::bind_tex_prolong(Grid3DDevice<double> &u_coarse_grid, Grid3DDevice<double> &u_fine_grid)
{
  //How can these be made to work with templates?
  cudaChannelFormatDesc channelDesc_U_C = cudaCreateChannelDesc<int2>();
  cudaChannelFormatDesc channelDesc_U_F = cudaCreateChannelDesc<int2>();

  // set up texture
  dtex_U_C.filterMode = cudaFilterModePoint;
  dtex_U_C.normalized = false;
  dtex_U_C.channelDesc = channelDesc_U_C;	

  dtex_U_F.filterMode = cudaFilterModePoint;
  dtex_U_F.normalized = false;
  dtex_U_F.channelDesc = channelDesc_U_F;	


  if (cudaBindTexture(NULL, &dtex_U_C, u_coarse_grid.buffer(), &channelDesc_U_C, u_coarse_grid.num_allocated_elements() * sizeof(double)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceD::bind_tex_prolong - Could not bind texture U_C\n");
    return false;
  }
  
  if (cudaBindTexture(NULL, &dtex_U_F, u_fine_grid.buffer(), &channelDesc_U_F, u_fine_grid.num_allocated_elements() * sizeof(double)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceD::bind_tex_prolong - Could not bind texture U_F\n");
    return false;
  }

  return true;
}

template<>
bool 
Sol_MultigridPressure3DDevice<double>::unbind_tex_prolong()
{
  cudaUnbindTexture(&dtex_U_C);
  cudaUnbindTexture(&dtex_U_F);
  return true;
}

template<>
bool 
Sol_MultigridPressure3DDevice<double>::invoke_kernel_relax(Grid3DDevice<double> &u_grid, Grid3DDevice<double> &b_grid, Grid2DDevice<double> &diag_grid, int red_black, double h)
{
  int tnx = (u_grid.nz()+1)/2;
  int tny = u_grid.ny();
  int tnz = u_grid.nx();

  int threadsInX = 16;
  int threadsInY = 4;
  int threadsInZ = 4;

  int blocksInX = (tnx+threadsInX-1)/threadsInX;
  int blocksInY = (tny+threadsInY-1)/threadsInY;
  int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

  PreKernel();

#if 0
  tnx = u_grid.nx();
  tny = u_grid.ny();
  tnz = (u_grid.nz()+1)/2;

  threadsInX = 32;
  threadsInY = 2;
  threadsInZ = 2;

  blocksInX = (tnx+threadsInZ-1)/threadsInZ;
  blocksInY = (tny+threadsInY-1)/threadsInY;

  Dg = dim3(blocksInX, blocksInY);
  Db = dim3(threadsInX, threadsInY, threadsInZ);
  Sol_MultigridPressure3DDeviceD_relax_VECTOR<<<Dg,Db>>>(&u_grid.at(0,0,0), u_grid.xstride(), u_grid.ystride(),
    u_grid.nx(), u_grid.ny(), u_grid.nz(), u_grid.nx()-1, u_grid.ny()-1, u_grid.nz()-1, 
    u_grid.shift_amount(), red_black, 
    (double)_omega, (double)-h*h, (double)_fx, (double)_fy, (double)_fz,  (double)(2*_fx + 2*_fy +2*_fz),
    (double)bc_diag_mod(bc.xpos, _fx), (double)bc_diag_mod(bc.xneg, _fx), (double)bc_diag_mod(bc.ypos, _fy), 
    (double)bc_diag_mod(bc.yneg, _fy), (double)bc_diag_mod(bc.zpos, _fz), (double)bc_diag_mod(bc.zneg, _fz));
  tnz = u_grid.nx();
#elif 0
  tnx = (u_grid.nz()/2);
  tny = u_grid.ny();
  tnz = u_grid.nx();

  threadsInX = 32;
  threadsInY = 4;
  threadsInZ = 4;

  blocksInX = (tnx+threadsInX-1)/threadsInX;
  blocksInY = (tny+threadsInY-1)/threadsInY;
  blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  Dg = dim3(blocksInX, blocksInY*blocksInZ);
  Db = dim3(threadsInX, threadsInY, threadsInZ);
  Sol_MultigridPressure3DDeviceD_relax_TEX<<<Dg,Db>>>(&b_grid.at(0,0,0), &u_grid.at(0,0,0), u_grid.xstride(), u_grid.ystride(),
    u_grid.nx(), u_grid.ny(), u_grid.nz(), u_grid.nx()-1, u_grid.ny()-1, u_grid.nz()-1, 
    u_grid.shift_amount(), red_black, 
    (double)_omega, (double)-h*h, (double)_fx, (double)_fy, (double)_fz,  (double)(2*_fx + 2*_fy +2*_fz),
    (double)bc_diag_mod(bc.xpos, _fx), (double)bc_diag_mod(bc.xneg, _fx), (double)bc_diag_mod(bc.ypos, _fy), 
    (double)bc_diag_mod(bc.yneg, _fy), (double)bc_diag_mod(bc.zpos, _fz), (double)bc_diag_mod(bc.zneg, _fz), 
    blocksInY, 1.0f/(float)blocksInY);

#elif 0
  tnx = u_grid.nx();
  tny = u_grid.ny();
  tnz = (u_grid.nz()+1)/2;

  threadsInX = 32;
  threadsInY = 2;
  threadsInZ = 2;

  blocksInX = (tnx+threadsInZ-1)/threadsInZ;
  blocksInY = (tny+threadsInY-1)/threadsInY;

  Dg = dim3(blocksInX, blocksInY);
  Db = dim3(threadsInX, threadsInY, threadsInZ);
  Sol_MultigridPressure3DDeviceD_relax_VECTORDIAG<<<Dg,Db>>>(&u_grid.at(0,0,0), u_grid.xstride(), u_grid.ystride(),
    u_grid.nx(), u_grid.ny(), u_grid.nz(), u_grid.nx()-1, u_grid.ny()-1, u_grid.nz()-1, 
    u_grid.shift_amount(), red_black, 
    (double)_omega, (double)-h*h, (double)_fx, (double)_fy, (double)_fz,
    (double)(2*_fx + 2*_fy +2*_fz),
    &diag_grid.at(0,0), diag_grid.xstride(),
    (double)bc_diag_mod(bc.zpos, _fz), (double)bc_diag_mod(bc.zneg, _fz));
  tnz = u_grid.nx();
#elif 0

  threadsInX = 16;
  threadsInY = 2;
  threadsInZ = 2;

  blocksInX = (tnx+threadsInX-1)/threadsInX;
  blocksInY = (tny+threadsInY-1)/threadsInY;
  blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  Dg = dim3(blocksInX, blocksInY*blocksInZ);
  Db = dim3(threadsInX, threadsInY, threadsInZ);
  unsigned int smem_size = sizeof(double) * (threadsInX+1) * threadsInY * threadsInZ;
  Sol_MultigridPressure3DDeviceD_relax_CACHE<<<Dg,Db,smem_size>>>(&u_grid.at(0,0,0), u_grid.xstride(), u_grid.ystride(), u_grid.nx(), u_grid.ny(), u_grid.nz(), u_grid.shift_amount(), red_black, 
    (double)_omega, (double)h*h, (double)_fx, (double)_fy, (double)_fz,  (double)(2*_fx + 2*_fy +2*_fz),
    (double)bc_diag_mod(bc.xpos, _fx), (double)bc_diag_mod(bc.xneg, _fx), (double)bc_diag_mod(bc.ypos, _fy), 
    (double)bc_diag_mod(bc.yneg, _fy), (double)bc_diag_mod(bc.zpos, _fz), (double)bc_diag_mod(bc.zneg, _fz), 
    blocksInY, 1.0f/(float)blocksInY,(threadsInX+1),(threadsInX+1) * threadsInY);
#elif 0

  threadsInX = 16;
  threadsInY = 2;
  threadsInZ = 2;

  blocksInX = (tnx+threadsInX-1)/threadsInX;
  blocksInY = (tny+threadsInY-1)/threadsInY;
  blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  Dg = dim3(blocksInX, blocksInY*blocksInZ);
  Db = dim3(threadsInX, threadsInY, threadsInZ);
  unsigned int smem_size = sizeof(double) * (2*threadsInX+2) * threadsInY * threadsInZ;
  Sol_MultigridPressure3DDeviceD_relax_CACHE2<<<Dg,Db,smem_size>>>(&u_grid.at(0,0,0), u_grid.xstride(), u_grid.ystride(), u_grid.nx(), u_grid.ny(), u_grid.nz(), u_grid.shift_amount(), red_black, 
    (double)_omega, (double)h*h, (double)_fx, (double)_fy, (double)_fz,  (double)(2*_fx + 2*_fy +2*_fz),
    (double)bc_diag_mod(bc.xpos, _fx), (double)bc_diag_mod(bc.xneg, _fx), (double)bc_diag_mod(bc.ypos, _fy), 
    (double)bc_diag_mod(bc.yneg, _fy), (double)bc_diag_mod(bc.zpos, _fz), (double)bc_diag_mod(bc.zneg, _fz), 
    blocksInY, 1.0f/(float)blocksInY,(2*threadsInX+2),(2*threadsInX+2) * threadsInY);
#elif 1 // best point relaxer 
  tnx = (u_grid.nz()/2);
  tny = u_grid.ny();
  tnz = u_grid.nx();

  threadsInX = 32;
  threadsInY = 4;
  threadsInZ = 4;

  blocksInX = (tnx+threadsInX-1)/threadsInX;
  blocksInY = (tny+threadsInY-1)/threadsInY;
  blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  Dg = dim3(blocksInX, blocksInY*blocksInZ);
  Db = dim3(threadsInX, threadsInY, threadsInZ);
  //printf("threads: %d %d %d\n", blocksInX*threadsInX, blocksInY*threadsInY, blocksInZ*threadsInZ);
  Sol_MultigridPressure3DDeviceD_relax_DIAG<<<Dg,Db>>>(&b_grid.at(0,0,0), &u_grid.at(0,0,0), u_grid.xstride(), u_grid.ystride(),
    u_grid.nx(), u_grid.ny(), u_grid.nz(), u_grid.nx()-1, u_grid.ny()-1, u_grid.nz()-1, 
    u_grid.shift_amount(), red_black, 
    (double)_omega, (double)-h*h, (double)_fx, (double)_fy, (double)_fz,  (double)(2*_fx + 2*_fy +2*_fz),
    &diag_grid.at(0,0), diag_grid.xstride(),
    (double)bc_diag_mod(bc.zpos, _fz), (double)bc_diag_mod(bc.zneg, _fz), 
    blocksInY, 1.0f/(float)blocksInY);
#elif 0 // second place
  tnx = u_grid.nz();
  tny = u_grid.ny()/2;
  tnz = u_grid.nx();

  threadsInX = 32;
  threadsInY = 4;
  threadsInZ = 4;

  blocksInX = (tnx+threadsInX-1)/threadsInX;
  blocksInY = (tny+threadsInY-1)/threadsInY;
  blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  Dg = dim3(blocksInX, blocksInY*blocksInZ);
  Db = dim3(threadsInX, threadsInY, threadsInZ);
  Sol_MultigridPressure3DDeviceD_relax_ALTDIAG<<<Dg,Db>>>(&b_grid.at(0,0,0), &u_grid.at(0,0,0), u_grid.xstride(), u_grid.ystride(),
    u_grid.nx(), u_grid.ny(), u_grid.nz(), u_grid.nx()-1, u_grid.ny()-1, u_grid.nz()-1, 
    u_grid.shift_amount(), red_black, 
    (double)_omega, (double)-h*h, (double)_fx, (double)_fy, (double)_fz,  (double)(2*_fx + 2*_fy +2*_fz),
    &diag_grid.at(0,0), diag_grid.xstride(),
    (double)bc_diag_mod(bc.zpos, _fz), (double)bc_diag_mod(bc.zneg, _fz), 
    blocksInY, 1.0f/(float)blocksInY);
#elif 0
  threadsInX = 16;
  threadsInY = 2;
  threadsInZ = 2;

  blocksInX = (tnx+threadsInX-1)/threadsInX;
  blocksInY = (tny+threadsInY-1)/threadsInY;
  blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  Dg = dim3(blocksInX, blocksInY*blocksInZ);
  Db = dim3(threadsInX, threadsInY, threadsInZ);
  Sol_MultigridPressure3DDeviceD_relax<<<Dg,Db>>>(&u_grid.at(0,0,0), u_grid.xstride(), u_grid.ystride(),
    u_grid.nx(), u_grid.ny(), u_grid.nz(), u_grid.nx()-1, u_grid.ny()-1, u_grid.nz()-1, 
    u_grid.shift_amount(), red_black, 
    (double)_omega, (double)-h*h, (double)_fx, (double)_fy, (double)_fz,  (double)(2*_fx + 2*_fy +2*_fz),
    (double)bc_diag_mod(bc.xpos, _fx), (double)bc_diag_mod(bc.xneg, _fx), (double)bc_diag_mod(bc.ypos, _fy), 
    (double)bc_diag_mod(bc.yneg, _fy), (double)bc_diag_mod(bc.zpos, _fz), (double)bc_diag_mod(bc.zneg, _fz), 
    blocksInY, 1.0f/(float)blocksInY);
#elif 0 // line relaxer

  // NB: this will not work with periodic boundary conditions.

  tny = u_grid.ny();
  tnz = (u_grid.nx()+1)/2;

  threadsInX = max(32 * (u_grid.nz()/32),32);
  threadsInY = 1;
  threadsInZ = 1;

  //printf("threadsInX = %d\n", threadsInX);

  blocksInX = 1;
  blocksInY = (tny+threadsInY-1)/threadsInY;
  blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  Dg = dim3(blocksInZ, blocksInY);
  Db = dim3(threadsInX, threadsInY, threadsInZ);

  unsigned int shmem = threadsInX * threadsInY * threadsInZ * sizeof(double) * 4;
  Sol_MultigridPressure3DDeviceD_relax_LINERELAX<<<Dg,Db,shmem>>>(&b_grid.at(0,0,0), &u_grid.at(0,0,0), u_grid.xstride(), u_grid.ystride(),
    u_grid.nx(), u_grid.ny(), u_grid.nz(), u_grid.nx()-1, u_grid.ny()-1, u_grid.nz()-1, 
    u_grid.shift_amount(), red_black, 
    (double)_omega, (double)h*h, (double)_fx, (double)_fy, (double)_fz,  (double)(2*_fx + 2*_fy +2*_fz),    
    (double)bc_diag_mod(bc.zpos, _fz), (double)bc_diag_mod(bc.zneg, _fz));
  tnz = u_grid.nx();
#elif 0
  // NB: this will not currently work with periodic boundary conditions.

  tny = u_grid.ny();
  tnz = (u_grid.nx()+1)/2;

  // half nz + 1, rounded up to nearest multiple of 32.
  threadsInX = 32 * (((u_grid.nz() / 2) + 1) + 31)/32;
  // half nz, rounded up to nearest multiple of 32.
  //threadsInX = 32 * ((u_grid.nz() / 2) + 31)/32;
  // nz rounded up to nearest multiple of 32
  //threadsInX = 32 * (u_grid.nz()+31)/32;
  threadsInY = 2;
  threadsInZ = 2;

  //printf("threadsInX = %d\n", threadsInX);
  blocksInX = 1;
  blocksInY = (tny+threadsInY-1)/threadsInY;
  blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

  Dg = dim3(blocksInZ, blocksInY);
  Db = dim3(threadsInX, threadsInY, threadsInZ);

  unsigned int shmem = u_grid.nz() * threadsInY * threadsInZ * sizeof(double)*2;
  Sol_MultigridPressure3DDeviceD_relax_LINERB<<<Dg,Db,shmem>>>(&b_grid.at(0,0,0), &u_grid.at(0,0,0), u_grid.xstride(), u_grid.ystride(),
    u_grid.nx(), u_grid.ny(), u_grid.nz(), u_grid.nx()-1, u_grid.ny()-1, u_grid.nz()-1, 
    u_grid.shift_amount(), red_black, 
    (double)_omega, (double)-h*h, (double)_fx, (double)_fy, (double)_fz,  (double)(2*_fx + 2*_fy +2*_fz),  
    &get_diag(g_relax_level).at(0,0), get_diag(g_relax_level).xstride(),
    (double)bc_diag_mod(bc.zpos, _fz), (double)bc_diag_mod(bc.zneg, _fz));
  tnz = u_grid.nx();
#endif

  return PostKernel("Sol_MultigridPressure3DDeviceD_relax", tnz);
}

template<>
bool 
Sol_MultigridPressure3DDevice<double>::invoke_kernel_calculate_residual(Grid3DDevice<double> &u_grid, Grid3DDevice<double> &b_grid, Grid3DDevice<double> &r_grid, double h)
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
  Sol_MultigridPressure3DDeviceD_calculate_residual<<<Dg, Db>>>(&b_grid.at(0,0,0), &r_grid.at(0,0,0), r_grid.xstride(), r_grid.ystride(), r_grid.nx(), r_grid.ny(), r_grid.nz(), r_grid.shift_amount(),
    fx_div_hsq, fy_div_hsq, fz_div_hsq, diag, blocksInY, 1.0f/(float)blocksInY);

  return PostKernel("Sol_MultigridPressure3DDeviceD_calculate_residual", tnz);
}

template<>
bool 
Sol_MultigridPressure3DDevice<double>::invoke_kernel_restrict(Grid3DDevice<double> &r_grid, Grid3DDevice<double> &b_coarse_grid)
{
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
  Sol_MultigridPressure3DDeviceD_restrict<<<Dg, Db>>>(&r_grid.at(0,0,0), r_grid.xstride(), r_grid.ystride(), 
    &b_coarse_grid.at(0,0,0), b_coarse_grid.xstride(), b_coarse_grid.ystride(), b_coarse_grid.nx(), b_coarse_grid.ny(), b_coarse_grid.nz(), 
    blocksInY, 1.0f/(float)blocksInY);
  return PostKernel("Sol_MultigridPressure3DDeviceD_restrict", tnz);
}

template<>
bool 
Sol_MultigridPressure3DDevice<double>::invoke_kernel_prolong(Grid3DDevice<double> &u_coarse_grid, Grid3DDevice<double> &u_fine_grid)
{
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
  Sol_MultigridPressure3DDeviceD_prolong<<<Dg, Db>>>(&u_fine_grid.at(0,0,0), u_fine_grid.xstride(), u_fine_grid.ystride(), u_fine_grid.shift_amount(),
    u_coarse_grid.xstride(), u_coarse_grid.ystride(), u_coarse_grid.nx(), u_coarse_grid.ny(), u_coarse_grid.nz(),  u_coarse_grid.shift_amount(),
    blocksInY, 1.0f/(float)blocksInY);
  return PostKernel("Sol_MultigridPressure3DDeviceD_prolong", tnz);
}


} // end namespace
#endif // OCU_DOUBLESUPPORT

