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
#include <cuda.h>

#include "ocuutil/float_routines.h"
#include "ocustorage/grid3dops.h"
#include "ocuequation/sol_selfadvection3d.h"


//! This routine works because u,v,w,phi, and dphidt must all be padded so that they have the same memory layout,
//! even though they have different dimensions.  Then we can calculate indexing math once, and reuse it for
//! all of the grids.
template<typename T, typename INTERP>
__global__ void Sol_SelfAdvection3D_apply_upwind(
  T *u, T *v, T *w, T *dudt, T *dvdt, T *dwdt, 
  T invhx, T invhy, T invhz, 
  int xstride, int ystride,
  int nx, int ny, int nz, int blocksInY, float invBlocksInY, INTERP interp)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);

  // transpose for coalescing since k is the fastest changing index 
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  if (i < nx && j < ny && k < nz) {
    // calc phi indexing
    int idx = __mul24(i, xstride) + __mul24(j,ystride) + k;
  
    int idx_pi = idx + xstride;
    int idx_pj = idx + ystride;
    int idx_pk = idx + 1;

    int idx_mi = idx - xstride;
    int idx_mj = idx - ystride;
    int idx_mk = idx - 1;

    T u_idx = u[idx];
    T v_idx = v[idx];
    T w_idx = w[idx];

    //---- dudt ----
    T u_iph_j_k = .5 * (u_idx + u[idx_pi]); 
    T u_imh_j_k = .5 * (u_idx + u[idx_mi]);
    T duu = (u_iph_j_k * interp(u_iph_j_k, u_idx    , u[idx_pi])) - 
            (u_imh_j_k * interp(u_imh_j_k, u[idx_mi], u_idx    ));

    T v_atu_i_jph_k = .5* (v[idx_pj] + v[idx_pj - xstride]);
    T v_atu_i_jmh_k = .5* (v_idx     + v[idx_mi]);
    T dvu = (v_atu_i_jph_k * interp(v_atu_i_jph_k, u_idx    , u[idx_pj])) - 
            (v_atu_i_jmh_k * interp(v_atu_i_jmh_k, u[idx_mj], u_idx));

    T w_atu_i_j_kph = .5* (w[idx_pk] + w[idx_pk - xstride]);
    T w_atu_i_j_kmh = .5* (w_idx     + w[idx_mi]);
    T dwu = (w_atu_i_j_kph * interp(w_atu_i_j_kph, u_idx    , u[idx_pk])) - 
            (w_atu_i_j_kmh * interp(w_atu_i_j_kmh, u[idx_mk], u_idx));   

    dudt[idx] = -duu*invhx - dvu*invhy - dwu*invhz;

    //---- dvdt ----
    T u_atv_iph_j_k = .5* (u[idx_pi] + u[idx_pi - ystride]);
    T u_atv_imh_j_k = .5* (u_idx     + u[idx_mj]);
    T duv = (u_atv_iph_j_k * interp(u_atv_iph_j_k, v_idx    , v[idx_pi])) - 
            (u_atv_imh_j_k * interp(u_atv_imh_j_k, v[idx_mi], v_idx));

    T v_i_jph_k = .5 * (v_idx + v[idx_pj]); 
    T v_i_jmh_k = .5 * (v_idx + v[idx_mj]);
    T dvv = (v_i_jph_k * interp(v_i_jph_k, v_idx    , v[idx_pj])) - 
            (v_i_jmh_k * interp(v_i_jmh_k, v[idx_mj], v_idx));

    T w_atv_i_j_kph = .5* (w[idx_pk] + w[idx_pk - ystride]);
    T w_atv_i_j_kmh = .5* (w_idx     + w[idx_mj]);
    T dwv = (w_atv_i_j_kph * interp(w_atv_i_j_kph, v_idx    , v[idx_pk])) - 
            (w_atv_i_j_kmh * interp(w_atv_i_j_kmh, v[idx_mk], v_idx));
    
    dvdt[idx] = -duv*invhx - dvv*invhy - dwv*invhz;

    //---- dwdt ----
    T u_atw_iph_j_k = .5* (u[idx_pi] + u[idx_pi - 1]);
    T u_atw_imh_j_k = .5* (u_idx     + u[idx_mk]);
    T duw = (u_atw_iph_j_k * interp(u_atw_iph_j_k, w_idx    , w[idx_pi])) - 
            (u_atw_imh_j_k * interp(u_atw_imh_j_k, w[idx_mi], w_idx));    

    T v_atw_i_jph_k = .5* (v[idx_pj] + v[idx_pj - 1]);
    T v_atw_i_jmh_k = .5* (v_idx     + v[idx_mk]);
    T dvw = (v_atw_i_jph_k * interp(v_atw_i_jph_k, w_idx    , w[idx_pj])) - 
            (v_atw_i_jmh_k * interp(v_atw_i_jmh_k, w[idx_mj], w_idx));    

    T w_i_j_kph = .5 * (w_idx + w[idx_pk]); 
    T w_i_j_kmh = .5 * (w_idx + w[idx_mk]);
    T dww = (w_i_j_kph * interp(w_i_j_kph, w_idx    , w[idx_pk])) - 
            (w_i_j_kmh * interp(w_i_j_kmh, w[idx_mk], w_idx));

    dwdt[idx] = -duw*invhx - dvw*invhy - dww*invhz;
  }
}

texture<float, 1, cudaReadModeElementType> tex_u;
texture<float, 1, cudaReadModeElementType> tex_v;
texture<float, 1, cudaReadModeElementType> tex_w;

__inline__ __device__ float tex1Dfetch_u(const int& i)
{
  return tex1Dfetch(tex_u, i);
}

__inline__ __device__ float tex1Dfetch_v(const int& i)
{
  return tex1Dfetch(tex_v, i);
}

__inline__ __device__ float tex1Dfetch_w(const int& i)
{
  return tex1Dfetch(tex_w, i);
}

template<typename INTERP>
__global__ void Advection3DF_apply_upwind_TEX(
  float *dudt, float *dvdt, float *dwdt, 
  float invhx, float invhy, float invhz, 
  int xstride, int ystride, int tex_offset,
  int nx, int ny, int nz, int blocksInY, float invBlocksInY, INTERP interp)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);

  // transpose for coalescing since k is the fastest changing index 
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  if (i < nx && j < ny && k < nz) {
    // calc phi indexing
    int idx_no_offset = __mul24(i, xstride) + __mul24(j,ystride) + k;
    int idx = idx_no_offset + tex_offset;
  
    int idx_pi = idx + xstride;
    int idx_pj = idx + ystride;
    int idx_pk = idx + 1;

    int idx_mi = idx - xstride;
    int idx_mj = idx - ystride;
    int idx_mk = idx - 1;

    float u_idx = tex1Dfetch_u(idx);
    float v_idx = tex1Dfetch_v(idx);
    float w_idx = tex1Dfetch_w(idx);

    //---- dudt ----
    float u_iph_j_k = .5f * (u_idx + tex1Dfetch_u(idx_pi)); 
    float u_imh_j_k = .5f * (u_idx + tex1Dfetch_u(idx_mi));
    float duu = (u_iph_j_k * interp(u_iph_j_k, u_idx                , tex1Dfetch_u(idx_pi))) - 
                (u_imh_j_k * interp(u_imh_j_k, tex1Dfetch_u(idx_mi) , u_idx               ));

    float v_atu_i_jph_k = .5f* (tex1Dfetch_v(idx_pj) + tex1Dfetch_v(idx_pj - xstride));
    float v_atu_i_jmh_k = .5f* (v_idx                + tex1Dfetch_v(idx_mi));
    float dvu = (v_atu_i_jph_k * interp(v_atu_i_jph_k, u_idx               , tex1Dfetch_u(idx_pj))) - 
                (v_atu_i_jmh_k * interp(v_atu_i_jmh_k, tex1Dfetch_u(idx_mj), u_idx));

    float w_atu_i_j_kph = .5f* (tex1Dfetch_w(idx_pk) + tex1Dfetch_w(idx_pk - xstride));
    float w_atu_i_j_kmh = .5f* (w_idx                + tex1Dfetch_w(idx_mi));
    float dwu = (w_atu_i_j_kph * interp(w_atu_i_j_kph, u_idx               , tex1Dfetch_u(idx_pk))) - 
                (w_atu_i_j_kmh * interp(w_atu_i_j_kmh, tex1Dfetch_u(idx_mk), u_idx));   

    dudt[idx_no_offset] = -duu*invhx - dvu*invhy - dwu*invhz;

    //---- dvdt ----
    float u_atv_iph_j_k = .5f* (tex1Dfetch_u(idx_pi) + tex1Dfetch_u(idx_pi - ystride));
    float u_atv_imh_j_k = .5f* (u_idx                + tex1Dfetch_u(idx_mj));
    float duv = (u_atv_iph_j_k * interp(u_atv_iph_j_k, v_idx                , tex1Dfetch_v(idx_pi))) - 
                (u_atv_imh_j_k * interp(u_atv_imh_j_k, tex1Dfetch_v(idx_mi), v_idx));

    float v_i_jph_k = .5f * (v_idx + tex1Dfetch_v(idx_pj)); 
    float v_i_jmh_k = .5f * (v_idx + tex1Dfetch_v(idx_mj));
    float dvv = (v_i_jph_k * interp(v_i_jph_k, v_idx               , tex1Dfetch_v(idx_pj))) - 
                (v_i_jmh_k * interp(v_i_jmh_k, tex1Dfetch_v(idx_mj), v_idx                ));

    float w_atv_i_j_kph = .5f* (tex1Dfetch_w(idx_pk) + tex1Dfetch_w(idx_pk - ystride));
    float w_atv_i_j_kmh = .5f* (w_idx                + tex1Dfetch_w(idx_mj));
    float dwv = (w_atv_i_j_kph * interp(w_atv_i_j_kph, v_idx               , tex1Dfetch_v(idx_pk))) - 
                (w_atv_i_j_kmh * interp(w_atv_i_j_kmh, tex1Dfetch_v(idx_mk), v_idx));
    
    dvdt[idx_no_offset] = -duv*invhx - dvv*invhy - dwv*invhz;

    //---- dwdt ----
    float u_atw_iph_j_k = .5f* (tex1Dfetch_u(idx_pi) + tex1Dfetch_u(idx_pi - 1));
    float u_atw_imh_j_k = .5f* (u_idx                + tex1Dfetch_u(idx_mk));
    float duw = (u_atw_iph_j_k * interp(u_atw_iph_j_k, w_idx                , tex1Dfetch_w(idx_pi))) - 
                (u_atw_imh_j_k * interp(u_atw_imh_j_k, tex1Dfetch_w(idx_mi), w_idx));    

    float v_atw_i_jph_k = .5f* (tex1Dfetch_v(idx_pj) + tex1Dfetch_v(idx_pj - 1));
    float v_atw_i_jmh_k = .5f* (v_idx                + tex1Dfetch_v(idx_mk));
    float dvw = (v_atw_i_jph_k * interp(v_atw_i_jph_k, w_idx                , tex1Dfetch_w(idx_pj))) - 
                (v_atw_i_jmh_k * interp(v_atw_i_jmh_k, tex1Dfetch_w(idx_mj), w_idx));    

    float w_i_j_kph = .5f * (w_idx + tex1Dfetch_w(idx_pk)); 
    float w_i_j_kmh = .5f * (w_idx + tex1Dfetch_w(idx_mk));
    float dww = (w_i_j_kph * interp(w_i_j_kph, w_idx               , tex1Dfetch_w(idx_pk))) - 
                (w_i_j_kmh * interp(w_i_j_kmh, tex1Dfetch_w(idx_mk), w_idx                ));

    dwdt[idx_no_offset] = -duw*invhx - dvw*invhy - dww*invhz;
  }
}



#ifdef OCU_DOUBLESUPPORT

texture<int2, 1, cudaReadModeElementType> dtex_u;
texture<int2, 1, cudaReadModeElementType> dtex_v;
texture<int2, 1, cudaReadModeElementType> dtex_w;

__inline__ __device__ double tex1Dfetchd_u(const int& i)
{
  int2 v = tex1Dfetch(dtex_u, i);
  return __hiloint2double(v.y, v.x);
}

__inline__ __device__ double tex1Dfetchd_v(const int& i)
{
  int2 v = tex1Dfetch(dtex_v, i);
  return __hiloint2double(v.y, v.x);
}

__inline__ __device__ double tex1Dfetchd_w(const int& i)
{
  int2 v = tex1Dfetch(dtex_w, i);
  return __hiloint2double(v.y, v.x);
}

template<typename INTERP>
__global__ void Advection3DD_apply_upwind_TEX(
  double *dudt, double *dvdt, double *dwdt, 
  double invhx, double invhy, double invhz, 
  int xstride, int ystride, int tex_offset,
  int nx, int ny, int nz, int blocksInY, float invBlocksInY, INTERP interp)
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
    int idx_no_offset = __mul24(i, xstride) + __mul24(j,ystride) + k;
    int idx = idx_no_offset + tex_offset;
  
    int idx_pi = idx + xstride;
    int idx_pj = idx + ystride;
    int idx_pk = idx + 1;

    int idx_mi = idx - xstride;
    int idx_mj = idx - ystride;
    int idx_mk = idx - 1;

    double u_idx = tex1Dfetchd_u(idx);
    double v_idx = tex1Dfetchd_v(idx);
    double w_idx = tex1Dfetchd_w(idx);

    //---- dudt ----
    double u_iph_j_k = .5 * (u_idx + tex1Dfetchd_u(idx_pi)); 
    double u_imh_j_k = .5 * (u_idx + tex1Dfetchd_u(idx_mi));
    double duu = (u_iph_j_k * interp(u_iph_j_k, u_idx                , tex1Dfetchd_u(idx_pi))) - 
                 (u_imh_j_k * interp(u_imh_j_k, tex1Dfetchd_u(idx_mi), u_idx                ));

    double v_atu_i_jph_k = .5* (tex1Dfetchd_v(idx_pj) + tex1Dfetchd_v(idx_pj - xstride));
    double v_atu_i_jmh_k = .5* (v_idx                 + tex1Dfetchd_v(idx_mi));
    double dvu = (v_atu_i_jph_k * interp(v_atu_i_jph_k, u_idx                , tex1Dfetchd_u(idx_pj))) - 
                 (v_atu_i_jmh_k * interp(v_atu_i_jmh_k, tex1Dfetchd_u(idx_mj), u_idx));

    double w_atu_i_j_kph = .5* (tex1Dfetchd_w(idx_pk) + tex1Dfetchd_w(idx_pk - xstride));
    double w_atu_i_j_kmh = .5* (w_idx                 + tex1Dfetchd_w(idx_mi));
    double dwu = (w_atu_i_j_kph * interp(w_atu_i_j_kph, u_idx                , tex1Dfetchd_u(idx_pk))) - 
                 (w_atu_i_j_kmh * interp(w_atu_i_j_kmh, tex1Dfetchd_u(idx_mk), u_idx));   

    dudt[idx_no_offset] = -duu*invhx - dvu*invhy - dwu*invhz;

    //---- dvdt ----
    double u_atv_iph_j_k = .5* (tex1Dfetchd_u(idx_pi) + tex1Dfetchd_u(idx_pi - ystride));
    double u_atv_imh_j_k = .5* (u_idx                 + tex1Dfetchd_u(idx_mj));
    double duv = (u_atv_iph_j_k * interp(u_atv_iph_j_k, v_idx                , tex1Dfetchd_v(idx_pi))) - 
                 (u_atv_imh_j_k * interp(u_atv_imh_j_k, tex1Dfetchd_v(idx_mi), v_idx));

    double v_i_jph_k = .5 * (v_idx + tex1Dfetchd_v(idx_pj)); 
    double v_i_jmh_k = .5 * (v_idx + tex1Dfetchd_v(idx_mj));
    double dvv = (v_i_jph_k * interp(v_i_jph_k, v_idx                , tex1Dfetchd_v(idx_pj))) - 
                 (v_i_jmh_k * interp(v_i_jmh_k, tex1Dfetchd_v(idx_mj), v_idx                ));

    double w_atv_i_j_kph = .5* (tex1Dfetchd_w(idx_pk) + tex1Dfetchd_w(idx_pk - ystride));
    double w_atv_i_j_kmh = .5* (w_idx                 + tex1Dfetchd_w(idx_mj));
    double dwv = (w_atv_i_j_kph * interp(w_atv_i_j_kph, v_idx                , tex1Dfetchd_v(idx_pk))) - 
                 (w_atv_i_j_kmh * interp(w_atv_i_j_kmh, tex1Dfetchd_v(idx_mk), v_idx));
    
    dvdt[idx_no_offset] = -duv*invhx - dvv*invhy - dwv*invhz;

    //---- dwdt ----
    double u_atw_iph_j_k = .5* (tex1Dfetchd_u(idx_pi) + tex1Dfetchd_u(idx_pi - 1));
    double u_atw_imh_j_k = .5* (u_idx                 + tex1Dfetchd_u(idx_mk));
    double duw = (u_atw_iph_j_k * interp(u_atw_iph_j_k, w_idx                , tex1Dfetchd_w(idx_pi))) - 
                 (u_atw_imh_j_k * interp(u_atw_imh_j_k, tex1Dfetchd_w(idx_mi), w_idx));    

    double v_atw_i_jph_k = .5* (tex1Dfetchd_v(idx_pj) + tex1Dfetchd_v(idx_pj - 1));
    double v_atw_i_jmh_k = .5* (v_idx                 + tex1Dfetchd_v(idx_mk));
    double dvw = (v_atw_i_jph_k * interp(v_atw_i_jph_k, w_idx                , tex1Dfetchd_w(idx_pj))) - 
                 (v_atw_i_jmh_k * interp(v_atw_i_jmh_k, tex1Dfetchd_w(idx_mj), w_idx));    

    double w_i_j_kph = .5 * (w_idx + tex1Dfetchd_w(idx_pk)); 
    double w_i_j_kmh = .5 * (w_idx + tex1Dfetchd_w(idx_mk));
    double dww = (w_i_j_kph * interp(w_i_j_kph, w_idx                , tex1Dfetchd_w(idx_pk))) - 
                 (w_i_j_kmh * interp(w_i_j_kmh, tex1Dfetchd_w(idx_mk), w_idx                ));

    dwdt[idx_no_offset] = -duw*invhx - dvw*invhy - dww*invhz;
  }
}


#endif // OCU_DOUBLESUPPORT




namespace ocu {

template<typename T>
Sol_SelfAdvection3DDevice<T>::Sol_SelfAdvection3DDevice()
{
  _nx = _ny = _nz = 0;
  u = 0;
  v = 0;
  w = 0;
  interp_type = IT_FIRST_ORDER_UPWIND;
}

template<typename T>
Sol_SelfAdvection3DDevice<T>::~Sol_SelfAdvection3DDevice()
{
  unbind_textures();
}


template<typename T>
bool 
Sol_SelfAdvection3DDevice<T>::solve_naive()
{
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
  if (interp_type == IT_FIRST_ORDER_UPWIND) {    
    Sol_SelfAdvection3D_apply_upwind<<<Dg, Db>>>(&u->at(0,0,0),&v->at(0,0,0),&w->at(0,0,0),&deriv_udt->at(0,0,0),&deriv_vdt->at(0,0,0),&deriv_wdt->at(0,0,0),
      (T)(1/_hx), (T)(1/_hy), (T)(1/_hz), u->xstride(), u->ystride(),
      _nx, _ny, _nz, blocksInY, 1.0f / (float)blocksInY, InterpolatorFirstOrderUpwind<T>());
  }
  else if (interp_type == IT_SECOND_ORDER_CENTERED) {
    Sol_SelfAdvection3D_apply_upwind<<<Dg, Db>>>(&u->at(0,0,0),&v->at(0,0,0),&w->at(0,0,0),&deriv_udt->at(0,0,0),&deriv_vdt->at(0,0,0),&deriv_wdt->at(0,0,0),
      (T)(1/_hx), (T)(1/_hy), (T)(1/_hz), u->xstride(), u->ystride(),
      _nx, _ny, _nz, blocksInY, 1.0f / (float)blocksInY, InterpolatorSecondOrderCentered<T>());
  }
  else {
    printf("[ERROR] Sol_SelfAdvection3DDevice::solve_naive - invalid interpolation type %d\n", interp_type);
    return false;
  }

  return PostKernel("Sol_SelfAdvection3D_apply_upwind");
}

template<>
bool
Sol_SelfAdvection3DDevice<float>::bind_textures()
{
  cudaChannelFormatDesc channelDesc_u = cudaCreateChannelDesc<float>();
  cudaChannelFormatDesc channelDesc_v = cudaCreateChannelDesc<float>();
  cudaChannelFormatDesc channelDesc_w = cudaCreateChannelDesc<float>();

  // set up texture
  tex_u.filterMode = cudaFilterModePoint;
  tex_u.normalized = false;
  tex_u.channelDesc = channelDesc_u;	

  tex_v.filterMode = cudaFilterModePoint;
  tex_v.normalized = false;
  tex_v.channelDesc = channelDesc_v;	

  tex_w.filterMode = cudaFilterModePoint;
  tex_w.normalized = false;
  tex_w.channelDesc = channelDesc_w;	

  if (cudaBindTexture(NULL, &tex_u, u->buffer(), &channelDesc_u, u->num_allocated_elements() * sizeof(float)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_SelfAdvection3DDevice<float>::solve_tex - Could not bind texture u\n");
    return false;
  }
  
  if (cudaBindTexture(NULL, &tex_v, v->buffer(), &channelDesc_v, v->num_allocated_elements() * sizeof(float)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_SelfAdvection3DDevice<float>::solve_tex - Could not bind texture v\n");
    return false;
  }

  if (cudaBindTexture(NULL, &tex_w, w->buffer(), &channelDesc_w, w->num_allocated_elements() * sizeof(float)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_SelfAdvection3DDevice<float>::solve_tex - Could not bind texture w\n");
    return false;
  }

  return true;
}


template<>
bool
Sol_SelfAdvection3DDevice<float>::unbind_textures()
{
  cudaUnbindTexture(&tex_u);
  cudaUnbindTexture(&tex_v);
  cudaUnbindTexture(&tex_w);

  return true;
}

template<>
bool 
Sol_SelfAdvection3DDevice<float>::solve_tex()
{
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
  
  if (interp_type == IT_FIRST_ORDER_UPWIND) {    
    Advection3DF_apply_upwind_TEX<<<Dg, Db>>>(&deriv_udt->at(0,0,0),&deriv_vdt->at(0,0,0),&deriv_wdt->at(0,0,0),
      (float)(1/_hx), (float)(1/_hy), (float)(1/_hz), u->xstride(), u->ystride(), u->shift_amount(),
      _nx, _ny, _nz, blocksInY, 1.0f / (float)blocksInY, InterpolatorFirstOrderUpwind<float>());
  } else if (interp_type == IT_SECOND_ORDER_CENTERED) {
    Advection3DF_apply_upwind_TEX<<<Dg, Db>>>(&deriv_udt->at(0,0,0),&deriv_vdt->at(0,0,0),&deriv_wdt->at(0,0,0),
      (float)(1/_hx), (float)(1/_hy), (float)(1/_hz), u->xstride(), u->ystride(), u->shift_amount(),
      _nx, _ny, _nz, blocksInY, 1.0f / (float)blocksInY, InterpolatorSecondOrderCentered<float>());
  }
  else {
    printf("[ERROR] Sol_SelfAdvection3DDevice::solve_tex - invalid interpolation type %d\n", interp_type);
    return false;
  }

  return PostKernel("Advection3DF_apply_upwind_TEX");
}

#ifdef OCU_DOUBLESUPPORT
template<>
bool
Sol_SelfAdvection3DDevice<double>::bind_textures()
{
  cudaChannelFormatDesc channelDesc_u = cudaCreateChannelDesc<int2>();
  cudaChannelFormatDesc channelDesc_v = cudaCreateChannelDesc<int2>();
  cudaChannelFormatDesc channelDesc_w = cudaCreateChannelDesc<int2>();

  // set up texture
  dtex_u.filterMode = cudaFilterModePoint;
  dtex_u.normalized = false;
  dtex_u.channelDesc = channelDesc_u;	

  dtex_v.filterMode = cudaFilterModePoint;
  dtex_v.normalized = false;
  dtex_v.channelDesc = channelDesc_v;	

  dtex_w.filterMode = cudaFilterModePoint;
  dtex_w.normalized = false;
  dtex_w.channelDesc = channelDesc_w;	

  if (cudaBindTexture(NULL, &dtex_u, u->buffer(), &channelDesc_u, u->num_allocated_elements() * sizeof(double)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_SelfAdvection3DDevice<double>::bind_textures - Could not bind texture u\n");
    return false;
  }
  
  if (cudaBindTexture(NULL, &dtex_v, v->buffer(), &channelDesc_v, v->num_allocated_elements() * sizeof(double)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_SelfAdvection3DDevice<double>::bind_textures - Could not bind texture v\n");
    return false;
  }

  if (cudaBindTexture(NULL, &dtex_w, w->buffer(), &channelDesc_w, w->num_allocated_elements() * sizeof(double)) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Sol_SelfAdvection3DDevice<double>::bind_textures - Could not bind texture w\n");
    return false;
  }

  return true;
}


template<>
bool
Sol_SelfAdvection3DDevice<double>::unbind_textures()
{
  cudaUnbindTexture(&dtex_u);
  cudaUnbindTexture(&dtex_v);
  cudaUnbindTexture(&dtex_w);
  return true;
}

template<>
bool 
Sol_SelfAdvection3DDevice<double>::solve_tex()
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
  if (interp_type == IT_FIRST_ORDER_UPWIND) {
    Advection3DD_apply_upwind_TEX<<<Dg, Db>>>(&deriv_udt->at(0,0,0),&deriv_vdt->at(0,0,0),&deriv_wdt->at(0,0,0),
      (double)(1/_hx), (double)(1/_hy), (double)(1/_hz), u->xstride(), u->ystride(), u->shift_amount(),
      _nx, _ny, _nz, blocksInY, 1.0f / (float)blocksInY, InterpolatorFirstOrderUpwind<double>());
  }
  else if (interp_type == IT_SECOND_ORDER_CENTERED) {
    Advection3DD_apply_upwind_TEX<<<Dg, Db>>>(&deriv_udt->at(0,0,0),&deriv_vdt->at(0,0,0),&deriv_wdt->at(0,0,0),
      (double)(1/_hx), (double)(1/_hy), (double)(1/_hz), u->xstride(), u->ystride(), u->shift_amount(),
      _nx, _ny, _nz, blocksInY, 1.0f / (float)blocksInY, InterpolatorSecondOrderCentered<double>());
  }
  else {
    printf("[ERROR] Sol_SelfAdvection3DDevice::solve_tex - invalid interpolation type %d\n", interp_type);
    return false;
  }

  return PostKernel("Advection3DD_apply_upwind_TEX");
}
#endif // OCU_DOUBLESUPPORT

template<typename T>
bool
Sol_SelfAdvection3DDevice<T>::solve()
{
  //return solve_naive();
  return solve_tex();
}

template<typename T>
bool 
Sol_SelfAdvection3DDevice<T>::initialize_storage(int nx, int ny, int nz, double hx, double hy, double hz, Grid3DDevice<T> *u_val, Grid3DDevice<T> *v_val, Grid3DDevice<T> *w_val,
                                             Grid3DDevice<T> *deriv_udt_val, Grid3DDevice<T> *deriv_vdt_val, Grid3DDevice<T> *deriv_wdt_val)
{
  // u,v,w must be the proper dimensions, i.e. staggered grid
  if (u_val->nx() != nx+1 || u_val->ny() != ny || u_val->nz() != nz) {
    printf("[ERROR] Sol_SelfAdvection3DDevice::initialize_storage - u dimensions mismatch\n");
    return false;
  }

  if (v_val->nx() != nx || v_val->ny() != ny+1 || v_val->nz() != nz) {
    printf("[ERROR] Sol_SelfAdvection3DDevice::initialize_storage - v dimensions mismatch\n");
    return false;
  }

  if (w_val->nx() != nx || w_val->ny() != ny || w_val->nz() != nz+1) {
    printf("[ERROR] Sol_SelfAdvection3DDevice::initialize_storage - v dimensions mismatch\n");
    return false;
  }

  // u,v,w must all share the same memory layout.  This is a cuda optimization to simplify indexing.
  if (!u_val->check_layout_match(*v_val) || !u_val->check_layout_match(*w_val)) {
    printf("[ERROR] Sol_SelfAdvection3DDevice::initialize_storage - u,v,w layout mismatch\n");
    return false;
  }

  if (u_val->gx() < 1 || u_val->gy() < 1 || u_val->gz() < 1) {
    printf("[ERROR] Sol_SelfAdvection3DDevice::initialize_storage - u has no ghost cells \n");
    return false;
  }

  if (v_val->gx() < 1 || v_val->gy() < 1 || v_val->gz() < 1) {
    printf("[ERROR] Sol_SelfAdvection3DDevice::initialize_storage - v has no ghost cells \n");
    return false;
  }

  if (w_val->gx() < 1 || w_val->gy() < 1 || w_val->gz() < 1) {
    printf("[ERROR] Sol_SelfAdvection3DDevice::initialize_storage - w has no ghost cells \n");
    return false;
  }

  u = u_val;
  v = v_val;
  w = w_val;
  deriv_udt = deriv_udt_val;
  deriv_vdt = deriv_vdt_val;
  deriv_wdt = deriv_wdt_val;


  if (!check_float(hx) || !check_float(hy) || !check_float(hz)) {
    printf("[ERROR] Sol_SelfAdvection3DDevice::initialize_storage - garbage hx,hy,hz value\n");
    return false;
  }


  _hx = hx;
  _hy = hy;
  _hz = hz;

  _nx = nx;
  _ny = ny;
  _nz = nz;


  if (!u->check_layout_match(*deriv_udt) || !v->check_layout_match(*deriv_vdt) || !w->check_layout_match(*deriv_wdt)) {
    printf("[ERROR] Sol_SelfAdvection3DDevice::initialize_storage - derivative layout error\n");
    return false;
  }

  if (!bind_textures()) {
    printf("[ERROR] Sol_SelfAdvection3DDevice::initialize_storage - failed on texture binding\n");
    return false;
  }

  return true;
}


template class Sol_SelfAdvection3DDevice<float>;

#ifdef OCU_DOUBLESUPPORT
template class Sol_SelfAdvection3DDevice<double>;
#endif //OCU_DOUBLESUPPORT

} // end namespace

