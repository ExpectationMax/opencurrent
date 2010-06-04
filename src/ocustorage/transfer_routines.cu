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

#include "ocustorage/coarray_internal.h"

template<typename T>
__global__ void kernel_region3d_to_region3d(
  T *to, const T *from,   
  int to_xstride, int to_ystride, 
  int from_xstride, int from_ystride, 
  int nx, int ny, int nz, 
  int blocksInY, float invBlocksInY)
{
  int blockIdxz = truncf(blockIdx.y * invBlocksInY);
  int blockIdxy = blockIdx.y - __mul24(blockIdxz,blocksInY);

  // transpose for coalescing since k is the fastest changing index 
  int k     = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int j     = __mul24(blockIdxy ,blockDim.y) + threadIdx.y;
  int i     = __mul24(blockIdxz ,blockDim.z) + threadIdx.z;

  // shift so we will get maximum coalescing?
  int from_idx = i * from_xstride + j * from_ystride + k;
  int to_idx   = i * to_xstride   + j * to_ystride   + k;

  if (i < nx && j < ny && k < nz) {
    T val = from[from_idx];
    to[to_idx] = val;
  }
}

template<typename T>
__global__ void kernel_pack_region3d(
  T *to, const T *from,   
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

  // shift so we will get maximum coalescing?
  int from_idx = i * xstride + j * ystride + k;

  if (i < nx && j < ny && k < nz) {
    T val = from[from_idx];
    int to_idx = i * nz * ny + j * nz + k;
    to[to_idx] = val;
  }
}

template<typename T>
__global__ void kernel_unpack_region3d(
  T *to, const T *from,   
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

  // shift so we will get maximum coalescing?
  int from_idx = i * nz * ny + j * nz + k;

  if (i < nx && j < ny && k < nz) {
    T val = from[from_idx];
    int to_idx = i * xstride + j * ystride + k;
    to[to_idx] = val;
  }
}

namespace ocu {


void remote_xfer_hregion1d_to_dregion1d(int image_id, void *d_dst, const void *h_src, size_t num_bytes)
{
  TransferRequest1D req;
  req.dst = d_dst;
  req.src = h_src;
  req.num_bytes = num_bytes;
  req.cmd = TRANSFER_HOST_TO_DEVICE;
  CoArrayManager::xfer_q(image_id)->push(req);
}

void remote_xfer_dregion1d_to_hregion1d(int image_id, void *h_dst, const void *d_src, size_t num_bytes)
{
  TransferRequest1D req;
  req.dst = h_dst;
  req.src = d_src;
  req.num_bytes = num_bytes;
  req.cmd = TRANSFER_DEVICE_TO_HOST;
  CoArrayManager::xfer_q(image_id)->push(req);
}

void remote_xfer_hbuffer_to_dregion3d(int image_id, const Region3D &dst, void *host_buffer, void *device_buffer, TransferBufferMethod tbm)
{
  TransferRequest3D req;
  req.dst = dst;
  req.host_buffer = host_buffer;
  req.device_buffer = device_buffer;
  req.cmd = TRANSFER_HOSTBUFFER_TO_DEVICE;
  req.method = tbm;
  CoArrayManager::xfer_q(image_id)->push(req);
}

void remote_xfer_dregion3d_to_hbuffer(int image_id, void *host_buffer, void *device_buffer, const ConstRegion3D &src, TransferBufferMethod tbm)
{
  TransferRequest3D req;
  req.src = src;
  req.host_buffer = host_buffer;
  req.device_buffer = device_buffer;
  req.cmd = TRANSFER_DEVICE_TO_HOSTBUFFER;
  req.method = tbm;
  CoArrayManager::xfer_q(image_id)->push(req);
}

bool xfer_hregion1d_to_dregion1d(void *d_dst, const void *h_src, size_t num_bytes)
{
  cudaError_t ok = cudaMemcpyAsync(d_dst, h_src, num_bytes, cudaMemcpyHostToDevice, ThreadManager::get_io_stream());
  if (ok != cudaSuccess) {
    printf("[ERROR] xfer_hregion1d_to_dregion1d - cudaMemcpyAsync(HtoD) failed: %s\n", cudaGetErrorString(ok));
    return false;
  }
  return true;

}

bool xfer_dregion1d_to_hregion1d(void *h_dst, const void *d_src, size_t num_bytes)
{
  cudaError_t ok = cudaMemcpyAsync(h_dst, d_src, num_bytes, cudaMemcpyDeviceToHost, ThreadManager::get_io_stream());
  if (ok != cudaSuccess) {
    printf("[ERROR] xfer_dregion1d_to_hregion1d - cudaMemcpyAsync(DtoH) failed: %s\n", cudaGetErrorString(ok));
    return false;
  }
  return true;
}


bool xfer_hbuffer_to_dregion3d(const Region3D &dst, void *host_buffer, void *device_buffer, TransferBufferMethod tbm)
{
  if (tbm == TBM_CONTIGUOUS) {
    cudaMemcpyAsync(
      dst.grid()->ptr_untyped(dst.x0,dst.y0,dst.z0), 
      host_buffer, 
      contiguous_distance(dst), cudaMemcpyHostToDevice, ThreadManager::get_io_stream());
    return true;
  }
  else if (tbm == TBM_PACKED) {

    cudaMemcpyAsync(device_buffer, host_buffer, dst.volume() * dst.grid()->atom_size(), cudaMemcpyHostToDevice, ThreadManager::get_io_stream());

    int extentx = dst.x1 - dst.x0+1;
    int extenty = dst.y1 - dst.y0+1;
    int extentz = dst.z1 - dst.z0+1;

    int tnx = extentz;
    int tny = extenty;
    int tnz = extentx;

    int threadsInX = 16;
    int threadsInY = 4;
    int threadsInZ = 2;

    int blocksInX = (tnx+threadsInX-1)/threadsInX;
    int blocksInY = (tny+threadsInY-1)/threadsInY;
    int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

    dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

    // pack src into a linear buffer in a kernel
    if (dst.grid()->atom_size() == 4) {
      kernel_unpack_region3d<<<Dg, Db, 0, ThreadManager::get_io_stream()>>>(
        (float *) dst.grid()->ptr_untyped(dst.x0, dst.y0, dst.z0), (const float *) device_buffer, 
        dst.grid()->xstride(), dst.grid()->ystride(), extentx, extenty, extentz, 
        blocksInY, 1.0f / (float)blocksInY);
    }
    else if (dst.grid()->atom_size() == 8) {
      kernel_unpack_region3d<<<Dg, Db, 0, ThreadManager::get_io_stream()>>>(
        (float2 *) dst.grid()->ptr_untyped(dst.x0, dst.y0, dst.z0), (const float2 *) device_buffer, 
        dst.grid()->xstride(), dst.grid()->ystride(), extentx, extenty, extentz, 
        blocksInY, 1.0f / (float)blocksInY);
    }
    else{
      printf("[ERROR] xfer_hbuffer_to_dregion3d - unsupported atom_size = %u\n", (unsigned int) dst.grid()->atom_size());
      return false;
    }

    return true;
  }
  else {
    printf("[ERROR] xfer_hbuffer_to_dregion3d - Invalid method %d\n", tbm);
    return false;
  }
}

bool xfer_dregion3d_to_hbuffer(void *host_buffer, void *device_buffer, const ConstRegion3D &src, TransferBufferMethod tbm)
{
  if (tbm == TBM_CONTIGUOUS) {
    cudaMemcpyAsync(
      host_buffer, 
      src.grid()->ptr_untyped(src.x0,src.y0,src.z0), 
      contiguous_distance(src), cudaMemcpyDeviceToHost, ThreadManager::get_io_stream());
    return true;
  }
  else if (tbm == TBM_PACKED) {

    int extentx = src.x1 - src.x0+1;
    int extenty = src.y1 - src.y0+1;
    int extentz = src.z1 - src.z0+1;

    int tnx = extentz;
    int tny = extenty;
    int tnz = extentx;

    int threadsInX = 16;
    int threadsInY = 4;
    int threadsInZ = 2;

    int blocksInX = (tnx+threadsInX-1)/threadsInX;
    int blocksInY = (tny+threadsInY-1)/threadsInY;
    int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

    dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

    // pack src into a linear buffer in a kernel
    if (src.grid()->atom_size() == 4) {
      kernel_pack_region3d<<<Dg, Db, 0, ThreadManager::get_io_stream()>>>(
        (float *)device_buffer, (const float *)src.grid()->ptr_untyped(src.x0, src.y0, src.z0), 
        src.grid()->xstride(), src.grid()->ystride(), extentx, extenty, extentz, 
        blocksInY, 1.0f / (float)blocksInY);
    }
    else if (src.grid()->atom_size() == 8) {
      kernel_pack_region3d<<<Dg, Db, 0, ThreadManager::get_io_stream()>>>(
        (float2 *)device_buffer, (const float2 *)src.grid()->ptr_untyped(src.x0, src.y0, src.z0), 
        src.grid()->xstride(), src.grid()->ystride(), extentx, extenty, extentz, 
        blocksInY, 1.0f / (float)blocksInY);
    }
    else{
      printf("[ERROR] xfer_dregion3d_to_hbuffer - unsupported atom_size = %u\n", (unsigned int) src.grid()->atom_size());
      return false;
    }

    // copy linear buffer to host
    cudaMemcpyAsync(host_buffer, device_buffer, src.volume() * src.grid()->atom_size(), cudaMemcpyDeviceToHost, ThreadManager::get_io_stream());
  }
  else {
    printf("[ERROR] xfer_dregion3d_to_hbuffer - Invalid method %d\n", tbm);
    return false;
  }

  return true;
}

bool 
xfer_hregion1d_to_hregion1d(void *h_dst, const void *h_src, size_t num_bytes)
{
  memcpy(h_dst, h_src, num_bytes);
  return true;
}

bool xfer_dregion1d_to_dregion1d(void *d_dst, const void *d_src, size_t num_bytes)
{
  cudaMemcpyAsync(d_dst, d_src, num_bytes, cudaMemcpyDeviceToDevice, ThreadManager::get_io_stream());
  return true;
}

bool xfer_dregion3d_to_dregion3d(const Region3D &d_dst, const ConstRegion3D &d_src, TransferBufferMethod tbm)
{
  if (tbm == TBM_CONTIGUOUS) {
    cudaMemcpyAsync(
      d_dst.grid()->ptr_untyped(d_dst.x0,d_dst.y0,d_dst.z0), 
      d_src.grid()->ptr_untyped(d_src.x0,d_src.y0,d_src.z0), 
      contiguous_distance(d_dst), cudaMemcpyDeviceToDevice, ThreadManager::get_io_stream());
    return true;
  }
  else if (tbm == TBM_PACKED) {

    int extentx = d_dst.x1 - d_dst.x0+1;
    int extenty = d_dst.y1 - d_dst.y0+1;
    int extentz = d_dst.z1 - d_dst.z0+1;

    int tnx = extentz;
    int tny = extenty;
    int tnz = extentx;

    int threadsInX = 16;
    int threadsInY = 4;
    int threadsInZ = 2;

    int blocksInX = (tnx+threadsInX-1)/threadsInX;
    int blocksInY = (tny+threadsInY-1)/threadsInY;
    int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

    dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

    // pack src into a linear buffer in a kernel
    if (d_dst.grid()->atom_size() == 4) {
      kernel_region3d_to_region3d<<<Dg, Db, 0, ThreadManager::get_io_stream()>>>(
        (float*)       d_dst.grid()->ptr_untyped(d_dst.x0, d_dst.y0, d_dst.z0),  
        (const float*) d_src.grid()->ptr_untyped(d_src.x0, d_src.y0, d_src.z0),  
        d_dst.grid()->xstride(), d_dst.grid()->ystride(), 
        d_src.grid()->xstride(), d_src.grid()->ystride(), 
        extentx, extenty, extentz, 
        blocksInY, 1.0f / (float)blocksInY);
    }
    else if (d_dst.grid()->atom_size() == 8) {
      kernel_region3d_to_region3d<<<Dg, Db, 0, ThreadManager::get_io_stream()>>>(
        (float2 *)       d_dst.grid()->ptr_untyped(d_dst.x0, d_dst.y0, d_dst.z0),  
        (const float2 *) d_src.grid()->ptr_untyped(d_src.x0, d_src.y0, d_src.z0),  
        d_dst.grid()->xstride(), d_dst.grid()->ystride(), 
        d_src.grid()->xstride(), d_src.grid()->ystride(), 
        extentx, extenty, extentz, 
        blocksInY, 1.0f / (float)blocksInY);
    }
    else{
      printf("[ERROR] xfer_dregion3d_to_dregion3d - unsupported atom_size = %u\n", (unsigned int) d_dst.grid()->atom_size());
      return false;
    }

    return true;
  }
  else {
    printf("[ERROR] xfer_dregion3d_to_dregion3d - Invalid method %d\n", tbm);
    return false;
  }
}

bool xfer_hregion3d_to_hregion3d(const Region3D &h_dst, const ConstRegion3D &h_src, TransferBufferMethod tbm)
{
  if (tbm == TBM_CONTIGUOUS) {
    memcpy(h_dst.grid()->ptr_untyped(h_dst.x0,h_dst.y0,h_dst.z0), h_src.grid()->ptr_untyped(h_src.x0,h_src.y0,h_src.z0), contiguous_distance(h_dst));
    return true;
  }
  else if (tbm == TBM_PACKED) {

    const char *src_ptr= (const char *)h_src.grid()->ptr_untyped(h_src.x0,h_src.y0,h_src.z0);
    char *dst_ptr      = (char *)      h_dst.grid()->ptr_untyped(h_dst.x0,h_dst.y0,h_dst.z0);

    int src_ywrap_jump = (const char *)h_src.grid()->ptr_untyped(1,h_src.y0,h_src.z0) - (const char *)h_src.grid()->ptr_untyped(0,h_src.y1+1,h_src.z0  );
    int dst_ywrap_jump = (char *)      h_dst.grid()->ptr_untyped(1,h_dst.y0,h_dst.z0) - (char *)      h_dst.grid()->ptr_untyped(0,h_dst.y1+1,h_dst.z0  );
    int src_ystride    = (const char *)h_src.grid()->ptr_untyped(0,1,0)               - (const char *)h_src.grid()->ptr_untyped(0,0,0);
    int dst_ystride    = (char *)      h_dst.grid()->ptr_untyped(0,1,0)               - (char *)      h_dst.grid()->ptr_untyped(0,0,0);

    int extentx = h_src.x1 - h_src.x0+1;
    int extenty = h_src.y1 - h_src.y0+1;

    size_t span_length =(h_src.z1 - h_src.z0 + 1);
    size_t span_bytes = h_src.grid()->atom_size() * span_length;

    int x=0,y=0;

    while (true) {
      
      // do copy
      memcpy(dst_ptr, src_ptr, span_bytes);
      src_ptr += src_ystride;
      dst_ptr += dst_ystride;
      
      y++;

      if (y == extenty) {
        src_ptr += src_ywrap_jump;
        dst_ptr += dst_ywrap_jump;
        y = 0;
        x++;

        if (x == extentx)
          break;
      }
    }

    return true;
  }
  else {
    printf("[ERROR] xfer_hregion3d_to_hregion3d - Invalid method %d\n", tbm);
    return false;
  }

}

bool xfer_hbuffer_to_hregion3d(const Region3D &h_dst, const void *h_src, TransferBufferMethod tbm)
{
  if (tbm == TBM_CONTIGUOUS) {
    memcpy(h_dst.grid()->ptr_untyped(h_dst.x0, h_dst.y0, h_dst.z0), h_src, contiguous_distance(h_dst));
    return true;
  }
  else if (tbm == TBM_PACKED) {

    char *dst_ptr      = (char *)h_dst.grid()->ptr_untyped(h_dst.x0,h_dst.y0,h_dst.z0);
    const char *buf_ptr= (char *)h_src;

    int dst_ywrap_jump = (char *)h_dst.grid()->ptr_untyped(1,h_dst.y0,h_dst.z0) - (char *)h_dst.grid()->ptr_untyped(0,h_dst.y1+1,h_dst.z0  );
    int dst_ystride    = (char *)h_dst.grid()->ptr_untyped(0,1,0)               - (char *)h_dst.grid()->ptr_untyped(0,0,0);

    int extentx = h_dst.x1 - h_dst.x0+1;
    int extenty = h_dst.y1 - h_dst.y0+1;
    int x=0,y=0;
    size_t span_length =(h_dst.z1 - h_dst.z0 + 1);
    size_t span_bytes = h_dst.grid()->atom_size() * span_length;

    while (true) {
      
      // do copy
      memcpy(dst_ptr, buf_ptr, span_bytes);
      dst_ptr += dst_ystride;
      buf_ptr += span_bytes;
      
      y++;

      if (y == extenty) {
        dst_ptr += dst_ywrap_jump;
        y = 0;
        x++;

        if (x == extentx)
          break;
      }
    }

    return true;
  }
  else {
    printf("[ERROR] xfer_hbuffer_to_hregion3d - Invalid method %d\n", tbm);
    return false;
  }
}



bool xfer_hregion3d_to_hbuffer(void *h_dst, const ConstRegion3D &h_src, TransferBufferMethod tbm)
{
  if (tbm == TBM_CONTIGUOUS) {
    memcpy(h_dst, h_src.grid()->ptr_untyped(h_src.x0, h_src.y0, h_src.z0), contiguous_distance(h_src));
    return true;
  }
  else if (tbm == TBM_PACKED) {

    const char *src_ptr= (const char *)h_src.grid()->ptr_untyped(h_src.x0,h_src.y0,h_src.z0);
    char *buf_ptr= (char *)h_dst;

    int src_ywrap_jump = (const char *)h_src.grid()->ptr_untyped(1,h_src.y0,h_src.z0) - (const char *)h_src.grid()->ptr_untyped(0,h_src.y1+1,h_src.z0  );
    int src_ystride    = (const char *)h_src.grid()->ptr_untyped(0,1,0)               - (const char *)h_src.grid()->ptr_untyped(0,0,0);

    int extentx = h_src.x1 - h_src.x0+1;
    int extenty = h_src.y1 - h_src.y0+1;
    int x=0,y=0;
    size_t span_length =(h_src.z1 - h_src.z0 + 1);
    size_t span_bytes = h_src.grid()->atom_size() * span_length;

    while (true) {
      
      // do copy
      memcpy(buf_ptr, src_ptr, span_bytes);
      src_ptr += src_ystride;
      buf_ptr += span_bytes;
      
      y++;

      if (y == extenty) {
        src_ptr += src_ywrap_jump;
        y = 0;
        x++;

        if (x == extentx)
          break;
      }
    }

    return true;
  }
  else {
    printf("[ERROR] xfer_hregion3d_to_hbuffer - Invalid method %d\n", tbm);
    return false;
  }
}





}
