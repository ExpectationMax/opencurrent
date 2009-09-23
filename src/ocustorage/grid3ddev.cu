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
#include "ocustorage/grid3d.h"
#include "ocustorage/grid3dops.h"
#include "ocuutil/timer.h"
#include "ocuutil/timing_pool.h"

template<typename T, typename S>
__global__ void Grid3DDevice_copy_all_data(T *to, const S*from, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < n) {
    to[i] = (T)from[i];  
  }
}

template<typename T>
__global__ void Grid3DDevice_linear_combination2_3D(T *result, T alpha1, const T *g1, T alpha2, const T *g2,
  int xstride, int ystride, 
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
    result[idx] = g1[idx] * alpha1 + g2[idx] * alpha2;
  }
}

template<typename T>
__global__ void Grid3DDevice_linear_combination1(T *result, T alpha1, const T *g1, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < n) {
    result[i] = g1[i] * alpha1;  
  }
}

template<typename T>
__global__ void Grid3DDevice_linear_combination2(T *result, T alpha1, const T *g1, T alpha2, const T *g2, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < n) {
    result[i] = g1[i] * alpha1 + g2[i] * alpha2;  
  }
}

template<typename T>
__global__ void Grid3DDevice_clear(T *grid, T val, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < n) {
    grid[i] = val;  
  }
}




namespace ocu {


template<>
bool Grid3DDevice<int>::reduce_max(int &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxI());
}

template<>
bool Grid3DDevice<float>::reduce_max(float &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxF());
}


template<>
bool Grid3DDevice<int>::reduce_min(int &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMinI());
}

template<>
bool Grid3DDevice<float>::reduce_min(float &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMinF());
}

template<>
bool Grid3DDevice<int>::reduce_maxabs(int &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxAbsI());
}

template<>
bool Grid3DDevice<float>::reduce_maxabs(float &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxAbsF());
}

#ifdef OCU_DOUBLESUPPORT
template<>
bool Grid3DDevice<double>::reduce_maxabs(double &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxAbsD());
}

template<>
bool Grid3DDevice<double>::reduce_max(double &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxD());
}

template<>
bool Grid3DDevice<double>::reduce_min(double &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMinD());
}

#endif // OCU_DOUBLESUPPORT

template<typename T>
bool Grid3DDevice<T>::reduce_sum(T &result) const
{
  return reduce_with_operator(*this, result, ReduceDevSum<T>());
}

template<typename T>
bool Grid3DDevice<T>::reduce_sqrsum(T &result) const
{
  return reduce_with_operator(*this, result, ReduceDevSqrSum<T>());
}

template<typename T>
bool Grid3DDevice<T>::reduce_checknan(T &result) const
{
  return reduce_with_operator(*this, result, ReduceDevCheckNan<T>());
}


template<typename T>
bool Grid3DDevice<T>::clear_zero()
{
  GPUTimer timer;
  timer.start();
  if ((unsigned int)CUDA_SUCCESS != cudaMemset(this->_buffer, 0, this->num_allocated_elements() * sizeof(T))) {
    printf("[ERROR] Grid3DDeviceT::clear_zero - cudaMemset failed\n");
    return false;
  }
  timer.stop();
  global_timer_add_timing("cudaMemset", timer.elapsed_ms());

  return true;
}

template<typename T>
bool Grid3DDevice<T>::clear(T val)
{
  dim3 Dg((this->num_allocated_elements()+255) / 256);
  dim3 Db(256);
  
  GPUTimer timer;
  timer.start();
  Grid3DDevice_clear<<<Dg, Db>>>(this->_buffer, val, this->num_allocated_elements());
  timer.stop();
  global_timer_add_timing("Grid3DDevice_clear", timer.elapsed_ms());

  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] Grid3DDeviceF::clear - CUDA error \"%s\"\n", cudaGetErrorString(er));
    return false;    
  }
  
  return true;
}


template<typename T>
bool 
Grid3DDevice<T>::copy_all_data(const Grid3DHost<T> &from)
{
  if (!this->check_layout_match(from)) {
    printf("[ERROR] Grid3DDevice::copy_all_data - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(),this->pny(),this->pnz(), from.pnx(), from.pny(),from.pnz());
    return false;
  }

  GPUTimer timer;
  timer.start();
  if ((unsigned int)CUDA_SUCCESS != cudaMemcpy(this->buffer(), from.buffer(), sizeof(T) * this->num_allocated_elements(), cudaMemcpyHostToDevice)) {
    printf("[ERROR] Grid3DDevice::copy_all_data - cudaMemcpy failed\n");
    return false;
  }
  timer.stop();
  global_timer_add_timing("cudaMemcpy(HostToDevice)", timer.elapsed_ms());
  
  return true;
}



template<typename T>
template<typename S>
bool 
Grid3DDevice<T>::copy_all_data(const Grid3DDevice<S> &from)
{
  if (!this->check_layout_match(from)) {
    printf("[ERROR] Grid3DDevice::copy_all_data - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(),this->pny(),this->pnz(), from.pnx(), from.pny(),from.pnz());
    return false;
  }

  dim3 Dg((this->num_allocated_elements()+511) / 512);
  dim3 Db(512);
  
  GPUTimer timer;
  timer.start();
  Grid3DDevice_copy_all_data<<<Dg, Db>>>(this->buffer(), from.buffer(), this->num_allocated_elements());
  timer.stop();
  global_timer_add_timing("Grid3DDevice_copy_all_data", timer.elapsed_ms());

  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] Grid3DDevice::copy_all_data - CUDA error \"%s\"\n", cudaGetErrorString(er));
    return false;    
  }

  return true;
}

template<>
template<>
bool 
Grid3DDevice<float>::copy_all_data(const Grid3DDevice<float> &from)
{
  if (!this->check_layout_match(from)) {
    printf("[ERROR] Grid3DDevice::copy_all_data - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(),this->pny(),this->pnz(), from.pnx(), from.pny(),from.pnz());
    return false;
  }

  GPUTimer timer;
  timer.start();
  if ((unsigned int)CUDA_SUCCESS != cudaMemcpy(this->buffer(), from.buffer(), sizeof(float) * this->num_allocated_elements(), cudaMemcpyDeviceToDevice)) {
    printf("[ERROR] Grid3DDevice::copy_all_data - cudaMemcpy failed\n");
    return false;
  }
  timer.stop();
  global_timer_add_timing("cudaMemcpy(DeviceToDevice)", timer.elapsed_ms());

  return true;
}

template<>
template<>
bool 
Grid3DDevice<int>::copy_all_data(const Grid3DDevice<int> &from)
{
  if (!this->check_layout_match(from)) {
    printf("[ERROR] Grid3DDevice::copy_all_data - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(),this->pny(),this->pnz(), from.pnx(), from.pny(),from.pnz());
    return false;
  }

  GPUTimer timer;
  timer.start();
  if ((unsigned int)CUDA_SUCCESS != cudaMemcpy(this->buffer(), from.buffer(), sizeof(int) * this->num_allocated_elements(), cudaMemcpyDeviceToDevice)) {
    printf("[ERROR] Grid3DDevice::copy_all_data - cudaMemcpy failed\n");
    return false;
  }
  timer.stop();
  global_timer_add_timing("cudaMemcpy(DeviceToDevice)", timer.elapsed_ms());

  return true;
}

template<>
template<>
bool 
Grid3DDevice<double>::copy_all_data(const Grid3DDevice<double> &from)
{
  if (!this->check_layout_match(from)) {
    printf("[ERROR] Grid3DDevice::copy_all_data - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(),this->pny(),this->pnz(), from.pnx(), from.pny(),from.pnz());
    return false;
  }

  GPUTimer timer;
  timer.start();
  if ((unsigned int)CUDA_SUCCESS != cudaMemcpy(this->buffer(), from.buffer(), sizeof(double) * this->num_allocated_elements(), cudaMemcpyDeviceToDevice)) {
    printf("[ERROR] Grid3DDevice::copy_all_data - cudaMemcpy failed\n");
    return false;
  }
  timer.stop();
  global_timer_add_timing("cudaMemcpy(DeviceToDevice)", timer.elapsed_ms());

  return true;
}

template<typename T>
bool Grid3DDevice<T>::linear_combination(T alpha1, const Grid3DDevice<T> &g1)
{
  if (!this->check_layout_match(g1)) {
    printf("[ERROR] Grid3DDevice::linear_combination - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(),this->pny(),this->pnz(), g1.pnx(), g1.pny(), g1.pnz());
    return false;
  }

  dim3 Dg((this->num_allocated_elements()+511) / 512);
  dim3 Db(512);
  
  GPUTimer timer;
  timer.start();
  Grid3DDevice_linear_combination1<<<Dg, Db>>>(this->buffer(), alpha1, g1.buffer(), this->num_allocated_elements());
  timer.stop();
  global_timer_add_timing("Grid3DDevice_linear_combination1", timer.elapsed_ms());


  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] Grid3DDeviceF::linear_combination - CUDA error \"%s\"\n", cudaGetErrorString(er));
    return false;    
  }
  
  return true;
}

template<typename T>
bool Grid3DDevice<T>::linear_combination(T alpha1, const Grid3DDevice<T> &g1, T alpha2, const Grid3DDevice<T> &g2)
{
  if (!this->check_layout_match(g1)) {
    printf("[ERROR] Grid3DDevice::linear_combination - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(),this->pny(),this->pnz(), g1.pnx(), g1.pny(), g1.pnz());
    return false;
  }

  if (!this->check_layout_match(g2)) {
    printf("[ERROR] Grid3DDevice::linear_combination - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(),this->pny(),this->pnz(), g2.pnx(), g2.pny(), g2.pnz());
    return false;
  }

  // Calculate how many elements in the linear array are actually out of bounds for the 3d array, i.e., wasted elements.
  // If there are too many of them (>3%), then we will use the 3d version.  Otherwise we will use the faster 1d version.
  if ((this->num_allocated_elements() - (this->nx() * this->ny() * this->nz())) / ((float) this->num_allocated_elements()) > .03f) {
    int tnx = this->nz();
    int tny = this->ny();
    int tnz = this->nx();

    int threadsInX = 16;
    int threadsInY = 2;
    int threadsInZ = 2;

    int blocksInX = (tnx+threadsInX-1)/threadsInX;
    int blocksInY = (tny+threadsInY-1)/threadsInY;
    int blocksInZ = (tnz+threadsInZ-1)/threadsInZ;

    dim3 Dg = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);

    GPUTimer timer;
    timer.start();
    Grid3DDevice_linear_combination2_3D<<<Dg, Db>>>(&this->at(0,0,0), alpha1, &g1.at(0,0,0), alpha2, &g2.at(0,0,0), 
      this->xstride(), this->ystride(), this->nx(), this->ny(), this->nz(), 
      blocksInY, 1.0f / (float)blocksInY);
    timer.stop();
    global_timer_add_timing("Grid3DDevice_linear_combination2_3D", timer.elapsed_ms());


  }
  else {
    int block_size = 512;
    dim3 Dg((this->num_allocated_elements() + block_size - 1) / block_size);
    dim3 Db(block_size);
    
    GPUTimer timer;
    timer.start();
    Grid3DDevice_linear_combination2<<<Dg, Db>>>(this->buffer(), alpha1, g1.buffer(), alpha2, g2.buffer(), this->num_allocated_elements());
    timer.stop();
    global_timer_add_timing("Grid3DDevice_linear_combination2", timer.elapsed_ms());
  }

  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] Grid3DDeviceF::linear_combination - CUDA error \"%s\"\n", cudaGetErrorString(er));
    return false;    
  }  

  return true;
}


template<typename T>
Grid3DDevice<T>::~Grid3DDevice()
{
  cudaFree(this->_buffer);
}

template<typename T>
bool Grid3DDevice<T>::init(int nx_val, int ny_val, int nz_val, int gx_val, int gy_val, int gz_val, int padx, int pady, int padz)
{
  this->_nx = nx_val;
  this->_ny = ny_val;
  this->_nz = nz_val;

  this->_gx = gx_val;
  this->_gy = gy_val;
  this->_gz = gz_val;
  
  // pad with ghost cells & user-specified padding
  this->_pnx = this->_nx + 2 * gx_val + padx;
  this->_pny = this->_ny + 2 * gy_val + pady;
  this->_pnz = this->_nz + 2 * gz_val + padz;

  // either shift by 4 bits (float and int) or 3 bits (double)
  // then the mask is either 15 or 7.
  //int shift_amount = (sizeof(T) == 4 ? 4 : 3);
  int shift_amount = 4;

  int mask = (0x1 << shift_amount) - 1;

  // round up pnz to next multiple of 16 if needed
  if (this->_pnz & mask)
    this->_pnz = ((this->_pnz >> shift_amount) + 1) << shift_amount;

  // calculate pre-padding to get k=0 elements start at a coalescing boundary.
  int pre_padding =  (16 - this->_gz); 

  this->_pnzpny = this->_pnz * this->_pny;
  this->_allocated_elements = this->_pnzpny * this->_pnx + pre_padding;

  if ((unsigned int)CUDA_SUCCESS != cudaMalloc((void **)&this->_buffer, sizeof(T) * this->num_allocated_elements())) {
    printf("[ERROR] Grid3DDeviceF::init - cudaMalloc failed\n");
    return false;
  }
  
  this->_shift_amount   = this->_gx * this->_pnzpny + this->_gy * this->_pnz + this->_gz + pre_padding; 
  this->_shifted_buffer = this->_buffer + this->_shift_amount;

  return true;
}


template bool Grid3DDevice<float>::init(int nx_val, int ny_val, int nz_val, int gx_val, int gy_val, int gz_val, int padx, int pady, int padz);
template bool Grid3DDevice<int>::init(int nx_val, int ny_val, int nz_val, int gx_val, int gy_val, int gz_val, int padx, int pady, int padz);
template bool Grid3DDevice<double>::init(int nx_val, int ny_val, int nz_val, int gx_val, int gy_val, int gz_val, int padx, int pady, int padz);

template Grid3DDevice<float>::~Grid3DDevice();
template Grid3DDevice<int>::~Grid3DDevice();
template Grid3DDevice<double>::~Grid3DDevice();

template bool Grid3DDevice<float>::reduce_sqrsum(float &result) const;
template bool Grid3DDevice<int>::reduce_sqrsum(int &result) const;

template bool Grid3DDevice<float>::reduce_sum(float &result) const;
template bool Grid3DDevice<int>::reduce_sum(int &result) const;

template bool Grid3DDevice<float>::reduce_checknan(float &result) const;

template bool Grid3DDevice<float>::copy_all_data(const Grid3DHost<float> &from);
template bool Grid3DDevice<int>::copy_all_data(const Grid3DHost<int> &from);
template bool Grid3DDevice<double>::copy_all_data(const Grid3DHost<double> &from);

template bool Grid3DDevice<float>::copy_all_data(const Grid3DDevice<int> &from);
template bool Grid3DDevice<float>::copy_all_data(const Grid3DDevice<double> &from);

template bool Grid3DDevice<int>::copy_all_data(const Grid3DDevice<float> &from);
template bool Grid3DDevice<int>::copy_all_data(const Grid3DDevice<double> &from);

template bool Grid3DDevice<float>::clear_zero();
template bool Grid3DDevice<int>::clear_zero();
template bool Grid3DDevice<double>::clear_zero();

template bool Grid3DDevice<float>::clear(float val);
template bool Grid3DDevice<int>::clear(int val);

template bool Grid3DDevice<float>::linear_combination(float alpha1, const Grid3DDevice<float> &g1);
template bool Grid3DDevice<float>::linear_combination(float alpha1, const Grid3DDevice<float> &g1, float alpha2, const Grid3DDevice<float> &g2);

#ifdef OCU_DOUBLESUPPORT

template bool Grid3DDevice<double>::reduce_sqrsum(double &result) const;
template bool Grid3DDevice<double>::reduce_sum(double &result) const;
template bool Grid3DDevice<double>::reduce_checknan(double &result) const;
template bool Grid3DDevice<double>::clear(double val);
template bool Grid3DDevice<double>::linear_combination(double  alpha1, const Grid3DDevice<double> &g1);
template bool Grid3DDevice<double>::linear_combination(double alpha1, const Grid3DDevice<double> &g1, double  alpha2, const Grid3DDevice<double> &g2);
template bool Grid3DDevice<double>::copy_all_data(const Grid3DDevice<float> &from);
template bool Grid3DDevice<double>::copy_all_data(const Grid3DDevice<int> &from);

#endif // OCU_DOUBLESUPPORT

} // end namespace

