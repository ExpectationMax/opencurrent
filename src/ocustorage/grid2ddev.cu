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

#include "ocustorage/grid2d.h"

template<typename T>
__global__ void Grid2DDevice_linear_combination1(T *result, T alpha1, const T *g1, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < n) {
    result[i] = g1[i] * alpha1;  
  }
}

template<typename T>
__global__ void Grid2DDevice_linear_combination2(T *result, T alpha1, const T *g1, T alpha2, const T *g2, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < n) {
    result[i] = g1[i] * alpha1 + g2[i] * alpha2;  
  }
}

template<typename T>
__global__ void Grid2DDevice_clear(T *grid, T val, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < n) {
    grid[i] = val;  
  }
}





namespace ocu {





template<typename T>
bool Grid2DDevice<T>::clear_zero()
{
  if ((unsigned int)CUDA_SUCCESS != cudaMemset(this->_buffer, 0, this->num_allocated_elements() * sizeof(T))) {
    printf("[ERROR] Grid2DDeviceT::clear_zero - cudaMemset failed\n");
    return false;
  }

  return true;
}

template<typename T>
bool Grid2DDevice<T>::clear(T val)
{
  dim3 Dg((this->num_allocated_elements()+255) / 256);
  dim3 Db(256);
  
  Grid2DDevice_clear<<<Dg, Db>>>(this->_buffer, val, this->num_allocated_elements());
  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] Grid2DDeviceF::clear - CUDA error \"%s\"\n", cudaGetErrorString(er));
    return false;    
  }
  
  return true;
}


template<typename T>
bool 
Grid2DDevice<T>::copy_all_data(const Grid2DHost<T> &from)
{
  if (!this->check_layout_match(from)) {
    printf("[ERROR] Grid2DDevice::copy_all_data - mismatch: (%d, %d) != (%d, %d)\n", this->pnx(),this->pny(), from.pnx(), from.pny());
    return false;
  }

  if ((unsigned int)CUDA_SUCCESS != cudaMemcpy(this->buffer(), from.buffer(), sizeof(T) * this->num_allocated_elements(), cudaMemcpyHostToDevice)) {
    printf("[ERROR] Grid2DDevice::copy_all_data - cudaMemcpy failed\n");
    return false;
  }
  
  return true;
}



template<typename T>
bool 
Grid2DDevice<T>::copy_all_data(const Grid2DDevice<T> &from)
{
  if (!this->check_layout_match(from)) {
    printf("[ERROR] Grid2DDevice::copy_all_data - mismatch: (%d, %d) != (%d, %d)\n", this->pnx(),this->pny(),from.pnx(), from.pny());
    return false;
  }

  if ((unsigned int)CUDA_SUCCESS != cudaMemcpy(this->buffer(), from.buffer(), sizeof(T) * this->num_allocated_elements(), cudaMemcpyDeviceToDevice)) {
    printf("[ERROR] Grid2DDevice::copy_all_data - cudaMemcpy failed\n");
    return false;
  }
  
  return true;
}



template<typename T>
bool Grid2DDevice<T>::linear_combination(T alpha1, const Grid2DDevice<T> &g1)
{
  if (!this->check_layout_match(g1)) {
    printf("[ERROR] Grid3DDevice::linear_combination - mismatch: (%d, %d) != (%d, %d)\n", this->pnx(),this->pny(),g1.pnx(), g1.pny());
    return false;
  }

  dim3 Dg((this->num_allocated_elements()+255) / 256);
  dim3 Db(256);
  
  Grid2DDevice_linear_combination1<<<Dg, Db>>>(this->buffer(), alpha1, g1.buffer(), this->num_allocated_elements());
  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] Grid2DDeviceF::linear_combination - CUDA error \"%s\"\n", cudaGetErrorString(er));
    return false;    
  }
  
  return true;
}

template<typename T>
bool Grid2DDevice<T>::linear_combination(T alpha1, const Grid2DDevice<T> &g1, T alpha2, const Grid2DDevice<T> &g2)
{
  if (!this->check_layout_match(g1)) {
    printf("[ERROR] Grid3DDevice::linear_combination - mismatch: (%d, %d) != (%d, %d)\n", this->pnx(),this->pny(), g1.pnx(), g1.pny());
    return false;
  }

  if (!this->check_layout_match(g2)) {
    printf("[ERROR] Grid2DDevice::linear_combination - mismatch: (%d, %d) != (%d, %d)\n", this->pnx(),this->pny(), g2.pnx(), g2.pny());
    return false;
  }

  dim3 Dg((this->num_allocated_elements() + 255) / 256);
  dim3 Db(256);
  
  Grid2DDevice_linear_combination2<<<Dg, Db>>>(this->buffer(), alpha1, g1.buffer(), alpha2, g2.buffer(), this->num_allocated_elements());
  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] Grid2DDeviceF::linear_combination - CUDA error \"%s\"\n", cudaGetErrorString(er));
    return false;    
  }
  
  return true;
}


template<typename T>
Grid2DDevice<T>::~Grid2DDevice()
{
  cudaFree(this->_buffer);
}

template<typename T>
bool Grid2DDevice<T>::init(int nx_val, int ny_val, int gx_val, int gy_val, int padx, int pady)
{
  this->_nx = nx_val;
  this->_ny = ny_val;

  this->_gx = gx_val;
  this->_gy = gy_val;
  
  // pad with ghost cells & user-specified padding
  this->_pnx = this->_nx + 2 * gx_val + padx;
  this->_pny = this->_ny + 2 * gy_val + pady;

  // either shift by 4 bits (float and int) or 3 bits (double)
  // then the mask is either 15 or 7.
  int shift_amount = (sizeof(T) == 4 ? 4 : 3);
  int mask = (0x1 << shift_amount) - 1;

  // round up pnz to next multiple of 16 if needed
  if (this->_pny & mask)
    this->_pny = ((this->_pny >> shift_amount) + 1) << shift_amount;

  if ((unsigned int)CUDA_SUCCESS != cudaMalloc((void **)&this->_buffer, sizeof(T) * this->num_allocated_elements())) {
    printf("[ERROR] Grid2DDeviceF::init - cudaMalloc failed\n");
    return false;
  }
  
  this->_shifted_buffer = this->_buffer + this->_gx * this->_pny + this->_gy;
  return true;
}


template bool Grid2DDevice<float>::init(int nx_val, int ny_val, int gx_val, int gy_val, int padx, int pady);
template bool Grid2DDevice<int>::init(int nx_val, int ny_val, int gx_val, int gy_val, int padx, int pady);
template bool Grid2DDevice<double>::init(int nx_val, int ny_val, int gx_val, int gy_val, int padx, int pady);

template Grid2DDevice<float>::~Grid2DDevice();
template Grid2DDevice<int>::~Grid2DDevice();
template Grid2DDevice<double>::~Grid2DDevice();

template bool Grid2DDevice<float>::copy_all_data(const Grid2DHost<float> &from);
template bool Grid2DDevice<int>::copy_all_data(const Grid2DHost<int> &from);
template bool Grid2DDevice<double>::copy_all_data(const Grid2DHost<double> &from);

template bool Grid2DDevice<float>::copy_all_data(const Grid2DDevice<float> &from);
template bool Grid2DDevice<int>::copy_all_data(const Grid2DDevice<int> &from);
template bool Grid2DDevice<double>::copy_all_data(const Grid2DDevice<double> &from);

template bool Grid2DDevice<float>::clear_zero();
template bool Grid2DDevice<int>::clear_zero();
template bool Grid2DDevice<double>::clear_zero();

template bool Grid2DDevice<float>::clear(float val);
template bool Grid2DDevice<int>::clear(int val);


template bool Grid2DDevice<float>::linear_combination(float alpha1, const Grid2DDevice<float> &g1);
template bool Grid2DDevice<float>::linear_combination(float alpha1, const Grid2DDevice<float> &g1, float alpha2, const Grid2DDevice<float> &g2);

#ifdef OCU_DOUBLESUPPORT
template bool Grid2DDevice<double>::linear_combination(double alpha1, const Grid2DDevice<double> &g1);
template bool Grid2DDevice<double>::linear_combination(double alpha1, const Grid2DDevice<double> &g1, double alpha2, const Grid2DDevice<double> &g2);
template bool Grid2DDevice<double>::clear(double val);
#endif
} // end namespace

