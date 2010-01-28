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
#include "ocustorage/grid1d.h"
#include "ocustorage/grid1dops.h"

template<typename T>
__global__ void Grid1DDevice_linear_combination1(T *result, T alpha1, const T *g1, int nx, int gx)
{
  int i = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
  
  i -= gx;

  if (i >= 0 && i < nx) {
    result[i] = g1[i] * alpha1;  
  }
}

template<typename T>
__global__ void Grid1DDevice_linear_combination2(T *result, T alpha1, const T *g1, T alpha2, const T *g2, int nx, int gx)
{
  int i = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
  
  i -= gx;

  if (i >= 0 && i < nx) {
    result[i] = g1[i] * alpha1 + g2[i] * alpha2;  
  }
}


namespace ocu {


  
template<>
bool Grid1DDevice<int>::reduce_max(int &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxI());
}

template<>
bool Grid1DDevice<float>::reduce_max(float &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxF());
}
  
template<>
bool Grid1DDevice<int>::reduce_min(int &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMinI());
}

template<>
bool Grid1DDevice<float>::reduce_min(float &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMinF());
}
  
template<>
bool Grid1DDevice<int>::reduce_maxabs(int &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxAbsI());
}

template<>
bool Grid1DDevice<float>::reduce_maxabs(float &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxAbsF());
}

#ifdef OCU_DOUBLESUPPORT

template<>
bool Grid1DDevice<double>::reduce_maxabs(double &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxAbsD());
}

template<>
bool Grid1DDevice<double>::reduce_max(double &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMaxD());
}

template<>
bool Grid1DDevice<double>::reduce_min(double &result) const
{
  return reduce_with_operator(*this, result, ReduceDevMinD());
}

#endif // OCU_DOUBLESUPPORT

template<typename T>
bool Grid1DDevice<T>::reduce_checknan(T &result) const
{
  return reduce_with_operator(*this, result, ReduceDevCheckNan<T>());
}

template<typename T>
bool Grid1DDevice<T>::reduce_sum(T &result) const
{
  return reduce_with_operator(*this, result, ReduceDevSum<T>());
}

template<typename T>
bool Grid1DDevice<T>::reduce_sqrsum(T &result) const
{
  return reduce_with_operator(*this, result, ReduceDevSqrSum<T>());
}




template<typename T>
void Grid1DDevice<T>::clear_zero()
{
  if (cudaMemset(this->_buffer, 0, this->_pnx * sizeof(T)) != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] Grid1DDeviceT::clear_zero - cudaMemset failed\n");
  }
}


template<typename T>
bool 
Grid1DDevice<T>::copy_interior_data(const Grid1DHost<T> &from)
{
  if (from.nx() != this->nx()) {
    printf("[ERROR] Grid1DDevice::copy_interior_data - nx mismatch: %d != %d\n", this->nx(), from.nx());
    return false;
  }

  if ((unsigned int) CUDA_SUCCESS != cudaMemcpy(this->_shifted_buffer, &from.at(0), sizeof(T) * this->nx(), cudaMemcpyHostToDevice)) {
    printf("[ERROR] Grid1DDevice::copy_interior_data - cudaMemcpy failed\n");
    return false;
  }
  
  return true;
}

template<typename T>
bool 
Grid1DDevice<T>::copy_all_data(const Grid1DHost<T> &from)
{
  if (from.pnx() != this->pnx()) {
    printf("[ERROR] Grid1DDevice::copy_all_data - pnx mismatch: %d != %d\n", this->pnx(), from.pnx());
    return false;
  }

  if ((unsigned int) CUDA_SUCCESS != cudaMemcpy(this->_buffer, from.buffer(), sizeof(T) * this->pnx(), cudaMemcpyHostToDevice)) {
    printf("[ERROR] Grid1DDevice::copy_all_data - cudaMemcpy failed\n");
    return false;
  }
  
  return true;
}


template<typename T>
bool 
Grid1DDevice<T>::copy_interior_data(const Grid1DDevice<T> &from)
{
  if (from.nx() != this->nx()) {
    printf("[ERROR] Grid1DDevice::copy_interior_data - nx mismatch: %d != %d\n", this->nx(), from.nx());
    return false;
  }

  if ((unsigned int) CUDA_SUCCESS != cudaMemcpy(this->_shifted_buffer, &from.at(0), sizeof(T) * this->nx(), cudaMemcpyDeviceToDevice)) {
    printf("[ERROR] Grid1DDevice::copy_interior_data - cudaMemcpy failed\n");
    return false;
  }
  
  return true;
}

template<typename T>
bool 
Grid1DDevice<T>::copy_all_data(const Grid1DDevice<T> &from)
{
  if (from.pnx() != this->pnx()) {
    printf("[ERROR] Grid1DDevice::copy_all_data - pnx mismatch: %d != %d\n", this->pnx(), from.pnx());
    return false;
  }

  if (cudaMemcpy(this->_buffer, from.buffer(), sizeof(T) * this->pnx(), cudaMemcpyDeviceToDevice) != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Grid1DDevice::copy_all_data - cudaMemcpy failed\n");
    return false;
  }
  
  return true;
}

template<typename T>
bool Grid1DDevice<T>::linear_combination(T alpha1, const Grid1DDevice<T> &g1)
{
  if (g1.pnx() != this->pnx()) {
    printf("[ERROR] Grid1DDevice::linear_combination - pnx mismatch: %d != %d\n", this->pnx(), g1.pnx());
    return false;
  }

  dim3 Dg((this->nx() + this->gx() + 255) / 256);
  dim3 Db(256);
  
  Grid1DDevice_linear_combination1<<<Dg, Db>>>(&this->at(0), alpha1, &g1.at(0), this->nx(), this->gx());
  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Grid1DDeviceF::linear_combination - CUDA error \"%s\"\n", cudaGetErrorString(er));
    return false;    
  }
  
  return true;
}

template<typename T>
bool Grid1DDevice<T>::linear_combination(T alpha1, const Grid1DDevice<T> &g1, T alpha2, const Grid1DDevice<T> &g2)
{
  if (g1.pnx() != this->pnx()) {
    printf("[ERROR] Grid1DDevice::linear_combination - pnx mismatch: %d != %d\n", this->pnx(), g1.pnx());
    return false;
  }

  if (g2.pnx() != this->pnx()) {
    printf("[ERROR] Grid1DDevice::linear_combination - pnx mismatch: %d != %d\n", this->pnx(), g2.pnx());
    return false;
  }


  dim3 Dg((this->nx() + this->gx() + 255) / 256);
  dim3 Db(256);
  
  Grid1DDevice_linear_combination2<<<Dg, Db>>>(&this->at(0), alpha1, &g1.at(0), alpha2, &g2.at(0), this->nx(), this->gx());
  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Grid1DDeviceF::linear_combination - CUDA error \"%s\"\n", cudaGetErrorString(er));
    return false;    
  }
  
  return true;
}


template<typename T>
Grid1DDevice<T>::~Grid1DDevice()
{
  cudaFree(this->_buffer);
}

template<typename T>
bool Grid1DDevice<T>::init(int nx_val, int gx_val)
{
  this->_nx = nx_val;
  this->_gx = gx_val;
  
  // pad with ghost cells
  this->_pnx = this->_nx + 2 * gx_val;

  // either shift by 4 bits (float and int) or 3 bits (double)
  // then the mask is either 15 or 7.
  int shift_amount = (sizeof(T) == 4 ? 4 : 3);
  int mask = (0x1 << shift_amount) - 1;
  
  // round up pnx to next multiple of 16 if needed
  if (this->_pnx & mask)
    this->_pnx = ((this->_pnx >> shift_amount) + 1) << shift_amount;

  if ((unsigned int) CUDA_SUCCESS != cudaMalloc((void **)&this->_buffer, sizeof(T) * this->_pnx)) {
    printf("[ERROR] Grid1DDeviceF::init - cudaMalloc failed\n");
    return false;
  }
  
  this->_shifted_buffer = this->_buffer + this->_gx;
  return true;
}


template bool Grid1DDevice<float>::init(int nx_val, int gx_val);
template bool Grid1DDevice<int>::init(int nx_val, int gx_val);
template bool Grid1DDevice<double>::init(int nx_val, int gx_val);

template Grid1DDevice<float>::~Grid1DDevice();
template Grid1DDevice<int>::~Grid1DDevice();
template Grid1DDevice<double>::~Grid1DDevice();

template bool Grid1DDevice<float>::reduce_sqrsum(float &result) const;
template bool Grid1DDevice<int>::reduce_sqrsum(int &result) const;

template bool Grid1DDevice<float>::reduce_sum(float &result) const;
template bool Grid1DDevice<int>::reduce_sum(int &result) const;

template bool Grid1DDevice<float>::reduce_checknan(float &result) const;

template bool Grid1DDevice<float>::copy_interior_data(const Grid1DHost<float> &from);
template bool Grid1DDevice<int>::copy_interior_data(const Grid1DHost<int> &from);
template bool Grid1DDevice<double>::copy_interior_data(const Grid1DHost<double> &from);

template bool Grid1DDevice<float>::copy_all_data(const Grid1DHost<float> &from);
template bool Grid1DDevice<int>::copy_all_data(const Grid1DHost<int> &from);
template bool Grid1DDevice<double>::copy_all_data(const Grid1DHost<double> &from);

template bool Grid1DDevice<float>::copy_interior_data(const Grid1DDevice<float> &from);
template bool Grid1DDevice<int>::copy_interior_data(const Grid1DDevice<int> &from);
template bool Grid1DDevice<double>::copy_interior_data(const Grid1DDevice<double> &from);

template bool Grid1DDevice<float>::copy_all_data(const Grid1DDevice<float> &from);
template bool Grid1DDevice<int>::copy_all_data(const Grid1DDevice<int> &from);
template bool Grid1DDevice<double>::copy_all_data(const Grid1DDevice<double> &from);

template void Grid1DDevice<float>::clear_zero();
template void Grid1DDevice<int>::clear_zero();
template void Grid1DDevice<double>::clear_zero();

template bool Grid1DDevice<float>::linear_combination(float alpha1, const Grid1DDevice<float> &g1);
template bool Grid1DDevice<float>::linear_combination(float alpha1, const Grid1DDevice<float> &g1, float alpha2, const Grid1DDevice<float> &g2);

#ifdef OCU_DOUBLESUPPORT

template bool Grid1DDevice<double>::reduce_sqrsum(double &result) const;
template bool Grid1DDevice<double>::reduce_sum(double &result) const;
template bool Grid1DDevice<double>::reduce_checknan(double &result) const;

#endif // OCU_DOUBLESUPPORT


} // end namespace
