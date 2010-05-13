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
#include "ocuutil/kernel_wrapper.h"
#include "ocustorage/coarray.h"
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
__global__ void Grid1DDevice_clear(T *result, T t, int nx, int gx)
{
  int i = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
  
  i -= gx;

  if (i >= 0 && i < nx) {
    result[i] = t;
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

template<>
bool Grid1DDevice<double>::reduce_checknan(double &result) const
{
  return reduce_with_operator(*this, result, ReduceDevCheckNan<double>());
}



#endif // OCU_DOUBLESUPPORT

template<>
bool Grid1DDevice<float>::reduce_checknan(float &result) const
{
  return reduce_with_operator(*this, result, ReduceDevCheckNan<float>());
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

template<>
bool Grid1DDevice<int>::reduce_checknan(int &result) const
{
  printf("[WARNING] Grid1DDevice<int>::reduce_checknan - operation not supported for 'int' types\n");
  return false;
}



template<typename T>
bool Grid1DDevice<T>::clear(T t)
{
  dim3 Dg((this->nx() + this->gx() + 255) / 256);
  dim3 Db(256);

  KernelWrapper wrapper;
  wrapper.PreKernel();  
  Grid1DDevice_clear<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(&this->at(0), t, this->nx(), this->gx());
  
  return wrapper.PostKernel("Grid1DDevice_clear");
}


template<typename T>
bool Grid1DDevice<T>::clear_zero()
{

  if (cudaMemset(this->_buffer, 0, this->_pnx * sizeof(T)) != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] Grid1DDeviceT::clear_zero - cudaMemset failed\n");
    return false;
  }

  return true;
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

  KernelWrapper wrapper;
  wrapper.PreKernel();
  Grid1DDevice_linear_combination1<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(&this->at(0), alpha1, &g1.at(0), this->nx(), this->gx());
  return wrapper.PostKernel("Grid1DDeviceF::linear_combination");
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
  
  
  KernelWrapper wrapper;

  wrapper.PreKernel();
  Grid1DDevice_linear_combination2<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(&this->at(0), alpha1, &g1.at(0), alpha2, &g2.at(0), this->nx(), this->gx());
  return wrapper.PostKernel("Grid1DDeviceF::linear_combination");
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
  
  this->_shifted_buffer = this->buffer() + this->_gx;
  return true;
}


template<typename T>
Grid1DDeviceCo<T>::Grid1DDeviceCo(const char *id) 
{ 
  this->_table = CoArrayManager::register_coarray(id, this->_image_id, this);
}

template<typename T>
Grid1DDeviceCo<T>::~Grid1DDeviceCo() 
{
  CoArrayManager::unregister_coarray(_table->name, this->_image_id);
}


template<typename T>
Grid1DDeviceCo<T> *Grid1DDeviceCo<T>::co(int image)       
{ 
  return (Grid1DDeviceCo<T> *)(_table->table[image]); 
}

template<typename T>
const Grid1DDeviceCo<T> *Grid1DDeviceCo<T>::co(int image) const 
{ 
  return (const Grid1DDeviceCo<T> *)(_table->table[image]); 
}





template<>
bool Grid1DDeviceCo<float>::co_reduce_maxabs(float &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsF()); 
  return ok;
}

template<>
bool Grid1DDeviceCo<int>::co_reduce_maxabs(int &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsI()); 
  return ok;
}

#ifdef OCU_DOUBLESUPPORT 
template<>
bool Grid1DDeviceCo<double>::co_reduce_maxabs(double &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsD()); 
  return ok;
}
#endif

template<typename T>
bool Grid1DDeviceCo<T>::co_reduce_sum(T &result) const
{
  bool ok = reduce_sum(result);
  result = ThreadManager::barrier_reduce(result, HostReduceSum<T>()); 
  return ok;
}

template<typename T>
bool Grid1DDeviceCo<T>::co_reduce_sqrsum(T &result) const
{
  bool ok = reduce_sqrsum(result);
  result = ThreadManager::barrier_reduce(result, HostReduceSum<T>()); 
  return ok;
}

template<typename T>
bool Grid1DDeviceCo<T>::co_reduce_max(T &result) const
{
  bool ok = reduce_max(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMax<T>()); 
  return ok;
}

template<typename T>
bool Grid1DDeviceCo<T>::co_reduce_min(T &result) const
{
  bool ok = reduce_min(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMin<T>()); 
  return ok;
}

template<typename T>
bool Grid1DDeviceCo<T>::co_reduce_checknan(T &result) const
{
  bool ok = reduce_checknan(result);
  result = ThreadManager::barrier_reduce(result, HostReduceCheckNan<T>()); 
  return ok;
}




template class Grid1DDevice<float>;
template class Grid1DDevice<int>;
template class Grid1DDeviceCo<float>;
template class Grid1DDeviceCo<int>;

#ifdef OCU_DOUBLESUPPORT

template class Grid1DDevice<double>;
template class Grid1DDeviceCo<double>;


#endif // OCU_DOUBLESUPPORT


} // end namespace
