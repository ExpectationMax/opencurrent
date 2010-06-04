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
#include <memory.h>


#include "ocustorage/coarray.h"
#include "ocuutil/memory.h"
#include "ocuutil/reduction_op.h"


namespace ocu {


template<typename T>
bool 
Grid1DHost<T>::copy_interior_data(const Grid1DDevice<T> &from)
{
  if (from.nx() != this->nx()) {
    printf("[ERROR] Grid1DHost::copy_interior_data - nx mismatch: %d != %d\n", this->nx(), from.nx());
    return false;
  }

  if ((unsigned int) CUDA_SUCCESS != cudaMemcpy(this->_shifted_buffer, &from.at(0), sizeof(T) * this->nx(), cudaMemcpyDeviceToHost)) {
    printf("[ERROR] Grid1DHost::copy_interior_data - cudaMemcpy failed\n");
    return false;
  }
  
  return true;
}

template<typename T>
bool 
Grid1DHost<T>::copy_all_data(const Grid1DDevice<T> &from)
{
  if (from.pnx() != this->pnx()) {
    printf("[ERROR] Grid1DHost::copy_all_data - pnx mismatch: %d != %d\n", this->pnx(), from.pnx());
    return false;
  }

  cudaError_t er = cudaMemcpy(this->_buffer, from.buffer(), sizeof(T) * this->pnx(), cudaMemcpyDeviceToHost);
  if (er != (unsigned int) CUDA_SUCCESS) {
    printf("[ERROR] Grid1DHost::copy_all_data - cudaMemcpy failed with %s\n", cudaGetErrorString(er));
    return false;
  }
  
  return true;
}




template<typename T, typename REDUCE>
void reduce_with_operator(const Grid1DHost<T> &grid, T &result, REDUCE reduce)
{
  const T *iter = &grid.at(0);
  const T *last = iter + grid.nx();
  while (iter != last) {
    result = reduce(reduce.process(*iter), result);
    ++iter;
  }
}

template<>
bool Grid1DHost<float>::reduce_maxabs(float &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceMaxAbsF());
  return true;
}

template<>
bool Grid1DHost<double>::reduce_maxabs(double &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceMaxAbsD());
  return true;
}


template<>
bool Grid1DHost<int>::reduce_maxabs(int &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceMaxAbsI());
  return true;
}

template<typename T>
bool Grid1DHost<T>::reduce_min(T &result) const
{
  result = this->at(0);
  reduce_with_operator(*this, result, HostReduceMin<T>());
  return true;
}

template<typename T>
bool Grid1DHost<T>::reduce_max(T &result) const
{
  result = this->at(0);
  reduce_with_operator(*this, result, HostReduceMax<T>());
  return true;
}

template<typename T>
bool Grid1DHost<T>::reduce_checknan(T &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceCheckNan<T>());
  return true;
}

template<typename T>
bool Grid1DHost<T>::reduce_sum(T &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceSum<T>());
  return true;
}

template<typename T>
bool Grid1DHost<T>::reduce_sqrsum(T &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceSqrSum<T>());
  return true;
}

template<typename T>
void Grid1DHost<T>::clear_zero()
{
  memset(this->_buffer, 0, this->_pnx * sizeof(T));
}

template<typename T>
void Grid1DHost<T>::clear(T t)
{
  T *ptr = &this->at(0);
  for (int i=0; i < this->_nx; i++) {
    (*ptr) = t;
    ptr++;
  }
}


template<typename T>
bool 
Grid1DHost<T>::copy_interior_data(const Grid1DHost<T> &from)
{
  if (from.nx() != this->nx()) {
    printf("[ERROR] Grid1DHost::copy_interior_data - nx mismatch: %d != %d\n", this->nx(), from.nx());
    return false;
  }

  memcpy(this->_shifted_buffer, from._shifted_buffer, sizeof(T) * this->nx());
  return true;
}

template<typename T>
bool 
Grid1DHost<T>::copy_all_data(const Grid1DHost<T> &from)
{
  if (from.pnx() != this->pnx()) {
    printf("[ERROR] Grid1DHost::copy_interior_data - pnx mismatch: %d != %d\n", this->pnx(), from.pnx());
    return false;
  }

  memcpy(this->_buffer, from._buffer, sizeof(T) * this->pnx());
  return true;
}

template<typename T>
bool 
Grid1DHost<T>::linear_combination(T alpha1, const Grid1DHost<T> &g1)
{
  if (g1.nx() != this->nx()) {
    printf("[ERROR] Grid1DHost::linear_combination - nx mismatch: %d != %d\n", this->nx(), g1.nx());
    return false;
  }

  T *this_ptr = &this->at(0);
  const T *g1_ptr = &g1.at(0);
  const T *last_ptr = &this->at(this->nx());
  while (this_ptr != last_ptr) {
    *this_ptr = (*g1_ptr) * alpha1;
    ++this_ptr;
    ++g1_ptr;
  }

  return true;
}


template<typename T>
bool 
Grid1DHost<T>::linear_combination(T alpha1, const Grid1DHost<T> &g1, T alpha2, const Grid1DHost<T> &g2)
{
  if (g1.nx() != this->nx()) {
    printf("[ERROR] Grid1DHost::linear_combination - nx mismatch: %d != %d\n", this->nx(), g1.nx());
    return false;
  }

  if (g2.nx() != this->nx()) {
    printf("[ERROR] Grid1DHost::linear_combination - nx mismatch: %d != %d\n", this->nx(), g2.nx());
    return false;
  }

  T *this_ptr = &this->at(0);
  const T *g1_ptr = &g1.at(0);
  const T *g2_ptr = &g2.at(0);
  const T *last_ptr = this_ptr + this->nx();
  while (this_ptr != last_ptr) {
    *this_ptr = (*g1_ptr) * alpha1 + (*g2_ptr) * alpha2;
    ++this_ptr;
    ++g1_ptr;
    ++g2_ptr;
  }

  return true;
}


template<typename T>
Grid1DHost<T>::~Grid1DHost()
{
  host_free(this->_buffer, this->_pinned);
}


template<typename T>
bool 
Grid1DHost<T>::init(int nx_val, int gx_val, bool pinned)
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

  this->_buffer = (T *)host_malloc(this->_pnx * sizeof(T), pinned);
  this->_pinned = pinned;
  this->_shifted_buffer = this->buffer() + this->_gx;

  if (!this->_buffer) {
    printf("[ERROR] Grid1DHost::init - memory allocation failed\n");
    return false;
  }

  return true;
}


template<typename T>
Grid1DHostCo<T>::Grid1DHostCo(const char *id) 
{ 
  _table = CoArrayManager::register_coarray(id, this->_image_id, this);
}

template<typename T>
Grid1DHostCo<T>::~Grid1DHostCo() 
{
  CoArrayManager::unregister_coarray(_table->name, this->_image_id);
}


template<typename T>
Grid1DHostCo<T> *Grid1DHostCo<T>::co(int image)       
{ 
  return (Grid1DHostCo<T> *)(_table->table[image]); 
}

template<typename T>
const Grid1DHostCo<T> *Grid1DHostCo<T>::co(int image) const 
{ 
  return (const Grid1DHostCo<T> *)(_table->table[image]); 
}


template<>
bool Grid1DHostCo<float>::co_reduce_maxabs(float &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsF()); 
  return ok;
}

template<>
bool Grid1DHostCo<int>::co_reduce_maxabs(int &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsI()); 
  return ok;
}

template<>
bool Grid1DHostCo<double>::co_reduce_maxabs(double &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsD()); 
  return ok;
}

template<typename T>
bool Grid1DHostCo<T>::co_reduce_sum(T &result) const
{
  bool ok = reduce_sum(result);
  result = ThreadManager::barrier_reduce(result, HostReduceSum<T>()); 
  return ok;
}

template<typename T>
bool Grid1DHostCo<T>::co_reduce_sqrsum(T &result) const
{
  bool ok = reduce_sqrsum(result);
  result = ThreadManager::barrier_reduce(result, HostReduceSum<T>()); 
  return ok;
}

template<typename T>
bool Grid1DHostCo<T>::co_reduce_max(T &result) const
{
  bool ok = reduce_max(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMax<T>()); 
  return ok;
}

template<typename T>
bool Grid1DHostCo<T>::co_reduce_min(T &result) const
{
  bool ok = reduce_min(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMin<T>()); 
  return ok;
}

template<typename T>
bool Grid1DHostCo<T>::co_reduce_checknan(T &result) const
{
  bool ok = reduce_checknan(result);
  result = ThreadManager::barrier_reduce(result, HostReduceCheckNan<T>()); 
  return ok;
}




template class Grid1DHost<float>;
template class Grid1DHost<double>;
template class Grid1DHost<int>;
template class Grid1DHostCo<float>;
template class Grid1DHostCo<double>;
template class Grid1DHostCo<int>;



} // end namespace

