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


#include <memory.h>
#include <cuda.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

#include "ocustorage/coarray.h"
#include "ocuutil/memory.h"
#include "ocuutil/reduction_op.h"
#include "ocuutil/kernel_wrapper.h"

namespace ocu {


template<typename T>
bool 
Grid3DHost<T>::copy_all_data(const Grid3DDevice<T> &from)
{
  if (!this->check_layout_match(from)) {
    printf("[ERROR] Grid3DHost::copy_all_data - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(), this->pny(), this->pnz(), from.pnx(),  from.pny(),  from.pnz());
    return false;
  }

  KernelWrapper wrapper;
  wrapper.PreKernel();
  if ((unsigned int) CUDA_SUCCESS != cudaMemcpy(this->_buffer, from.buffer(), sizeof(T) * this->num_allocated_elements(), cudaMemcpyDeviceToHost)) {
    printf("[ERROR] Grid3DHost::copy_all_data - cudaMemcpy failed\n");
    return false;
  }
  return wrapper.PostKernel("cudaMemcpy(DeviceToHost)");
}


template<typename T, typename REDUCE>
void reduce_with_operator(const Grid3DHost<T> &grid, T &result, REDUCE reduce)
{
  for (int i=0; i < grid.nx(); i++)
    for (int j=0; j < grid.ny(); j++) {
      const T *iter = &grid.at(i,j,0);
      const T *last = iter + grid.nz();
      while (iter != last) {
        result = reduce(reduce.process(*iter), result);
        ++iter;
      }
    }
}



template<>
bool Grid3DHost<double>::reduce_maxabs(double &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceMaxAbsD());
  return true;
}


template<>
bool Grid3DHost<float>::reduce_maxabs(float &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceMaxAbsF());
  return true;
}

template<>
bool Grid3DHost<int>::reduce_maxabs(int &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceMaxAbsI());
  return true;
}

template<typename T>
bool Grid3DHost<T>::reduce_sum(T &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceSum<T>());
  return true;
}

template<typename T>
bool Grid3DHost<T>::reduce_checknan(T &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceCheckNan<T>());
  return true;
}

template<>
bool Grid3DHost<int>::reduce_checknan(int &result) const
{
  printf("[WARNING] Grid3DHost<int>::reduce_checknan - operation not supported for 'int' types\n");
  return false;
}


template<typename T>
bool Grid3DHost<T>::reduce_min(T &result) const
{
  result = this->at(0,0,0);
  reduce_with_operator(*this, result, HostReduceMin<T>());
  return true;
}

template<typename T>
bool Grid3DHost<T>::reduce_max(T &result) const
{
  result = this->at(0,0,0);
  reduce_with_operator(*this, result, HostReduceMax<T>());
  return true;
}


template<typename T>
bool Grid3DHost<T>::reduce_sqrsum(T &result) const
{
  result = 0;
  reduce_with_operator(*this, result, HostReduceSqrSum<T>());
  return true;
}

template<typename T>
void Grid3DHost<T>::clear_zero()
{
  memset(this->_buffer, 0, this->num_allocated_elements() * sizeof(T));
}

template<typename T>
void Grid3DHost<T>::clear(T val)
{
  T *this_ptr = this->buffer();
  const T*last_ptr = this_ptr + this->num_allocated_elements();
  while (this_ptr != last_ptr) {
    *this_ptr = val;
    ++this_ptr;
  }
}

template<typename T>
template<typename S>
bool 
Grid3DHost<T>::copy_all_data(const Grid3DHost<S> &from)
{
  if (!check_layout_match(from)) {
    printf("[ERROR] Grid3DHost::copy_interior_data - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(), this->pny(), this->pnz(), from.pnx(), from.pny(), from.pnz());
    return false;
  }

  T *this_ptr = this->buffer();
  const T *last_ptr = this_ptr + this->num_allocated_elements();
  const S *from_ptr = from.buffer();
  while (this_ptr != last_ptr) {
    *this_ptr = (T)(*from_ptr);
    ++this_ptr;
    ++from_ptr;
  }

  return true;
}

template<>
template<>
bool
Grid3DHost<float>::copy_all_data(const Grid3DHost<float> &from)
{
  if (!check_layout_match(from)) {
    printf("[ERROR] Grid3DHost::copy_interior_data - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(), this->pny(), this->pnz(), from.pnx(), from.pny(), from.pnz());
    return false;
  }

  // faster bulk copy
  memcpy(_buffer, from.buffer(), num_allocated_elements() * sizeof(float));
  return true;
}

template<>
template<>
bool
Grid3DHost<int>::copy_all_data(const Grid3DHost<int> &from)
{
  if (!check_layout_match(from)) {
    printf("[ERROR] Grid3DHost::copy_interior_data - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", pnx(), pny(), pnz(), from.pnx(), from.pny(), from.pnz());
    return false;
  }

  // faster bulk copy
  memcpy(_buffer, from.buffer(), num_allocated_elements() * sizeof(int));
  return true;
}

template<>
template<>
bool
Grid3DHost<double>::copy_all_data(const Grid3DHost<double> &from)
{
  if (!check_layout_match(from)) {
    printf("[ERROR] Grid3DHost::copy_interior_data - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", pnx(), pny(), pnz(), from.pnx(), from.pny(), from.pnz());
    return false;
  }

  // faster bulk copy
  memcpy(_buffer, from.buffer(), num_allocated_elements() * sizeof(double));
  return true;
}


template<typename T>
bool 
Grid3DHost<T>::linear_combination(T alpha1, const Grid3DHost<T> &g1)
{
  if (check_layout_match(g1)) {

    // fast version if all dimensions match
    T *this_ptr = &this->at(0,0,0);
    const T *g1_ptr = &g1.at(0,0,0);
    const T *last_ptr = &this->at(this->nx(), this->ny(), this->nz());
    while (this_ptr != last_ptr) {
      *this_ptr = (*g1_ptr) * alpha1;
      ++this_ptr;
      ++g1_ptr;
    }
  }
  else if (check_interior_dimension_match(g1)) {
    
    // slow version if not all dimensions match
    for (int i=0; i < this->nx(); i++)
      for (int j=0; j < this->ny(); j++) {
        T *this_ptr = &this->at(i,j,0);
        const T *g1_ptr = &g1.at(i,j,0);
        const T *last_ptr = &this->at(i,j,this->nz());
        while (this_ptr != last_ptr) {
          *this_ptr = (*g1_ptr) * alpha1;
          ++this_ptr;
          ++g1_ptr;
        }
      }
  }
  else {
    printf("[ERROR] Grid1DHost::linear_combination - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->nx(), this->ny(), this->nz(), g1.nx(), g1.ny(), g1.nz());
    return false;
  }

  return true;
}


template<typename T>
bool 
Grid3DHost<T>::linear_combination(T alpha1, const Grid3DHost<T> &g1, T alpha2, const Grid3DHost<T> &g2)
{
  if (check_layout_match(g1) && check_layout_match(g2)) {

    // fast version if all dimensions match
    T *this_ptr = &this->at(0,0,0);
    const T *g1_ptr = &g1.at(0,0,0);
    const T *g2_ptr = &g2.at(0,0,0);
    const T *last_ptr = &this->at(this->nx(), this->ny(), this->nz());
    while (this_ptr != last_ptr) {
    *this_ptr = (*g1_ptr) * alpha1 + (*g2_ptr) * alpha2;
      ++this_ptr;
      ++g1_ptr;
      ++g2_ptr;
    }
  }
  else if (check_interior_dimension_match(g1) && check_interior_dimension_match(g2)) {
    
    // slow version if not all dimensions match
    for (int i=0; i < this->nx(); i++)
      for (int j=0; j < this->ny(); j++) {
        T *this_ptr = &this->at(i,j,0);
        const T *g1_ptr = &g1.at(i,j,0);
        const T *g2_ptr = &g2.at(i,j,0);
        const T *last_ptr = &this->at(i,j,this->nz());
        while (this_ptr != last_ptr) {
          *this_ptr = (*g1_ptr) * alpha1 + (*g2_ptr) * alpha2;
          ++this_ptr;
          ++g1_ptr;
          ++g2_ptr;
        }
      }
  }
  else {
    printf("[ERROR] Grid1DHost::linear_combination - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->nx(), this->ny(), this->nz(), g1.nx(), g1.ny(), g1.nz());
    return false;
  }

  return true;
}


template<typename T>
Grid3DHost<T>::~Grid3DHost()
{
  host_free(this->_buffer, this->_pinned);
}


template<typename T>
bool 
Grid3DHost<T>::init(int nx_val, int ny_val, int nz_val, int gx_val, int gy_val, int gz_val, bool pinned , int padx, int pady, int padz)
{
  this->_nx = nx_val;
  this->_ny = ny_val;
  this->_nz = nz_val;

  this->_gx = gx_val;
  this->_gy = gy_val;
  this->_gz = gz_val;
  
  // pad with ghost cells
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

  int pre_padding =  (16 - this->_gz); 

  this->_pnzpny = this->_pnz * this->_pny;
  this->_allocated_elements = this->_pnzpny * this->_pnx + pre_padding;

  this->_buffer = (T *)host_malloc(this->num_allocated_elements() * sizeof(T), pinned);
  this->_pinned = pinned;
  this->_shift_amount   = this->_gx * this->_pnzpny + this->_gy * this->_pnz + this->_gz + pre_padding; 
  this->_shifted_buffer = this->buffer() + this->_shift_amount;

  if (!this->_buffer) {
    printf("[ERROR] Grid3DHost::init - memory allocation failed\n");
    return false;
  }

  return true;
}


template<typename T>
Grid3DHostCo<T>::Grid3DHostCo(const char *id) 
{
  this->_table = CoArrayManager::register_coarray(id, this->_image_id, this);
}

template<typename T>
Grid3DHostCo<T>::~Grid3DHostCo() 
{
  CoArrayManager::unregister_coarray(this->_table->name, this->_image_id);
}

template<typename T>
Grid3DHostCo<T> *Grid3DHostCo<T>::co(int image)
{ 
  return (Grid3DHostCo<T> *)(this->_table->table[image]); 
}

template<typename T>
const Grid3DHostCo<T> *Grid3DHostCo<T>::co(int image) const 
{ 
  return (const Grid3DHostCo<T> *)(this->_table->table[image]); 
}

template<>
bool Grid3DHostCo<float>::co_reduce_maxabs(float &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsF()); 
  return ok;
}

template<>
bool Grid3DHostCo<int>::co_reduce_maxabs(int &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsI()); 
  return ok;
}

template<>
bool Grid3DHostCo<double>::co_reduce_maxabs(double &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsD()); 
  return ok;
}

template<typename T>
bool Grid3DHostCo<T>::co_reduce_sum(T &result) const
{
  bool ok = reduce_sum(result);
  result = ThreadManager::barrier_reduce(result, HostReduceSum<T>()); 
  return ok;
}

template<typename T>
bool Grid3DHostCo<T>::co_reduce_sqrsum(T &result) const
{
  bool ok = reduce_sqrsum(result);
  result = ThreadManager::barrier_reduce(result, HostReduceSum<T>()); 
  return ok;
}

template<typename T>
bool Grid3DHostCo<T>::co_reduce_max(T &result) const
{
  bool ok = reduce_max(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMax<T>()); 
  return ok;
}

template<typename T>
bool Grid3DHostCo<T>::co_reduce_min(T &result) const
{
  bool ok = reduce_min(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMin<T>()); 
  return ok;
}

template<typename T>
bool Grid3DHostCo<T>::co_reduce_checknan(T &result) const
{
  bool ok = reduce_checknan(result);
  result = ThreadManager::barrier_reduce(result, HostReduceCheckNan<T>()); 
  return ok;
}

template<>
bool Grid3DHostCo<int>::co_reduce_checknan(int &result) const
{
  printf("[WARNING] Grid3DHostCo<int>::reduce_checknan - operation not supported for 'int' types\n");
  return false;
}

template class Grid3DHost<float>;
template class Grid3DHost<int>;
template class Grid3DHost<double>;
template class Grid3DHostCo<float>;
template class Grid3DHostCo<int>;
template class Grid3DHostCo<double>;

// because these are doubly-templated, they need to be explicitly instantiated
template bool Grid3DHost<float>::copy_all_data(const Grid3DHost<int> &from);
template bool Grid3DHost<float>::copy_all_data(const Grid3DHost<double> &from);
template bool Grid3DHost<int>::copy_all_data(const Grid3DHost<float> &from);
template bool Grid3DHost<int>::copy_all_data(const Grid3DHost<double> &from);
template bool Grid3DHost<double>::copy_all_data(const Grid3DHost<float> &from);
template bool Grid3DHost<double>::copy_all_data(const Grid3DHost<int> &from);



} // end namespace

