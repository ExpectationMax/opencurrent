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

#include "ocustorage/grid1d.h"
#include "ocuutil/memory.h"
#include "ocuutil/reduction_op.h"

namespace ocu {

template<typename T, typename REDUCE>
void reduce_with_operator(const Grid1DHost<T> &grid, T &result, REDUCE reduce)
{
  const T *iter = &grid.at(0);
  const T *last = iter + grid.nx();
  while (iter != last) {
    result = reduce(*iter, result);
    ++iter;
  }
}

template<>
bool Grid1DHost<float>::reduce_maxabs(float &result) const
{
  result = 0;
  reduce_with_operator(*this, result, ReduceMaxAbsF());
  return true;
}

template<>
bool Grid1DHost<double>::reduce_maxabs(double &result) const
{
  result = 0;
  reduce_with_operator(*this, result, ReduceMaxAbsD());
  return true;
}


template<>
bool Grid1DHost<int>::reduce_maxabs(int &result) const
{
  result = 0;
  reduce_with_operator(*this, result, ReduceMaxAbsI());
  return true;
}

template<typename T>
bool Grid1DHost<T>::reduce_min(T &result) const
{
  result = this->at(0);
  reduce_with_operator(*this, result, ReduceMin<T>());
  return true;
}

template<typename T>
bool Grid1DHost<T>::reduce_max(T &result) const
{
  result = this->at(0);
  reduce_with_operator(*this, result, ReduceMax<T>());
  return true;
}

template<typename T>
bool Grid1DHost<T>::reduce_checknan(T &result) const
{
  result = 0;
  reduce_with_operator(*this, result, ReduceCheckNan<T>());
  return true;
}

template<typename T>
bool Grid1DHost<T>::reduce_sum(T &result) const
{
  result = 0;
  reduce_with_operator(*this, result, ReduceSum<T>());
  return true;
}

template<typename T>
bool Grid1DHost<T>::reduce_sqrsum(T &result) const
{
  result = 0;
  reduce_with_operator(*this, result, ReduceSqrSum<T>());
  return true;
}

template<typename T>
void Grid1DHost<T>::clear_zero()
{
  memset(this->_buffer, 0, this->_pnx * sizeof(T));
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
  this->_shifted_buffer = this->_buffer + this->_gx;

  if (!this->_buffer) {
    printf("[ERROR] Grid1DHost::init - memory allocation failed\n");
    return false;
  }

  return true;
}






// instantiate template functions here
template bool Grid1DHost<float>::init(int nx_val, int gx_val, bool pinned);
template bool Grid1DHost<int>::init(int nx_val, int gx_val, bool pinned);
template bool Grid1DHost<double>::init(int nx_val, int gx_val, bool pinned);

template Grid1DHost<float>::~Grid1DHost();
template Grid1DHost<int>::~Grid1DHost();
template Grid1DHost<double>::~Grid1DHost();

template bool Grid1DHost<float>::reduce_max(float &result) const;
template bool Grid1DHost<int>::reduce_max(int &result) const;
template bool Grid1DHost<double>::reduce_max(double &result) const;

template bool Grid1DHost<float>::reduce_min(float &result) const;
template bool Grid1DHost<int>::reduce_min(int &result) const;
template bool Grid1DHost<double>::reduce_min(double &result) const;

template bool Grid1DHost<float>::reduce_sum(float &result) const;
template bool Grid1DHost<int>::reduce_sum(int &result) const;
template bool Grid1DHost<double>::reduce_sum(double &result) const;

template bool Grid1DHost<float>::reduce_sqrsum(float &result) const;
template bool Grid1DHost<int>::reduce_sqrsum(int &result) const;
template bool Grid1DHost<double>::reduce_sqrsum(double &result) const;

template bool Grid1DHost<float>::reduce_checknan(float &result) const;
template bool Grid1DHost<double>::reduce_checknan(double &result) const;


template bool Grid1DHost<float>::copy_interior_data(const Grid1DHost<float> &from);
template bool Grid1DHost<int>::copy_interior_data(const Grid1DHost<int> &from);
template bool Grid1DHost<double>::copy_interior_data(const Grid1DHost<double> &from);

template bool Grid1DHost<float>::copy_all_data(const Grid1DHost<float> &from);
template bool Grid1DHost<int>::copy_all_data(const Grid1DHost<int> &from);
template bool Grid1DHost<double>::copy_all_data(const Grid1DHost<double> &from);

template void Grid1DHost<float>::clear_zero();
template void Grid1DHost<int>::clear_zero();
template void Grid1DHost<double>::clear_zero();

template bool Grid1DHost<float>::linear_combination(float alpha1, const Grid1DHost<float> &g1);
template bool Grid1DHost<int>::linear_combination(int alpha1, const Grid1DHost<int> &g1);
template bool Grid1DHost<double>::linear_combination(double alpha1, const Grid1DHost<double> &g1);

template bool Grid1DHost<float>::linear_combination(float alpha1, const Grid1DHost<float> &g1, float alpha2, const Grid1DHost<float> &g2);
template bool Grid1DHost<int>::linear_combination(int alpha1, const Grid1DHost<int> &g1, int alpha2, const Grid1DHost<int> &g2);
template bool Grid1DHost<double>::linear_combination(double alpha1, const Grid1DHost<double> &g1, double alpha2, const Grid1DHost<double> &g2);

} // end namespace

