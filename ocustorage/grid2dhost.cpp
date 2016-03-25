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

#include "ocustorage/grid2d.h"
#include "ocuutil/memory.h"

namespace ocu {


template<typename T>
void Grid2DHost<T>::clear_zero()
{
  memset(this->_buffer, 0, this->num_allocated_elements() * sizeof(T));
}

template<typename T>
void Grid2DHost<T>::clear(T val)
{
  T *this_ptr = this->_buffer;
  const T*last_ptr = this_ptr + this->num_allocated_elements();
  while (this_ptr != last_ptr) {
    *this_ptr = val;
    ++this_ptr;
  }
}


template<typename T>
bool 
Grid2DHost<T>::copy_interior_data(const Grid2DHost<T> &from)
{
  if (check_layout_match(from))
    return copy_all_data(from);

  if (!check_interior_dimension_match(from)) {
    printf("[ERROR] Grid2DHost::copy_interior_data - mismatch: (%d, %d) != (%d, %d)\n", this->nx(), this->ny(), from.nx(), from.ny());
    return false;
  }

  // slower since it has to skip the ghost nodes.
  for (int i=0; i < this->nx(); i++)
      memcpy(&this->at(i,0), &from.at(i,0), sizeof(T) * this->ny());

  return true;
}

template<typename T>
bool 
Grid2DHost<T>::copy_all_data(const Grid2DHost<T> &from)
{
  if (!check_layout_match(from)) {
    printf("[ERROR] Grid2DHost::copy_interior_data - mismatch: (%d, %d) != (%d, %d)\n", this->pnx(), this->pny(), from.pnx(), from.pny());
    return false;
  }

  // faster bulk copy
  memcpy(this->_buffer, from._buffer, this->num_allocated_elements() * sizeof(T));
  return true;
}

template<typename T>
bool 
Grid2DHost<T>::linear_combination(T alpha1, const Grid2DHost<T> &g1)
{
  if (check_layout_match(g1)) {

    // fast version if all dimensions match
    T *this_ptr = &this->at(0,0);
    const T *g1_ptr = &g1.at(0,0);
    const T *last_ptr = &this->at(this->nx(), this->ny());
    while (this_ptr != last_ptr) {
      *this_ptr = (*g1_ptr) * alpha1;
      ++this_ptr;
      ++g1_ptr;
    }
  }
  else if (check_interior_dimension_match(g1)) {
    
    // slow version if not all dimensions match
    for (int i=0; i < this->nx(); i++) {
        T *this_ptr = &this->at(i,0);
        const T *g1_ptr = &g1.at(i,0);
        const T *last_ptr = &this->at(i,this->ny());
        while (this_ptr != last_ptr) {
          *this_ptr = (*g1_ptr) * alpha1;
          ++this_ptr;
          ++g1_ptr;
        }
      }
  }
  else {
    printf("[ERROR] Grid2DHost::linear_combination - mismatch: (%d, %d) != (%d, %d)\n", this->nx(), this->ny(), g1.nx(), g1.ny());
    return false;
  }

  return true;
}


template<typename T>
bool 
Grid2DHost<T>::linear_combination(T alpha1, const Grid2DHost<T> &g1, T alpha2, const Grid2DHost<T> &g2)
{
  if (check_layout_match(g1) && check_layout_match(g2)) {

    // fast version if all dimensions match
    T *this_ptr = &this->at(0,0);
    const T *g1_ptr = &g1.at(0,0);
    const T *g2_ptr = &g2.at(0,0);
    const T *last_ptr = &this->at(this->nx(), this->ny());
    while (this_ptr != last_ptr) {
    *this_ptr = (*g1_ptr) * alpha1 + (*g2_ptr) * alpha2;
      ++this_ptr;
      ++g1_ptr;
      ++g2_ptr;
    }
  }
  else if (check_interior_dimension_match(g1) && check_interior_dimension_match(g2)) {
    
    // slow version if not all dimensions match
    for (int i=0; i < this->nx(); i++) {
        T *this_ptr = &this->at(i,0);
        const T *g1_ptr = &g1.at(i,0);
        const T *g2_ptr = &g2.at(i,0);
        const T *last_ptr = &this->at(i,this->ny());
        while (this_ptr != last_ptr) {
          *this_ptr = (*g1_ptr) * alpha1 + (*g2_ptr) * alpha2;
          ++this_ptr;
          ++g1_ptr;
          ++g2_ptr;
        }
      }
  }
  else {
    printf("[ERROR] Grid2DHost::linear_combination - mismatch: (%d, %d) != (%d, %d)\n", this->nx(), this->ny(), g1.nx(), g1.ny());
    return false;
  }

  return true;
}


template<typename T>
Grid2DHost<T>::~Grid2DHost()
{
  host_free(this->_buffer, this->_pinned);
}


template<typename T>
bool 
Grid2DHost<T>::init(int nx_val, int ny_val, int gx_val, int gy_val, bool pinned , int padx, int pady)
{
  this->_nx = nx_val;
  this->_ny = ny_val;

  this->_gx = gx_val;
  this->_gy = gy_val;
  
  // pad with ghost cells
  this->_pnx = this->_nx + 2 * gx_val + padx;
  this->_pny = this->_ny + 2 * gy_val + pady;

  // either shift by 4 bits (float and int) or 3 bits (double)
  // then the mask is either 15 or 7.
  int shift_amount = (sizeof(T) == 4 ? 4 : 3);
  int mask = (0x1 << shift_amount) - 1;

  // round up pnz to next multiple of 16 if needed
  if (this->_pny & mask)
    this->_pny = ((this->_pny >> shift_amount) + 1) << shift_amount;


  this->_buffer = (T *)host_malloc(this->num_allocated_elements() * sizeof(T), pinned);
  this->_pinned = pinned;
  this->_shifted_buffer = this->_buffer + this->_gx * this->_pny + this->_gy;

  if (!this->_buffer) {
    printf("[ERROR] Grid2DHost::init - memory allocation failed\n");
    return false;
  }

  return true;
}




// instantiate template functions here
template bool Grid2DHost<float>::init(int nx_val, int ny_val, int gx_val, int gy_val, bool pinned, int padx, int pady);
template bool Grid2DHost<int>::init(int nx_val, int ny_val, int gx_val, int gy_val, bool pinned, int padx, int pady);
template bool Grid2DHost<double>::init(int nx_val, int ny_val, int gx_val, int gy_val, bool pinned, int padx, int pady);

template Grid2DHost<float>::~Grid2DHost();
template Grid2DHost<int>::~Grid2DHost();
template Grid2DHost<double>::~Grid2DHost();

template bool Grid2DHost<float>::copy_interior_data(const Grid2DHost<float> &from);
template bool Grid2DHost<int>::copy_interior_data(const Grid2DHost<int> &from);
template bool Grid2DHost<double>::copy_interior_data(const Grid2DHost<double> &from);

template bool Grid2DHost<float>::copy_all_data(const Grid2DHost<float> &from);
template bool Grid2DHost<int>::copy_all_data(const Grid2DHost<int> &from);
template bool Grid2DHost<double>::copy_all_data(const Grid2DHost<double> &from);

template void Grid2DHost<float>::clear_zero();
template void Grid2DHost<int>::clear_zero();
template void Grid2DHost<double>::clear_zero();

template void Grid2DHost<float>::clear(float t);
template void Grid2DHost<int>::clear(int t);
template void Grid2DHost<double>::clear(double t);

template bool Grid2DHost<float>::linear_combination(float alpha1, const Grid2DHost<float> &g1);
template bool Grid2DHost<int>::linear_combination(int alpha1, const Grid2DHost<int> &g1);
template bool Grid2DHost<double>::linear_combination(double alpha1, const Grid2DHost<double> &g1);

template bool Grid2DHost<float>::linear_combination(float alpha1, const Grid2DHost<float> &g1, float alpha2, const Grid2DHost<float> &g2);
template bool Grid2DHost<int>::linear_combination(int alpha1, const Grid2DHost<int> &g1, int alpha2, const Grid2DHost<int> &g2);
template bool Grid2DHost<double>::linear_combination(double alpha1, const Grid2DHost<double> &g1, double alpha2, const Grid2DHost<double> &g2);

} // end namespace

