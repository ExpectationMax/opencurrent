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

#include "ocustorage/grid1d.h"


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







template bool Grid1DHost<float>::copy_interior_data(const Grid1DDevice<float> &from);
template bool Grid1DHost<int>::copy_interior_data(const Grid1DDevice<int> &from);
template bool Grid1DHost<double>::copy_interior_data(const Grid1DDevice<double> &from);

template bool Grid1DHost<float>::copy_all_data(const Grid1DDevice<float> &from);
template bool Grid1DHost<int>::copy_all_data(const Grid1DDevice<int> &from);
template bool Grid1DHost<double>::copy_all_data(const Grid1DDevice<double> &from);



} // end namespace

