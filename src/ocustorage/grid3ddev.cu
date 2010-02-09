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

#include "ocuutil/reduction_op.h"
#include "ocustorage/grid3d.h"
#include "ocustorage/grid3dops.h"
#include "ocuutil/kernel_wrapper.h"
#include "ocustorage/coarray.h"

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

template<>
bool Grid3DDevice<int>::reduce_checknan(int &result) const
{
  printf("[WARNING] Grid3DDevice<int>::reduce_checknan - operation not supported for 'int' types\n");
  return false;
}


template<typename T>
bool Grid3DDevice<T>::clear_zero()
{
  KernelWrapper wrapper;
  wrapper.PreKernel();
  cudaMemset(this->_buffer, 0, this->num_allocated_elements() * sizeof(T));
  return wrapper.PostKernel("cudaMemset");
}

template<typename T>
bool Grid3DDevice<T>::clear(T val)
{
  dim3 Dg((this->num_allocated_elements()+255) / 256);
  dim3 Db(256);
  
  KernelWrapper wrapper;
  wrapper.PreKernel();
  Grid3DDevice_clear<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(this->buffer(), val, this->num_allocated_elements());
  return wrapper.PostKernel("Grid3DDevice_clear");
}


template<typename T>
bool 
Grid3DDevice<T>::copy_all_data(const Grid3DHost<T> &from)
{
  if (!this->check_layout_match(from)) {
    printf("[ERROR] Grid3DDevice::copy_all_data - mismatch: (%d, %d, %d) != (%d, %d, %d)\n", this->pnx(),this->pny(),this->pnz(), from.pnx(), from.pny(),from.pnz());
    return false;
  }

  KernelWrapper wrapper;
  wrapper.PreKernel();
  cudaMemcpy(this->buffer(), from.buffer(), sizeof(T) * this->num_allocated_elements(), cudaMemcpyHostToDevice);
  return wrapper.PostKernel("cudaMemcpy(HostToDevice)");
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
  
  KernelWrapper wrapper;
  wrapper.PreKernel();
  Grid3DDevice_copy_all_data<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(this->buffer(), from.buffer(), this->num_allocated_elements());
  return wrapper.PostKernel("Grid3DDevice_copy_all_data");
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

  KernelWrapper wrapper;
  wrapper.PreKernel();
  cudaMemcpy(this->buffer(), from.buffer(), sizeof(float) * this->num_allocated_elements(), cudaMemcpyDeviceToDevice);
  return wrapper.PostKernel("cudaMemcpy(DeviceToDevice)");
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

  KernelWrapper wrapper;
  wrapper.PreKernel();
  cudaMemcpy(this->buffer(), from.buffer(), sizeof(int) * this->num_allocated_elements(), cudaMemcpyDeviceToDevice);
  return wrapper.PostKernel("cudaMemcpy(DeviceToDevice)");
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

  KernelWrapper wrapper;
  wrapper.PreKernel();
  cudaMemcpy(this->buffer(), from.buffer(), sizeof(double) * this->num_allocated_elements(), cudaMemcpyDeviceToDevice);
  return wrapper.PostKernel("cudaMemcpy(DeviceToDevice)");
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
  
  KernelWrapper wrapper;
  wrapper.PreKernel();
  Grid3DDevice_linear_combination1<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(this->buffer(), alpha1, g1.buffer(), this->num_allocated_elements());
  return wrapper.PostKernel("Grid3DDevice_linear_combination");
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

    KernelWrapper wrapper;
    wrapper.PreKernel();
    Grid3DDevice_linear_combination2_3D<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(&this->at(0,0,0), alpha1, &g1.at(0,0,0), alpha2, &g2.at(0,0,0), 
      this->xstride(), this->ystride(), this->nx(), this->ny(), this->nz(), 
      blocksInY, 1.0f / (float)blocksInY);
    return wrapper.PostKernel("Grid3DDevice_linear_combination2_3D");
  }
  else {
    int block_size = 512;
    dim3 Dg((this->num_allocated_elements() + block_size - 1) / block_size);
    dim3 Db(block_size);
    
    KernelWrapper wrapper;
    wrapper.PreKernel();
    Grid3DDevice_linear_combination2<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(this->buffer(), alpha1, g1.buffer(), alpha2, g2.buffer(), this->num_allocated_elements());
    return wrapper.PostKernel("Grid3DDevice_linear_combination2_3D");
  }

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
  this->_shifted_buffer = this->buffer() + this->_shift_amount;

  return true;
}

template<typename T>
Grid3DDeviceCo<T>::Grid3DDeviceCo(const char *id) 
{
  this->_table = CoArrayManager::register_coarray(id, this->_image_id, this);
}

template<typename T>
Grid3DDeviceCo<T>::~Grid3DDeviceCo() 
{
  CoArrayManager::unregister_coarray(this->_table->name, this->_image_id);
}

template<typename T>
Grid3DDeviceCo<T> *Grid3DDeviceCo<T>::co(int image)
{ 
  return (Grid3DDeviceCo<T> *)(this->_table->table[image]); 
}

template<typename T>
const Grid3DDeviceCo<T> *Grid3DDeviceCo<T>::co(int image) const 
{ 
  return (const Grid3DDeviceCo<T> *)(this->_table->table[image]); 
}

template<>
bool Grid3DDeviceCo<float>::co_reduce_maxabs(float &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsF()); 
  return ok;
}

template<>
bool Grid3DDeviceCo<int>::co_reduce_maxabs(int &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsI()); 
  return ok;
}

#ifdef OCU_DOUBLESUPPORT 
template<>
bool Grid3DDeviceCo<double>::co_reduce_maxabs(double &result) const
{
  bool ok = reduce_maxabs(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMaxAbsD()); 
  return ok;
}
#endif

template<typename T>
bool Grid3DDeviceCo<T>::co_reduce_sum(T &result) const
{
  bool ok = reduce_sum(result);
  result = ThreadManager::barrier_reduce(result, HostReduceSum<T>()); 
  return ok;
}

template<typename T>
bool Grid3DDeviceCo<T>::co_reduce_sqrsum(T &result) const
{
  bool ok = reduce_sqrsum(result);
  result = ThreadManager::barrier_reduce(result, HostReduceSum<T>()); 
  return ok;
}

template<typename T>
bool Grid3DDeviceCo<T>::co_reduce_max(T &result) const
{
  bool ok = reduce_max(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMax<T>()); 
  return ok;
}

template<typename T>
bool Grid3DDeviceCo<T>::co_reduce_min(T &result) const
{
  bool ok = reduce_min(result);
  result = ThreadManager::barrier_reduce(result, HostReduceMin<T>()); 
  return ok;
}

template<typename T>
bool Grid3DDeviceCo<T>::co_reduce_checknan(T &result) const
{
  bool ok = reduce_checknan(result);
  result = ThreadManager::barrier_reduce(result, HostReduceCheckNan<T>()); 
  return ok;
}

template<>
bool Grid3DDeviceCo<int>::co_reduce_checknan(int &result) const
{
  printf("[WARNING] Grid3DDeviceCo<int>::reduce_checknan - operation not supported for 'int' types\n");
  return false;
}


template class Grid3DDevice<float>;
template class Grid3DDevice<int>;
template class Grid3DDeviceCo<float>;
template class Grid3DDeviceCo<int>;

// because these are doubly-templated, they need to be explicitly instantiated
template bool Grid3DDevice<float>::copy_all_data(const Grid3DDevice<int> &from);
template bool Grid3DDevice<float>::copy_all_data(const Grid3DDevice<double> &from);
template bool Grid3DDevice<int>::copy_all_data(const Grid3DDevice<float> &from);
template bool Grid3DDevice<int>::copy_all_data(const Grid3DDevice<double> &from);

#ifdef OCU_DOUBLESUPPORT

template class Grid3DDevice<double>;
template class Grid3DDeviceCo<double>;

// because these are doubly-templated, they need to be explicitly instantiated
template bool Grid3DDevice<double>::copy_all_data(const Grid3DDevice<float> &from);
template bool Grid3DDevice<double>::copy_all_data(const Grid3DDevice<int> &from);

#endif // OCU_DOUBLESUPPORT

} // end namespace

