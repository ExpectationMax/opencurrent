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
#include "ocuutil/timer.h"
#include "ocuutil/timing_pool.h"

namespace ocu {



template<typename T>
bool 
Grid2DHost<T>::copy_all_data(const Grid2DDevice<T> &from)
{
  if (!this->check_layout_match(from)) {
    printf("[ERROR] Grid3DHost::copy_all_data - mismatch: (%d, %d) != (%d, %d)\n", this->pnx(), this->pny(), from.pnx(),  from.pny());
    return false;
  }

  GPUTimer timer;
  timer.start();
  if ((unsigned int)CUDA_SUCCESS != cudaMemcpy(this->_buffer, from.buffer(), sizeof(T) * this->num_allocated_elements(), cudaMemcpyDeviceToHost)) {
    printf("[ERROR] Grid2DHost::copy_all_data - cudaMemcpy failed\n");
    return false;
  }
  timer.stop();
  global_timer_add_timing("cudaMemcpy(DeviceToHost)", timer.elapsed_ms());

  return true;
}




template bool Grid2DHost<float>::copy_all_data(const Grid2DDevice<float> &from);
template bool Grid2DHost<int>::copy_all_data(const Grid2DDevice<int> &from);
template bool Grid2DHost<double>::copy_all_data(const Grid2DDevice<double> &from);



} // end namespace

