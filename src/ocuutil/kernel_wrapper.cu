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
#include "ocuutil/defines.h"
#include "ocuutil/timing_pool.h"
#include "ocuutil/kernel_wrapper.h"

namespace ocu {


KernelWrapper::KernelWrapper() 
{
#ifdef OCU_ENABLE_GPU_TIMING_BY_DEFAULT
  _timing_mode = TM_GPU;
#else
  _timing_mode = TM_CPU;
#endif
}


void KernelWrapper::PreKernel() 
{
  if (_timing_mode & TM_CPU) {
    _cpu_timer.start();
  }
  if (_timing_mode & TM_GPU) {
    _gpu_timer.start();
  }
}


bool KernelWrapper::PostKernel(const char *kernel_name, int resolution)
{
  char buff[4096];
  sprintf(buff, "%s(%d)", kernel_name, resolution);
  return PostKernel(buff);
}

bool KernelWrapper::PostKernel(const char *kernel_name) 
{

  if (_timing_mode & TM_GPU) {
    _gpu_timer.stop();
    global_timer_add_timing(kernel_name, _gpu_timer.elapsed_ms());
  }
  if (_timing_mode & TM_CPU) {
    _cpu_timer.stop();
    char buff[4096];
    sprintf(buff, "%sCPU", kernel_name);
    global_timer_add_timing(buff, _cpu_timer.elapsed_ms());
  }

  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] %s - CUDA error \"%s\"\n", kernel_name, cudaGetErrorString(er));
    return false;
  }

  return true;
}


} // end namespace

