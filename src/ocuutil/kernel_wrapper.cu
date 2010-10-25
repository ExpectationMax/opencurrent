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

#include <fstream>
#include <cstdio>

#include "ocuutil/defines.h"
#include "ocuutil/timing_pool.h"
#include "ocuutil/kernel_wrapper.h"
#include "ocuutil/thread.h"

namespace ocu {

int KernelWrapper::s_last_id = -1;
int KernelWrapper::s_next_id = 0;
int KernelWrapper::s_logging_enabled = 0;
std::ofstream KernelWrapper::s_output;

void KernelWrapper::CheckTraceFileOpen()
{
  if (!s_output.is_open()) {
    char buff[1024];
    sprintf(buff, "event_log.%d.yml", ThreadManager::this_image());
    s_output.open(buff);
    s_output << "tasks:\n\n";
  }
}

void 
KernelWrapper::WriteTraceFile(const char *name, double usecs)
{
  if (!s_logging_enabled)
    return;

  CheckTraceFileOpen();
  
  // GpuKernel, CpuFunction, MemcpyD2H, MemcpyH2D, MemcpyD2D, MemcpyH2H, MemcpyHtoDasync, MemcpyDtoHasync
  
  s_output << "  - id: " << _id << std::endl;
  s_output << "    name: " << name << std::endl;
  s_output << "    type: ";
  switch(_kt) {
    case KT_CPU: s_output << "CpuFunction" << std::endl; break;
    case KT_GPU: s_output << "GpuKernel" << std::endl; break;
    case KT_DTOH: s_output << "MemcpyD2H" << std::endl; break;
    case KT_HTOD: s_output << "MemcpyH2D" << std::endl; break;
    case KT_DTOD: s_output << "MemcpyD2D" << std::endl; break;
    case KT_HTOH: s_output << "MemcpyH2H" << std::endl; break;
  }
  s_output << "    duration: " << usecs << std::endl;

  if (_kt == KT_GPU) {
    s_output << "    gridDim: [ " << _grid.x << ", " << _grid.y << " ]" << std::endl;
    s_output << "    blockDim: [ " << _block.x << ", " << _block.y << ", " << _block.z << " ]" << std::endl;
  }
  if (_kt == KT_DTOH || _kt == KT_HTOD || _kt == KT_HTOH || _kt == KT_DTOD) {
    s_output << "    transferBytes: " << _bytes << std::endl;
  }

  if (!_deps.empty()) {
    s_output << "    dependencies: [ ";
    for (int i=0; i < _deps.size(); i++) {
      if (i > 0) s_output << ", ";
      s_output << _deps[i];
    }
    s_output << " ]" << std::endl;
  }

  s_output << std::endl;

}


KernelWrapper::KernelWrapper(KernelType kt) 
{
  _id = -1;
  _kt = kt;

  if (_kt == KT_CPU || _kt == KT_HTOH) {
    _timing_mode = TM_CPU;
  }
  else {
#ifdef OCU_ENABLE_GPU_TIMING_BY_DEFAULT
    _timing_mode = TM_GPU;
#else
    _timing_mode = TM_CPU;
#endif
  }

  _grid = dim3(1,1);
  _block = dim3(1,1,1);
  _bytes = 0;

}


void KernelWrapper::PreKernel() 
{
  _id = s_next_id++;
  MakeIndependent();

  if (GetLastId() >= 0)
    AddDependency(s_last_id);

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
  s_last_id = _id;

  if (_timing_mode & TM_GPU) {
    _gpu_timer.stop();
    global_timer_add_timing(kernel_name, _gpu_timer.elapsed_ms());
    WriteTraceFile(kernel_name, _gpu_timer.elapsed_ms()*1000.0);
  }
  if (_timing_mode & TM_CPU) {
    if (_kt != KT_CPU || _kt != KT_HTOH) {
      cudaThreadSynchronize();
    }

    _cpu_timer.stop();
    char buff[4096];
    sprintf(buff, "%sCPU", kernel_name);
    global_timer_add_timing(buff, _cpu_timer.elapsed_ms());

    // only write to file if we didn't already above
    if (! (_timing_mode & TM_GPU) )
      WriteTraceFile(kernel_name, _cpu_timer.elapsed_ms()*1000.0);
  }

  cudaError_t er = cudaGetLastError();
  if (er != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] %s - CUDA error \"%s\"\n", kernel_name, cudaGetErrorString(er));
    return false;
  }

  return true;
}

bool KernelWrapper::PostKernelDim(const char *kernel_name, dim3 grid, dim3 block, int res)
{
  char buff[4096];
  sprintf(buff, "%s(%d)", kernel_name, res);
  return PostKernelDim(buff, grid, block);
}

bool KernelWrapper::PostKernelDim(const char *kernel_name, dim3 grid, dim3 block)
{
  _grid = grid;
  _block = block;
  return PostKernel(kernel_name);
}

bool KernelWrapper::PostKernelBytes(const char *kernel_name, int bytes)
{
  _bytes = bytes;
  return PostKernel(kernel_name);
}

void KernelWrapper::DisableLogging()
{
  s_logging_enabled = 0;
}

void KernelWrapper::EnableLogging()
{
  s_logging_enabled = 1;
  s_last_id = -1;
}

} // end namespace

