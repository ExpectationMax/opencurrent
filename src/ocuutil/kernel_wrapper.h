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

#ifndef __OCU_UTIL_KERNEL_WRAPPER_H__
#define __OCU_UTIL_KERNEL_WRAPPER_H__

#include <vector>

#include <vector_types.h>
#include "ocuutil/timer.h"

namespace ocu {



class KernelWrapper {

public:

  enum KernelType {
    KT_CPU,
    KT_GPU,
    KT_DTOH,
    KT_HTOD,
    KT_HTOH,
    KT_DTOD
  };

  enum TimingMode {
    TM_OFF = 0,
    TM_GPU = 0x1,
    TM_CPU = 0x2,
  };

private:

  GPUTimer     _gpu_timer;
  CPUTimer     _cpu_timer;
  unsigned int _timing_mode;
  KernelType   _kt;
  std::vector<int> _deps;
  int          _id;
  dim3         _grid;
  dim3         _block;
  int          _bytes;

  static int s_last_id;
  static int s_next_id;
  static int s_logging_enabled;
  static std::ofstream s_output;

  static void CheckTraceFileOpen();

  void WriteTraceFile(const char *name, double usecs);

public:

  KernelWrapper(KernelType kt=KT_GPU);

  void PreKernel();
  bool PostKernel(const char *kernel_name);
  bool PostKernel(const char *kernel_name, int resolution);
  bool PostKernelDim(const char *kernel_name, dim3 grid, dim3 block);
  bool PostKernelDim(const char *kernel_name, dim3 grid, dim3 block, int resolution);
  bool PostKernelBytes(const char *kernel_name, int bytes);

  void ToggleCPUTiming(bool onoff) {
    if (onoff)
      _timing_mode |= TM_CPU;
    else
      _timing_mode &= ~TM_CPU;
  }

  void ToggleGPUTiming(bool onoff) {
    if (onoff)
      _timing_mode |= TM_GPU;
    else
      _timing_mode &= ~TM_GPU;

  }

  void AddDependency(int id) { _deps.push_back(id); }
  void MakeIndependent()     { _deps.clear(); }

  static int GetLastId() { return s_last_id; }

  static void DisableLogging();
  static void EnableLogging();
};





} // end namespace

#endif

