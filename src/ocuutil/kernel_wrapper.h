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

#include "ocuutil/timer.h"

namespace ocu {



class KernelWrapper {

public:

  enum TimingMode {
    TM_OFF = 0,
    TM_GPU = 0x1,
    TM_CPU = 0x2,
  };

private:

  GPUTimer     _gpu_timer;
  CPUTimer     _cpu_timer;
  unsigned int _timing_mode;

public:

  void PreKernel();
  bool PostKernel(const char *kernel_name);
  bool PostKernel(const char *kernel_name, int resolution);

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

public:

  KernelWrapper();
};





} // end namespace

#endif

