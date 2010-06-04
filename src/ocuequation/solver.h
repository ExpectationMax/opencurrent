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

#ifndef __OCU_EQUATION_SOLVER_H__
#define __OCU_EQUATION_SOLVER_H__

#include "ocuutil/kernel_wrapper.h"
#include "ocuequation/error_handler.h"

namespace ocu {



class Solver : public ErrorHandler {

private:

  KernelWrapper _wrapper;

protected:

  void PreKernel()                          { _wrapper.PreKernel(); }
  bool PostKernel(const char *kernel_name);
  bool PostKernel(const char *kernel_name, int resolution);

  void ToggleCPUTiming(bool onoff)          { _wrapper.ToggleCPUTiming(onoff); }
  void ToggleGPUTiming(bool onoff)          { _wrapper.ToggleGPUTiming(onoff); }

public:

  Solver();
  virtual ~Solver();

};





} // end namespace

#endif

