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
#include "ocuutil/timing_pool.h"
#include "ocuequation/solver.h"

namespace ocu {





Solver::~Solver()
{
}


Solver::Solver() {

}

bool Solver::PostKernelDim(const char *kernel_name, dim3 grid, dim3 block)
{
  if (!_wrapper.PostKernelDim(kernel_name, grid, block)) {
    add_error();
    return false;
  }
  return true;
}

bool Solver::PostKernelDim(const char *kernel_name, dim3 grid, dim3 block, int resolution)
{
  if (!_wrapper.PostKernelDim(kernel_name, grid, block, resolution)) {
    add_error();
    return false;
  }
  return true;
}


bool Solver::PostKernel(const char *kernel_name) {
  if (!_wrapper.PostKernel(kernel_name)) {
    add_error();
    return false;
  }
  return true;
}

bool Solver::PostKernel(const char *kernel_name, int resolution) {
  if (!_wrapper.PostKernel(kernel_name, resolution)) {
    add_error();
    return false;
  }
  return true;
}

} // end namespace
