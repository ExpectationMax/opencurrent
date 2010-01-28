#  Copyright 2008-2009 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License

CUDA_ADD_LIBRARY(ocuutil 
  defines.h
  timing_pool.h
  timer.h
  timing_pool.cpp
  imagefile.h
  imagefile.cpp
  float_routines.h
  boundary_condition.h
  timer.cu
  memory.h
  memory.cu
  convergence.h
  reduction_op.h
  color.h
  interpolation.h
  timestep.h
  direction.h
  kernel_wrapper.h
  kernel_wrapper.cu
) 


