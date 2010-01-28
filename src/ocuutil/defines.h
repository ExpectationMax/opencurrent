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

#ifndef __OCUUTIL_DEFINES_H__
#define __OCUUTIL_DEFINES_H__

#define OCU_ENABLE_GPU_TIMING_BY_DEFAULT


#ifdef __CUDACC__

#define OCU_HOST __host__
#define OCU_DEVICE __device__
#define OCU_HOSTDEVICE __host__ __device__

#else // __CUDACC__

#define OCU_HOST
#define OCU_DEVICE
#define OCU_HOSTDEVICE

#endif




#endif

