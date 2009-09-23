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

#ifndef __OCU_UTIL_FLOAT_ROUTINES_H__
#define __OCU_UTIL_FLOAT_ROUTINES_H__

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include <cmath>
#include <float.h>
#endif

#include "ocuutil/defines.h"

namespace ocu {



OCU_HOSTDEVICE 
inline bool check_float(float f) 
{
#ifdef __CUDACC__
  return __finite(f);
#elif defined(_WIN32)
  return _finite(f);
#else
  return finite(f);
#endif
}

template<typename T>
OCU_HOSTDEVICE
T min3(T a, T b, T c)
{
  return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

template<typename T>
OCU_HOSTDEVICE
T max3(T a, T b, T c)
{
  return a > b ? (a > c ? a : c) : (b > c ? b : c);
}


// Return a smooth value in [0,1], where the transition from 0
// to 1 takes place for values of x in [edge0,edge1].
template<typename T>
OCU_HOSTDEVICE
inline T smoothstep( T edge0, T edge1, T x )
{
  T t = (x-edge0) / (edge1-edge0);
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  return t*t * ( (T)3.0 - (T)2.0*t );
}



} // end namespace




#endif

