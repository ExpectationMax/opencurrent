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

#ifndef __OCU_UTIL_REDUCTION_OPS_H__
#define __OCU_UTIL_REDUCTION_OPS_H__

#include "ocuutil/defines.h"
#include "ocuutil/float_routines.h"

#ifdef __CUDACC__

#include "cuda.h"
#include <math_constants.h>

namespace ocu {

#ifdef OCU_DOUBLESUPPORT

struct ReduceDevMaxAbsD {
  OCU_HOSTDEVICE double reduce(double a, double b) const { return fmax(a,b); }
  OCU_HOSTDEVICE double process(double a) const { return fabs(a); }
  OCU_HOSTDEVICE double identity() const { return 0; }
};

struct ReduceDevMaxD {
  OCU_HOSTDEVICE double reduce(double a, double b) const { return fmax(a,b); }
  OCU_HOSTDEVICE double process(double a) const { return a; }
  OCU_HOSTDEVICE double identity() const { return -1e100; }
};

struct ReduceDevMinD {
  OCU_HOSTDEVICE double reduce(double a, double b) const { return fmin(a,b); }
  OCU_HOSTDEVICE double process(double a) const { return a; }
  OCU_HOSTDEVICE double identity() const { return 1e100; }
};

#endif // OCU_DOUBLESUPPORT

struct ReduceDevMaxAbsF {
  OCU_HOSTDEVICE float reduce(float a, float b) const { return fmaxf(a,b); }
  OCU_HOSTDEVICE float process(float a) const { return fabsf(a); }
  OCU_HOSTDEVICE float identity() const { return 0; }
};

struct ReduceDevMaxAbsI {
  OCU_HOSTDEVICE int reduce(int a, int b) const { return max(a,b); }
  OCU_HOSTDEVICE int process(int a) const { return abs(a); }
  OCU_HOSTDEVICE int identity() const { return 0; }
};

struct ReduceDevMaxF {
  OCU_HOSTDEVICE float reduce(float a, float b) const { return fmaxf(a,b); }
  OCU_HOSTDEVICE float process(float a) const { return a; }
  OCU_HOSTDEVICE float identity() const { return -CUDART_NORM_HUGE_F; }
};

struct ReduceDevMaxI {
  OCU_HOSTDEVICE int reduce(int a, int b) const { return a > b ? a : b; }
  OCU_HOSTDEVICE int process(int a) const { return a; }
  OCU_HOSTDEVICE int identity() const { return -INT_MAX; }
};

struct ReduceDevMinF {
  OCU_HOSTDEVICE float reduce(float a, float b) const { return fminf(a,b); }
  OCU_HOSTDEVICE float process(double a) const { return a; }
  OCU_HOSTDEVICE float identity() const { return CUDART_NORM_HUGE_F; }
};

struct ReduceDevMinI {
  OCU_HOSTDEVICE int reduce(int a, int b) const { return a < b ? a : b; }
  OCU_HOSTDEVICE int process(int a) const { return a; }
  OCU_HOSTDEVICE int identity() const { return INT_MAX; }
};


template<typename T>
struct ReduceDevSum {
  OCU_HOSTDEVICE T reduce(T a, T b) const { return a + b; }
  OCU_HOSTDEVICE T process(T a) const { return a; }
  OCU_HOSTDEVICE T identity() const { return 0; }
};

template<typename T>
struct ReduceDevSqrSum {
  OCU_HOSTDEVICE T reduce(T a, T b) const { return a+b; }
  OCU_HOSTDEVICE T process(T a) const { return a*a; }
  OCU_HOSTDEVICE T identity() const { return 0; }
};

template<typename T>
struct ReduceDevCheckNan {
  OCU_HOSTDEVICE T reduce(T a, T b) const { return check_float(a) ? b : a; }
  OCU_HOSTDEVICE T process(T a) const { return a; }
  OCU_HOSTDEVICE T identity() const { return 0; }
};


} // end namespace

#else // not __CUDACC__

#include <cmath>
#include <algorithm>

namespace ocu {

struct ReduceMaxAbsF
{
  float operator()(float a, float b) const { return std::max(fabsf(a), b); }
};

struct ReduceMaxAbsD
{
  double operator()(double a, double b) const { return std::max(fabs(a), b); }
};

struct ReduceMaxAbsI
{
  int operator()(int a, int b) const { return std::max(abs(a), b); }
};

template<typename T>
struct ReduceMax
{
  T operator()(T a, T b) const { return std::max(a, b); }
};

template<typename T>
struct ReduceMin
{
  T operator()(T a, T b) const { return std::min(a, b); }
};


template<typename T>
struct ReduceSum
{
  T operator()(T a, T b) const { return a+b; }
};

template<typename T>
struct ReduceSqrSum
{
  T operator()(T a, T b) const { return (a*a)+b; }
};

template<typename T>
struct ReduceCheckNan
{
  T operator()(T a, T b) const { return check_float(a) ? b : a; }
};


} // end namespace

#endif // __CUDACC__


#endif

