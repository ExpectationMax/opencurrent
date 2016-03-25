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

#ifndef __OCU_UTIL_INTERPOLATION_H__
#define __OCU_UTIL_INTERPOLATION_H__

#include "ocuutil/defines.h"


namespace ocu {


enum InterpolationType {
  IT_ERROR = 0,
  IT_FIRST_ORDER_UPWIND,
  IT_SECOND_ORDER_CENTERED,
  IT_THIRD_ORDER_VAN_LEER_FLUX_LIMITED
};



template<typename T>
struct InterpolatorFirstOrderUpwind {
  OCU_HOSTDEVICE T operator()(T u, T x0, T x1) {
    return u < 0 ? x1 : x0;
  }
};


template<typename T>
struct InterpolatorSecondOrderCentered {
  OCU_HOSTDEVICE T operator()(T u, T x0, T x1) {
    return ((T).5) * (x1 + x0);
  }
};


template<typename T>
struct InterpolatorThirdOrderVanLeerFluxLimited {
  OCU_HOSTDEVICE T operator()(T u, T x0, T x1, T x2, T x3) {
    // interpolated value half-wayf between x1 and x2
    T r = (u > 0) ? (x2 - x1 + ((T)1e-7)) / (x1 - x0 + ((T)1e-7)) :
                    (x1 - x2 + ((T)1e-7)) / (x2 - x3 + ((T)1e-7));
    T psi = ((T).6666666666666666) * r + ((T).333333333333333); // smoothness measure
    T abs_psi = psi < 0 ? psi : -psi;
    T psi_limited = ( psi + abs_psi ) / ( 1 + abs_psi ); // van Leer limiter
    return (u > 0) ? x1 + ((T).5) * psi_limited * (x1 - x0) :
                     x2 + ((T).5) * psi_limited * (x2 - x3);
  }
};


} // end namespace

#endif

