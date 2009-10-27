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

#ifndef __OCU_UTIL_CONVERGENCE_H__
#define __OCU_UTIL_CONVERGENCE_H__


namespace ocu {





enum ConvergenceType {
  CONVERGENCE_CALC_L2 = 0x01,
  CONVERGENCE_CALC_LINF = 0x02,
  CONVERGENCE_CRITERIA_L2 = 0x04,
  CONVERGENCE_CRITERIA_LINF = 0x08,
  CONVERGENCE_CRITERIA_NONE = 0x10,

  // Actual Types:

  CONVERGENCE_L2 = CONVERGENCE_CALC_L2 | CONVERGENCE_CRITERIA_L2,
  CONVERGENCE_LINF = CONVERGENCE_CALC_LINF | CONVERGENCE_CRITERIA_LINF,
  CONVERGENCE_NONE = CONVERGENCE_CRITERIA_NONE,
  CONVERGENCE_NONE_CALC_L2LINF = CONVERGENCE_CRITERIA_NONE | CONVERGENCE_CALC_L2 | CONVERGENCE_CALC_LINF,
};




}

#endif

