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

#ifndef __OCU_UTIL_BOUNDARY_CONDITION_H__
#define __OCU_UTIL_BOUNDARY_CONDITION_H__

#include "ocuutil/defines.h"


namespace ocu {


enum BoundaryConditionType {
  BC_INVALID,
  BC_PERIODIC,
  BC_DIRICHELET,
  BC_NEUMANN,
  BC_SECOND_DERIV,
  BC_DIRICHELET_AND_NEUMANN,
  BC_FORCED_INFLOW_VARIABLE_SLIP,
  BC_SCALAR_SLIP, // the scalar bc's that apply to the tangential part of a MAC grid
};

struct BoundaryCondition {
  BoundaryConditionType type;  // what value is to be constrained
  float value;                // the constraint value, if applicable.
  float aux_value;            // additional the constraint value, if applicable.

  BoundaryCondition() {
    type = BC_INVALID;
    value = aux_value = 0;
  }

  OCU_HOSTDEVICE  bool check_type(BoundaryConditionType t) const {
    return t == type;
  }

  OCU_HOSTDEVICE  bool check_type(BoundaryConditionType t1, BoundaryConditionType t2) const {
    return t1 == type || t2 == type;
  }

  OCU_HOSTDEVICE  bool check_type(BoundaryConditionType t1, BoundaryConditionType t2, BoundaryConditionType t3) const {
    return t1 == type || t2 == type || t3 == type;
  }

  OCU_HOSTDEVICE  bool check_type(BoundaryConditionType t1, BoundaryConditionType t2, BoundaryConditionType t3, BoundaryConditionType t4) const {
    return t1 == type || t2 == type || t3 == type || t4 == type;
  }
};



struct BoundaryConditionSet {
  //**** PUBLIC STATE ****
  BoundaryCondition xpos, xneg, ypos, yneg, zpos, zneg;

  //**** MANAGER ****
  BoundaryConditionSet() { }
  BoundaryConditionSet(
    const BoundaryCondition &xpos_val, const BoundaryCondition &xneg_val, const BoundaryCondition &ypos_val, 
    const BoundaryCondition &yneg_val, const BoundaryCondition &zpos_val, const BoundaryCondition &zneg_val) : 
      xpos(xpos_val), xneg(xneg_val), ypos(ypos_val), yneg(yneg_val), zpos(zpos_val), zneg(zneg_val) { }
  BoundaryConditionSet(const BoundaryCondition &val) : 
    xpos(val), xneg(val), ypos(val), yneg(val), zpos(val), zneg(val) { }

  //**** PUBLIC INTERFACE ****
  OCU_HOSTDEVICE  bool check_type(BoundaryConditionType t) const {
    return xpos.check_type(t) && xneg.check_type(t) && ypos.check_type(t) && yneg.check_type(t) && zpos.check_type(t) && zneg.check_type(t);
  }

  OCU_HOSTDEVICE  bool check_type(BoundaryConditionType t1, BoundaryConditionType t2) const {
    return xpos.check_type(t1,t2) && xneg.check_type(t1,t2) && ypos.check_type(t1,t2) && yneg.check_type(t1,t2) && zpos.check_type(t1,t2) && zneg.check_type(t1,t2);
  }

  OCU_HOSTDEVICE  bool check_type(BoundaryConditionType t1, BoundaryConditionType t2, BoundaryConditionType t3) const {
    return xpos.check_type(t1,t2,t3) && xneg.check_type(t1,t2,t3) && ypos.check_type(t1,t2,t3) && yneg.check_type(t1,t2,t3) && zpos.check_type(t1,t2,t3) && zneg.check_type(t1,t2,t3);
  }

  OCU_HOSTDEVICE  bool check_type(BoundaryConditionType t1, BoundaryConditionType t2, BoundaryConditionType t3, BoundaryConditionType t4) const {
    return xpos.check_type(t1,t2,t3,t4) && xneg.check_type(t1,t2,t3,t4) && ypos.check_type(t1,t2,t3,t4) && yneg.check_type(t1,t2,t3,t4) && zpos.check_type(t1,t2,t3,t4) && zneg.check_type(t1,t2,t3,t4);
  }

  OCU_HOST void make_homogeneous() {
    xpos.value = 0; 
    xpos.aux_value = 0;
    xneg.value = 0; 
    xneg.aux_value = 0;
    ypos.value = 0; 
    ypos.aux_value = 0;
    yneg.value = 0; 
    yneg.aux_value = 0;
    zpos.value = 0; 
    zpos.aux_value = 0;
    zneg.value = 0; 
    zneg.aux_value = 0;
  }

};







} // end namespace


#endif

