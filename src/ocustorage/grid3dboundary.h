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

#ifndef __OCU_STORAGE_GRID3D_BOUNDARY_H__
#define __OCU_STORAGE_GRID3D_BOUNDARY_H__

#include "ocuutil/boundary_condition.h"
#include "ocustorage/grid3d.h"


namespace ocu {

template<typename T>
bool 
apply_3d_boundary_conditions_level1(
  Grid3DDevice<T> &grid, 
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template<typename T>
bool 
apply_3d_boundary_conditions_level1_nocorners(
  Grid3DHost<T> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template<typename T>
bool 
apply_3d_boundary_conditions_level1_nocorners(
  Grid3DDevice<T> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

template<typename T>
bool 
apply_3d_mac_boundary_conditions_level1(
  Grid3DDevice<T> &u_grid, Grid3DDevice<T> &v_grid, Grid3DDevice<T> &w_grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);




template<typename T>
bool 
apply_3d_boundary_conditions_level2_nocorners(
  Grid3DHost<T> &grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);


template<typename T>
bool 
apply_3d_mac_boundary_conditions_level2(
  Grid3DDevice<T> &u_grid, Grid3DDevice<T> &v_grid, Grid3DDevice<T> &w_grid,  
  const BoundaryConditionSet &bc,
  double hx, double hy, double hz);

} // end namespace 

#endif
