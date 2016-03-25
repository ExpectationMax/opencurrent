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

#ifndef __OCU_STORAGE_GRID3D_SAMPLE_H__
#define __OCU_STORAGE_GRID3D_SAMPLE_H__

#include "ocuutil/boundary_condition.h"
#include "ocustorage/grid1d.h"
#include "ocustorage/grid3d.h"


namespace ocu {

template<typename T>
bool
sample_points_3d(
  Grid1DDevice<T> &phi_sampled,
  const Grid1DDevice<float> &position_x, const Grid1DDevice<float> &position_y, const Grid1DDevice<float> &position_z,
  const Grid3DDevice<T> &phi,
  const BoundaryConditionSet &bc,
  float period_nx, float period_ny, float period_nz, 
  float hx, float hy, float hz);


template<typename T>
bool
sample_points_mac_grid_3d(
  Grid1DDevice<T> &vx_sampled, Grid1DDevice<T> &vy_sampled, Grid1DDevice<T> &vz_sampled,
  const Grid1DDevice<float> &position_x, const Grid1DDevice<float> &position_y, const Grid1DDevice<float> &position_z,
  const Grid3DDevice<T> &u, const Grid3DDevice<T> &v, const Grid3DDevice<T> &w,
  const BoundaryConditionSet &bc,
  float hx, float hy, float hz);


} // end namespace


#endif

