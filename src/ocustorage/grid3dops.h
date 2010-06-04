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

#ifndef __OCU_STORAGE_GRID3D_OPS_H__
#define __OCU_STORAGE_GRID3D_OPS_H__

#include <vector>
#include "ocuutil/imagefile.h"
#include "ocustorage/grid3d.h"



namespace ocu {



template<typename T, typename REDUCE>
bool 
reduce_with_operator(
  const ocu::Grid3DDevice<T> &grid, T &result, REDUCE reduce);

void 
plot_scalar_value(
  const Grid3DHostF &grid, std::vector<ImageFile> &slices);

bool 
check_valid_mac_dimensions(const Grid3DUntyped &u, const Grid3DUntyped &v, const Grid3DUntyped &w, int nx, int ny, int nz);





} // end namespace

#endif

