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

#include <cstdio>
#include "ocustorage/grid3dops.h"
#include "ocuutil/color.h"

namespace ocu {

bool 
check_valid_mac_dimensions(const Grid3DUntyped &u, const Grid3DUntyped &v, const Grid3DUntyped &w, int nx, int ny, int nz)
{
  // u,v,w must be the proper dimensions, i.e. staggered grid
  if (u.nx() != nx+1 || u.ny() != ny || u.nz() != nz) {
    return false;
  }

  if (v.nx() != nx || v.ny() != ny+1 || v.nz() != nz) {
    return false;
  }

  if (w.nx() != nx || w.ny() != ny || w.nz() != nz+1) {
    return false;
  }

  // u,v,w must all share the same memory layout.  This is a cuda optimization to simplify indexing.
  if (!u.check_layout_match(v) || !u.check_layout_match(w)) {
    return false;
  }

  return true;
}




void 
plot_scalar_value(const Grid3DHostF &grid, std::vector<ImageFile> &slices)
{
  slices.clear();
  slices.resize(grid.nz());

  int nx = grid.nx();
  int ny = grid.ny();
  int nz = grid.nz();

  float max_val;
  float min_val;
  grid.reduce_max(max_val);
  grid.reduce_min(min_val);

  // map to color wheel from min to max

  float delta = max_val - min_val;

  for (int k=0; k < nz; k++) {
    // do each slice
    ImageFile &img = slices[k];
    img.allocate(nx, ny);
    
    for (int i=0; i < nx; i++)
      for (int j=0; j < ny; j++) {
        float3 color = hsv_to_rgb(make_float3(270 * (grid.at(i,j,k) - min_val) / delta,1,1));
        img.set_rgb(i,j,(unsigned char)(color.x * 255), (unsigned char) (color.y * 255), (unsigned char ) (color.z * 255));
      }
  }
}


} // end namespace

