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

#include "ocustorage/grid3d.h"

namespace ocu {

void Grid3DDimension::init(int nx_val, int ny_val, int nz_val, int gx_val, int gy_val, int gz_val)
{
  // Perform calculation for padding, even though don't actually allocate any memory.
  this->_nx = nx_val;
  this->_ny = ny_val;
  this->_nz = nz_val;

  this->_gx = gx_val;
  this->_gy = gy_val;
  this->_gz = gz_val;
  
  // pad with ghost cells
  this->_pnx = this->_nx + 2 * gx_val;
  this->_pny = this->_ny + 2 * gy_val;
  this->_pnz = this->_nz + 2 * gz_val;

  // either shift by 4 bits (float and int) or 3 bits (double)
  // then the mask is either 15 or 7.
  //int shift_amount = (sizeof(T) == 4 ? 4 : 3);
  int shift_amount = 4;
  int mask = (0x1 << shift_amount) - 1;

  // round up pnz to next multiple of 16 if needed
  if (this->_pnz & mask)
    this->_pnz = ((this->_pnz >> shift_amount) + 1) << shift_amount;

  this->_pnzpny = this->_pnz * this->_pny;
  this->_allocated_elements = this->_pnzpny * this->_pnx;
  this->_shift_amount   = this->_gx * this->_pnzpny + this->_gy * this->_pnz + this->_gz; 
}

void Grid3DDimension::pad_for_congruence(std::vector<Grid3DDimension> &grids)
{
  // get the max values for gx,gy,gz,nx,ny,nz
  int max_nx=0, max_ny=0, max_nz=0;
  int max_gx=0, max_gy=0, max_gz=0;

  int i;
  for (i=0; i < grids.size(); i++) {
    max_nx = std::max(max_nx, grids[i].nx());
    max_ny = std::max(max_nx, grids[i].ny());
    max_nz = std::max(max_nx, grids[i].nz());
    max_gx = std::max(max_gx, grids[i].gx());
    max_gy = std::max(max_gx, grids[i].gy());
    max_gz = std::max(max_gx, grids[i].gz());
  }

  for (i=0; i < grids.size(); i++) {
    grids[i]._pnx = (max_nx + 2 * max_gx);
    grids[i]._pny = (max_ny + 2 * max_gy);
    grids[i]._pnz = (max_nz + 2 * max_gz);
  }
}


} // end namespace

