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

#ifndef __OCU_STORAGE_REGION_1D_H__
#define __OCU_STORAGE_REGION_1D_H__

#include "ocuutil/memory.h"

namespace ocu {

class Grid1DBase;

class ConstRegion1D {
protected:
  Grid1DBase *_grid;
  MemoryType  _memtype; // mem space where this pointer valid
  int _imageid;

public:
  // maybe these should be private... need to think about this.
  int x0,x1;

  ConstRegion1D(int x0_val, int x1_val, const Grid1DBase *grid_val, MemoryType memtype_val, int imageid_val) :
    x0(x0_val), x1(x1_val), _grid(const_cast<Grid1DBase *>(grid_val)), _memtype(memtype_val), _imageid(imageid_val) { }
  
  ConstRegion1D() {
    _grid = 0; x0 = x1 = 0; _memtype = MEM_INVALID; _imageid = -1;
  }

  int imageid()    const { return _imageid; }
  int volume()     const { return (x1-x0+1); }
  bool is_valid()  const { return volume() > 0 && _grid != 0 && _memtype != MEM_INVALID; }
  MemoryType memtype() const { return _memtype; }
  const Grid1DBase *grid() const { return _grid; }
  bool is_inside(int x) const { return (x0 <= x) && (x <= x1); }
};

class Region1D : public ConstRegion1D {
public:

  Region1D(int x0_val, int x1_val, const Grid1DBase *grid_val, MemoryType memtype_val, int imageid_val) :
    ConstRegion1D(x0_val, x1_val, grid_val, memtype_val, imageid_val) { }  

  Region1D() : ConstRegion1D() {}

  Grid1DBase *grid() const { return _grid; }
};


} // end namespace

#endif

