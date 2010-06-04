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

#ifndef __OCU_STORAGE_REGION_3D_H__
#define __OCU_STORAGE_REGION_3D_H__

#include "ocuutil/memory.h"

namespace ocu {

class Grid3DBase;

class ConstRegion3D {
protected:
  Grid3DBase *_grid;
  MemoryType  _memtype; // mem space where this pointer valid
  int _imageid;

public:
  // maybe these should be private... need to think about this.
  int x0,x1;
  int y0,y1;
  int z0,z1;

  ConstRegion3D(int x0_val, int x1_val, int y0_val, int y1_val, int z0_val, int z1_val, const Grid3DBase *grid_val, MemoryType memtype_val, int imageid_val) :
    x0(x0_val), x1(x1_val), y0(y0_val), y1(y1_val), z0(z0_val), z1(z1_val), 
    _grid(const_cast<Grid3DBase *>(grid_val)), _memtype(memtype_val), _imageid(imageid_val) { }
  
  ConstRegion3D() {
    _grid = 0; x0 = x1 = y0 = y1 = z0 = z1 = 0; _memtype = MEM_INVALID; _imageid = -1;
  }

  int imageid()    const { return _imageid; }
  int volume()     const { return (x1-x0+1)*(y1-y0+1)*(z1-z0+1); }
  bool is_valid()  const { return volume() > 0 && _grid != 0 && _memtype != MEM_INVALID; }
  MemoryType memtype() const { return _memtype; }
  const Grid3DBase *grid() const { return _grid; }
  bool is_inside(int x, int y, int z) const { return (x0 <= x) && (x <= x1) && (y0 <= y) && (y <= y1) && (z0 <= z) && (z <= z1); }
};

class ConstRegion3DY : public ConstRegion3D {

public:
  ConstRegion3DY(int x0_val, int x1_val, int y0_val, int y1_val, int z0_val, int z1_val, const Grid3DBase *grid_val, MemoryType memtype_val, int imageid_val) :
    ConstRegion3D(x0_val, x1_val, y0_val, y1_val, z0_val, z1_val, grid_val, memtype_val, imageid_val) { }  

  ConstRegion3D operator()(void)                 const { return ConstRegion3D(x0,x1,y0,y1,z0,z1,      _grid,_memtype,_imageid); }
  ConstRegion3D operator()(int zval)             const { return ConstRegion3D(x0,x1,y0,y1,zval,zval,  _grid,_memtype,_imageid); }
  ConstRegion3D operator()(int zval0, int zval1) const { return ConstRegion3D(x0,x1,y0,y1,zval0,zval1,_grid,_memtype,_imageid); }
};


class ConstRegion3DX : public ConstRegion3D {

public:
  ConstRegion3DX(int x0_val, int x1_val, int y0_val, int y1_val, int z0_val, int z1_val, const Grid3DBase *grid_val, MemoryType memtype_val, int imageid_val) :
    ConstRegion3D(x0_val, x1_val, y0_val, y1_val, z0_val, z1_val, grid_val, memtype_val, imageid_val) { }  

  ConstRegion3DY operator()(void)                 const { return ConstRegion3DY(x0,x1,y0,y1,z0,z1,      _grid,_memtype,_imageid); }
  ConstRegion3DY operator()(int yval)             const { return ConstRegion3DY(x0,x1,yval,yval,z0,z1,  _grid,_memtype,_imageid); }
  ConstRegion3DY operator()(int yval0, int yval1) const { return ConstRegion3DY(x0,x1,yval0,yval1,z0,z1,_grid,_memtype,_imageid); }
};



class Region3D : public ConstRegion3D {
public:

  Region3D(int x0_val, int x1_val, int y0_val, int y1_val, int z0_val, int z1_val, const Grid3DBase *grid_val, MemoryType memtype_val, int imageid_val) :
    ConstRegion3D(x0_val, x1_val, y0_val, y1_val, z0_val, z1_val, grid_val, memtype_val, imageid_val) { }  

  Region3D() : ConstRegion3D() {}

  Grid3DBase *grid() const { return _grid; }
};

class Region3DY : public Region3D {

public:
  Region3DY(int x0_val, int x1_val, int y0_val, int y1_val, int z0_val, int z1_val, const Grid3DBase *grid_val, MemoryType memtype_val, int imageid_val) :
    Region3D(x0_val, x1_val, y0_val, y1_val, z0_val, z1_val, grid_val, memtype_val, imageid_val) { }  

  Region3D operator()(void)                 const { return Region3D(x0,x1,y0,y1,z0,z1,      _grid,_memtype,_imageid); }
  Region3D operator()(int zval)             const { return Region3D(x0,x1,y0,y1,zval,zval,  _grid,_memtype,_imageid); }
  Region3D operator()(int zval0, int zval1) const { return Region3D(x0,x1,y0,y1,zval0,zval1,_grid,_memtype,_imageid); }
};


class Region3DX : public Region3D {

public:
  Region3DX(int x0_val, int x1_val, int y0_val, int y1_val, int z0_val, int z1_val, const Grid3DBase *grid_val, MemoryType memtype_val, int imageid_val) :
    Region3D(x0_val, x1_val, y0_val, y1_val, z0_val, z1_val, grid_val, memtype_val, imageid_val) { }  

  Region3DY operator()(void)                 const { return Region3DY(x0,x1,y0,y1,z0,z1,      _grid,_memtype,_imageid); }
  Region3DY operator()(int yval)             const { return Region3DY(x0,x1,yval,yval,z0,z1,  _grid,_memtype,_imageid); }
  Region3DY operator()(int yval0, int yval1) const { return Region3DY(x0,x1,yval0,yval1,z0,z1,_grid,_memtype,_imageid); }
};


} // end namespace

#endif

