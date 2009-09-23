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

#ifndef __OCU_STORAGE_GRID3D_H__
#define __OCU_STORAGE_GRID3D_H__

#include "ocuutil/defines.h"
#include "ocuutil/direction.h"
#include <vector>

namespace ocu {

class Grid3DUntyped
{
protected:

  //**** MEMBER VARIABLES ****
  int _nx, _ny, _nz; // computational cells in x
  int _gx, _gy, _gz; // ghost cells in x
  int _pnx, _pny, _pnz; // padded & aligned sizes
  int _pnzpny;
  int _shift_amount; // offset from start of buffer to (0,0,0)
  int _allocated_elements; // total # element allocated

  //**** MANAGERS ****
  ~Grid3DUntyped() { }
  Grid3DUntyped() { 
    _nx = _ny = _nz = 0; 
    _gx = _gy = _gz = 0; 
    _pnx = _pny = _pnz = 0; 
    _pnzpny = 0; 
    _shift_amount = 0;
    _allocated_elements = 0;
  }

public:

  //**** PUBLIC INTERFACE ****
  OCU_HOSTDEVICE int    nx() const { return _nx; }
  OCU_HOSTDEVICE int    ny() const { return _ny; }
  OCU_HOSTDEVICE int    nz() const { return _nz; }

  OCU_HOSTDEVICE int   pnx() const { return _pnx; }
  OCU_HOSTDEVICE int   pny() const { return _pny; }
  OCU_HOSTDEVICE int   pnz() const { return _pnz; }

  OCU_HOSTDEVICE int    gx() const { return _gx; }
  OCU_HOSTDEVICE int    gy() const { return _gy; }
  OCU_HOSTDEVICE int    gz() const { return _gz; }

  OCU_HOSTDEVICE int    paddingx() const { return _pnx - (_nx + 2 * _gx); }
  OCU_HOSTDEVICE int    paddingy() const { return _pny - (_ny + 2 * _gy); }
  OCU_HOSTDEVICE int    paddingz() const { return _pnz - (_nz + 2 * _gz); }

  OCU_HOSTDEVICE int    num_allocated_elements() const { return _allocated_elements; }

  OCU_HOSTDEVICE int    xstride() const { return _pnzpny; }
  OCU_HOSTDEVICE int    ystride() const { return _pnz; }
  OCU_HOSTDEVICE int    zstride() const { return 1; }
  OCU_HOSTDEVICE int    stride(DirectionType dir) const {
    return (dir & DIR_XAXIS_FLAG) ? xstride() :
           (dir & DIR_YAXIS_FLAG) ? ystride() : zstride();
  }

  OCU_HOSTDEVICE int    shift_amount() const { return _shift_amount; } 

  bool check_interior_dimension_match(const Grid3DUntyped &other) const { return nx() == other.nx() && ny() == other.ny() && nz() == other.nz(); }
  bool check_layout_match            (const Grid3DUntyped &other) const { return pnx() == other.pnx() && pny() == other.pny() && pnz() == other.pnz(); }


};


class Grid3DDimension : public Grid3DUntyped {
public:
  // Does not allocate any memory, just sets the internal sizes
  void init(int nx_val, int ny_val, int nz_val, int gx_val, int gy_val, int gz_val);

  // This will adjust all passed in grid px,py,pz values so that congruence is achieved.
  static void pad_for_congruence(std::vector<Grid3DDimension> &grids);
};

template<typename T>
class Grid3DBase : public Grid3DUntyped
{
  // disallow
  Grid3DBase &operator=(const Grid3DBase &) { return *this; }
  Grid3DBase(const Grid3DBase &) { }

protected:

  //**** MEMBER VARIABLES ****
  T *_buffer;
  T *_shifted_buffer;

  //**** MANAGERS ****
  ~Grid3DBase() { }
  Grid3DBase() { 
    _buffer = 0; 
    _shifted_buffer = 0; 
  }

public:

  //**** PUBLIC INTERFACE ****
  OCU_HOSTDEVICE T       &at       (int i, int j, int k)         { return *(_shifted_buffer + i * _pnzpny + j * _pnz + k); }
  OCU_HOSTDEVICE const T &at       (int i, int j, int k)  const  { return *(_shifted_buffer + i * _pnzpny + j * _pnz + k); }
  
  OCU_HOSTDEVICE const T *buffer() const { return _buffer; }
  OCU_HOSTDEVICE       T *buffer()       { return _buffer; }
};


template <typename T>
class Grid3DHost;

template <typename T>
class Grid3DDevice;


template <typename T>
class Grid3DHost : public Grid3DBase<T>
{
  //! Whether allocated host memory is pinned or not.  Pinned memory can be transfered to the device faster, but it cannot be swapped by the OS
  //! and hence reduces the amount of usable virtual memory in the system.
  bool _pinned;
public:
  
  ~Grid3DHost();

  //*** PUBLIC INTERFACE ****

  //! Allocate memory for this grid with nx cells and gx "ghost" padded cells in either end.
  //! Optionally specify whether this should be pinned memory, and extra elements per row with which
  //! to pad the arrays to make the memory layout line up in some particular way.
  //! The actual padding may be larger as determined by hardware constraints.
  bool init(int nx_val, int ny_val, int nz_val, int gx_val, int gy_val, int gz_val, bool pinned = true, int padx=0, int pady=0, int padz=0);

  //! Allocate memory so that the memory layout of this will be congruent with the memory layout of "other"
  bool init_congruent(const Grid3DUntyped &other, bool pinned=true) {
    return init(other.nx(), other.ny(), other.nz(), other.gx(), other.gy(), other.gz(), pinned, other.paddingx(), other.paddingy(), other.paddingz());
  }

  //! Clear the entire array to zero.
  void clear_zero();

  //! Set value over entire grid.
  void clear(T t);

  //! Copy interior and ghost points.  Fails if px dimensions mismatch.
  template<typename S>
  bool copy_all_data(const Grid3DHost<S> &from);

  //! Copy interior and ghost points from device across to host.  Fails if px dimensions mismatch.
  bool copy_all_data(const Grid3DDevice<T> &from);

  //! set this = alpha * g1, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  bool linear_combination(T alpha1, const Grid3DHost<T> &g1);

  //! set this = alpha1 * g1 + alpha2 * g2, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  bool linear_combination(T alpha1, const Grid3DHost<T> &g1, T alpha2, const Grid3DHost<T> &g2);

  bool reduce_maxabs(T &result) const;
  bool reduce_max(T &result) const;
  bool reduce_min(T &result) const;
  bool reduce_sum(T &result) const;
  bool reduce_sqrsum(T &result) const;
  bool reduce_checknan(T &result) const; // if any nans found, returns nan, otherwise returns non-nan
};



template <typename T>
class Grid3DDevice : public Grid3DBase<T>
{
public:

  OCU_HOST ~Grid3DDevice();
  
  OCU_HOST bool init(int nx_val, int ny_val, int nz_val, int gx_val, int gy_val, int gz_val, int padx=0, int pady=0, int padz=0);
  
  //! Allocate memory for this grid with nx cells and gx "ghost" padded cells in either end.
  //! Optionally specify whether this should be pinned memory, and extra elements per row with which
  //! to pad the arrays to make the memory layout line up in some particular way.
  //! The actual padding may be larger as determined by hardware constraints.
  OCU_HOST bool init_congruent(const Grid3DUntyped &other) {
    return init(other.nx(), other.ny(), other.nz(), other.gx(), other.gy(), other.gz(), other.paddingx(), other.paddingy(), other.paddingz());
  }

  OCU_HOST bool clear_zero();
  OCU_HOST bool clear(T t);

  template<typename S>
  OCU_HOST bool copy_all_data(const Grid3DDevice<S> &from);

  OCU_HOST bool copy_all_data(const Grid3DHost<T> &from);

  //! set this = alpha * g1, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  OCU_HOST bool linear_combination(T alpha1, const Grid3DDevice<T> &g1);

  //! set this = alpha1 * g1 + alpha2 * g2, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  OCU_HOST bool linear_combination(T alpha1, const Grid3DDevice<T> &g1, T alpha2, const Grid3DDevice<T> &g2);

  OCU_HOST bool reduce_maxabs(T &result) const;
  OCU_HOST bool reduce_sum(T &result) const;
  OCU_HOST bool reduce_sqrsum(T &result) const;
  OCU_HOST bool reduce_max(T &result) const;
  OCU_HOST bool reduce_min(T &result) const;
  OCU_HOST bool reduce_checknan(T &result) const; // if any nans found, returns nan, otherwise returns non-nan
};

typedef Grid3DHost<float> Grid3DHostF;
typedef Grid3DHost<double> Grid3DHostD;
typedef Grid3DHost<int> Grid3DHostI;

typedef Grid3DDevice<float> Grid3DDeviceF;
typedef Grid3DDevice<double> Grid3DDeviceD;
typedef Grid3DDevice<int> Grid3DDeviceI;




} // end namespace

#endif
