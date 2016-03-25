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

#ifndef __OCU_STORAGE_GRID2D_H__
#define __OCU_STORAGE_GRID2D_H__

#include "ocuutil/defines.h"

namespace ocu {

class Grid2DUntyped
{
protected:

  //**** MEMBER VARIABLES ****
  int _nx, _ny; // computational cells in x
  int _gx, _gy; // ghost cells in x
  int _pnx, _pny; // padded & aligned sizes

  //**** MANAGERS ****
  ~Grid2DUntyped() { }
  Grid2DUntyped() { 
    _nx = _ny = 0; 
    _gx = _gy = 0; 
    _pnx = _pny = 0; 
  }

public:

  //**** PUBLIC INTERFACE ****
  OCU_HOSTDEVICE int    nx() const { return _nx; }
  OCU_HOSTDEVICE int    ny() const { return _ny; }

  OCU_HOSTDEVICE int   pnx() const { return _pnx; }
  OCU_HOSTDEVICE int   pny() const { return _pny; }

  OCU_HOSTDEVICE int    gx() const { return _gx; }
  OCU_HOSTDEVICE int    gy() const { return _gy; }

  OCU_HOSTDEVICE int    paddingx() const { return _pnx - (_nx + 2 * _gx); }
  OCU_HOSTDEVICE int    paddingy() const { return _pny - (_ny + 2 * _gy); }

  OCU_HOSTDEVICE int    num_allocated_elements() const { return _pny * _pnx; }

  OCU_HOSTDEVICE int    xstride() const { return _pny; }
  OCU_HOSTDEVICE int    ystride() const { return 1; }

  bool check_interior_dimension_match(const Grid2DUntyped &other) const { return nx() == other.nx() && ny() == other.ny(); }
  bool check_layout_match            (const Grid2DUntyped &other) const { return pnx() == other.pnx() && pny() == other.pny(); }
};

template<typename T>
class Grid2DBase : public Grid2DUntyped
{
  // disallow
  Grid2DBase &operator=(const Grid2DBase &) { return *this; }
  Grid2DBase(const Grid2DBase &) { }

protected:

  T *_buffer;
  T *_shifted_buffer;

  //**** MANAGERS ****
  ~Grid2DBase() { }
  Grid2DBase() { 
    _buffer = 0; 
    _shifted_buffer = 0; 
  }

public:

  //**** PUBLIC INTERFACE ****
  OCU_HOSTDEVICE T       &at       (int i, int j)         { return *(_shifted_buffer + i * _pny + j); }
  OCU_HOSTDEVICE const T &at       (int i, int j)  const  { return *(_shifted_buffer + i * _pny + j); }

  OCU_HOSTDEVICE const T *buffer() const { return _buffer; }
  OCU_HOSTDEVICE       T *buffer()       { return _buffer; }
};


template <typename T>
class Grid2DHost;

template <typename T>
class Grid2DDevice;


template <typename T>
class Grid2DHost : public Grid2DBase<T>
{
  //! Whether allocated host memory is pinned or not.  Pinned memory can be transfered to the device faster, but it cannot be swapped by the OS
  //! and hence reduces the amount of usable virtual memory in the system.
  bool _pinned;
public:
  
  ~Grid2DHost();

  //*** PUBLIC INTERFACE ****

  //! Allocate memory for this grid with nx cells and gx "ghost" padded cells in either end.
  //! Optionally specify whether this should be pinned memory, and extra elements per row with which
  //! to pad the arrays to make the memory layout line up in some particular way.
  bool init(int nx_val, int ny_val, int gx_val, int gy_val, bool pinned = true, int padx=0, int pady=0);

  //! Clear the entire array to zero.
  void clear_zero();

  //! Set value over entire grid.
  void clear(T t);

  //! Copy interior points only.  Fails if nx dimensions mismatch.
  bool copy_interior_data(const Grid2DHost<T> &from);

  //! Copy interior and ghost points.  Fails if px dimensions mismatch.
  bool copy_all_data(const Grid2DHost<T> &from);

  //! Copy interior and ghost points from device across to host.  Fails if px dimensions mismatch.
  bool copy_all_data(const Grid2DDevice<T> &from);

  //! set this = alpha * g1, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  bool linear_combination(T alpha1, const Grid2DHost<T> &g1);

  //! set this = alpha1 * g1 + alpha2 * g2, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  bool linear_combination(T alpha1, const Grid2DHost<T> &g1, T alpha2, const Grid2DHost<T> &g2);

};



template <typename T>
class Grid2DDevice : public Grid2DBase<T>
{
public:

  OCU_HOST ~Grid2DDevice();
  
  OCU_HOST bool init(int nx_val, int ny_val, int gx_val, int gy_val, int padx=0, int pady=0);
  OCU_HOST bool clear_zero();
  OCU_HOST bool clear(T t);

  OCU_HOST bool copy_all_data(const Grid2DDevice<T> &from);
  OCU_HOST bool copy_all_data(const Grid2DHost<T> &from);

  //! set this = alpha * g1, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  OCU_HOST bool linear_combination(T alpha1, const Grid2DDevice<T> &g1);

  //! set this = alpha1 * g1 + alpha2 * g2, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  OCU_HOST bool linear_combination(T alpha1, const Grid2DDevice<T> &g1, T alpha2, const Grid2DDevice<T> &g2);

};

typedef Grid2DHost<float> Grid2DHostF;
typedef Grid2DHost<double> Grid2DHostD;
typedef Grid2DHost<int> Grid2DHostI;

typedef Grid2DDevice<float> Grid2DDeviceF;
typedef Grid2DDevice<double> Grid2DDeviceD;
typedef Grid2DDevice<int> Grid2DDeviceI;



} // end namespace

#endif

