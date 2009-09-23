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

#ifndef __OCU_STORAGE_GRID1D_H__
#define __OCU_STORAGE_GRID1D_H__

#include "ocuutil/defines.h"

namespace ocu {

class Grid1DUntyped {

protected:
  //**** MEMBER VARIABLES ****
  int _nx; // computational cells in x
  int _gx; // ghost cells in x
  int _pnx; // padded & aligned sizes

  //**** MANAGERS ****
  ~Grid1DUntyped() { }
  Grid1DUntyped() { _nx = 0; _gx = 0; _pnx = 0; }

public:
  //**** PUBLIC INTERFACE ****
  OCU_HOSTDEVICE int    nx() const { return _nx; }
  OCU_HOSTDEVICE int   pnx() const { return _pnx; }
  OCU_HOSTDEVICE int    gx() const { return _gx; }

};

template<typename T>
class Grid1DBase : public Grid1DUntyped
{
  // disallow
  Grid1DBase &operator=(const Grid1DBase &) { return *this; }
  Grid1DBase(const Grid1DBase &) { }

protected:

  //**** MEMBER VARIABLES ****
  T *_buffer;
  T *_shifted_buffer;

  //**** MANAGERS ****
  ~Grid1DBase() { }
  Grid1DBase() { _buffer = 0; _shifted_buffer = 0; }

public:

  //**** PUBLIC INTERFACE ****
  OCU_HOSTDEVICE T       &at       (int i)         { return *(_shifted_buffer + i); }
  OCU_HOSTDEVICE const T &at       (int i)  const  { return *(_shifted_buffer + i); }
  
  OCU_HOSTDEVICE const T *buffer() const { return _buffer; }
  OCU_HOSTDEVICE       T *buffer()       { return _buffer; }
};


template <typename T>
class Grid1DHost;

template <typename T>
class Grid1DDevice;


template <typename T>
class Grid1DHost : public Grid1DBase<T>
{
  //! Whether allocated host memory is pinned or not.  Pinned memory can be transfered to the device faster, but it cannot be swapped by the OS
  //! and hence reduces the amount of usable virtual memory in the system.
  bool _pinned;
public:
  
  ~Grid1DHost();

  //*** PUBLIC INTERFACE ****

  //! Allocate memory for this grid with nx cells and gx "ghost" padded cells in either end.
  bool init(int nx_val, int gx_val, bool pinned = true);

  //! Clear the entire array to zero.
  void clear_zero();

  //! Copy interior points only.  Fails if nx dimensions mismatch.
  bool copy_interior_data(const Grid1DHost<T> &from);

  //! Copy interior and ghost points.  Fails if px dimensions mismatch.
  bool copy_all_data(const Grid1DHost<T> &from);

  //! Copy interior points only from device across to host.  Fails if nx dimensions mismatch.
  bool copy_interior_data(const Grid1DDevice<T> &from);

  //! Copy interior and ghost points from device across to host.  Fails if px dimensions mismatch.
  bool copy_all_data(const Grid1DDevice<T> &from);

  //! set this = alpha * g1, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  bool linear_combination(T alpha1, const Grid1DHost<T> &g1);

  //! set this = alpha1 * g1 + alpha2 * g2, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  bool linear_combination(T alpha1, const Grid1DHost<T> &g1, T alpha2, const Grid1DHost<T> &g2);

  bool reduce_maxabs(T &result) const;
  bool reduce_max(T &result) const;
  bool reduce_min(T &result) const;
  bool reduce_sum(T &result) const;
  bool reduce_sqrsum(T &result) const;
  bool reduce_checknan(T &result) const;
};



template <typename T>
class Grid1DDevice : public Grid1DBase<T>
{
public:

  OCU_HOST ~Grid1DDevice();
  
  OCU_HOST bool init(int nx_val, int gx_val);
  OCU_HOST void clear_zero();

  OCU_HOST bool copy_interior_data(const Grid1DDevice<T> &from);
  OCU_HOST bool copy_all_data(const Grid1DDevice<T> &from);

  OCU_HOST bool copy_interior_data(const Grid1DHost<T> &from);
  OCU_HOST bool copy_all_data(const Grid1DHost<T> &from);

  //! set this = alpha * g1, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  OCU_HOST bool linear_combination(T alpha1, const Grid1DDevice<T> &g1);

  //! set this = alpha1 * g1 + alpha2 * g2, term-by-term.  Ignores ghost cells, but fails if interior dimensions don't match.
  OCU_HOST bool linear_combination(T alpha1, const Grid1DDevice<T> &g1, T alpha2, const Grid1DDevice<T> &g2);

  OCU_HOST bool reduce_maxabs(T &result) const;
  OCU_HOST bool reduce_max(T &result) const;
  OCU_HOST bool reduce_min(T &result) const;
  OCU_HOST bool reduce_sum(T &result) const;
  OCU_HOST bool reduce_sqrsum(T &result) const;
  OCU_HOST bool reduce_checknan(T &result) const;
};

typedef Grid1DHost<float> Grid1DHostF;
typedef Grid1DHost<double> Grid1DHostD;
typedef Grid1DHost<int> Grid1DHostI;

typedef Grid1DDevice<float> Grid1DDeviceF;
typedef Grid1DDevice<double> Grid1DDeviceD;
typedef Grid1DDevice<int> Grid1DDeviceI;



} // end namespace

#endif

