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
#include "ocuutil/thread.h"
#include "ocustorage/region1d.h"

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


class Grid1DBase : public Grid1DUntyped 
{
  // disallow
  Grid1DBase &operator=(const Grid1DBase &) { return *this; }
  Grid1DBase(const Grid1DBase &) { }

protected:

  //**** MEMBER VARIABLES ****
  void *_buffer;
  void *_shifted_buffer;
  size_t _atom_size;
  int _image_id; // which image created and can access this grid

  //**** MANAGERS ****
  ~Grid1DBase() { }
  Grid1DBase(size_t atom_size_val) : _atom_size(atom_size_val) { _buffer = 0; _shifted_buffer = 0; _image_id = ThreadManager::this_image(); }

public:

  //**** PUBLIC INTERFACE ****
  OCU_HOSTDEVICE size_t       atom_size() const { return _atom_size; }
  OCU_HOSTDEVICE void *       ptr_untyped(int i)         { return ((char *)_shifted_buffer) + _atom_size * i; }
  OCU_HOSTDEVICE const void * ptr_untyped(int i)  const  { return ((const char *)_shifted_buffer) + _atom_size * i; }  
};

template<typename T>
class Grid1DTypedBase : public Grid1DBase
{

protected:

  //**** MANAGERS ****
  ~Grid1DTypedBase() { }
  Grid1DTypedBase() : Grid1DBase(sizeof(T)) { }

public:

  //**** PUBLIC INTERFACE ****
  OCU_HOSTDEVICE T       &at       (int i)         { return *(((T*)_shifted_buffer) + i); }
  OCU_HOSTDEVICE const T &at       (int i)  const  { return *(((const T*)_shifted_buffer) + i); }
  
  OCU_HOSTDEVICE const T *buffer() const { return (const T*)_buffer; }
  OCU_HOSTDEVICE       T *buffer()       { return (T *)_buffer; }
};


template <typename T>
class Grid1DHost;

template <typename T>
class Grid1DDevice;


template <typename T>
class Grid1DHost : public Grid1DTypedBase<T>
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

  //! Clear the entire array to t.
  void clear(T t);

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

  ConstRegion1D region()                     const { return ConstRegion1D(-this->gx(),this->nx()-1+this->gx(), this, MEM_HOST, this->_image_id); }
  ConstRegion1D region(int xval)             const { return ConstRegion1D(xval,xval, this, MEM_HOST, this->_image_id); }
  ConstRegion1D region(int xval0, int xval1) const { return ConstRegion1D(xval0,xval1, this, MEM_HOST, this->_image_id); }
  Region1D      region()                           { return Region1D(-this->gx(),this->nx()-1+this->gx(), this, MEM_HOST, this->_image_id); }
  Region1D      region(int xval)                   { return Region1D(xval,xval, this, MEM_HOST, this->_image_id); }
  Region1D      region(int xval0, int xval1)       { return Region1D(xval0,xval1, this, MEM_HOST, this->_image_id); }
};



template <typename T>
class Grid1DDevice : public Grid1DTypedBase<T>
{
public:

  OCU_HOST Grid1DDevice() {}
  OCU_HOST ~Grid1DDevice();
  
  OCU_HOST bool init(int nx_val, int gx_val);
  OCU_HOST bool clear_zero();
  
  OCU_HOST bool clear(T t);

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

  ConstRegion1D region()                     const { return ConstRegion1D(-this->gx(),this->nx()-1+this->gx(), this, MEM_DEVICE, this->_image_id); }
  ConstRegion1D region(int xval)             const { return ConstRegion1D(xval,xval, this, MEM_DEVICE, this->_image_id); }
  ConstRegion1D region(int xval0, int xval1) const { return ConstRegion1D(xval0,xval1, this, MEM_DEVICE, this->_image_id); }
  Region1D      region()                           { return Region1D(-this->gx(),this->nx()-1+this->gx(), this, MEM_DEVICE, this->_image_id); }
  Region1D      region(int xval)                   { return Region1D(xval,xval, this, MEM_DEVICE, this->_image_id); }
  Region1D      region(int xval0, int xval1)       { return Region1D(xval0,xval1, this, MEM_DEVICE, this->_image_id); }
};

class CoArrayTable;
template<typename T>
class Grid1DDeviceCo : public Grid1DDevice<T>
{
  CoArrayTable *_table; // directory to siblings

public:
  Grid1DDeviceCo(const char *id);
  ~Grid1DDeviceCo();

  bool init(int nx_val, int gx_val) { 
    bool ok = Grid1DDevice<T>::init(nx_val, gx_val);
    ThreadManager::barrier();
    return ok;
  }

        Grid1DDeviceCo<T> *co(int image);
  const Grid1DDeviceCo<T> *co(int image) const;

  // reduce over all images
  bool co_reduce_maxabs(T &result) const;
  bool co_reduce_sum(T &result) const;
  bool co_reduce_sqrsum(T &result) const;
  bool co_reduce_max(T &result) const;
  bool co_reduce_min(T &result) const;
  bool co_reduce_checknan(T &result) const;
};

template<typename T>
class Grid1DHostCo : public Grid1DHost<T>
{
  CoArrayTable *_table; // directory to siblings

public:
  Grid1DHostCo(const char *id);
  ~Grid1DHostCo();

  bool init(int nx_val, int gx_val, bool pinned=true) { 
    bool ok = Grid1DHost<T>::init(nx_val, gx_val, pinned);
    ThreadManager::barrier();
    return ok;
  }

        Grid1DHostCo<T> *co(int image);
  const Grid1DHostCo<T> *co(int image) const;

  // reduce over all images
  bool co_reduce_maxabs(T &result) const;
  bool co_reduce_sum(T &result) const;
  bool co_reduce_sqrsum(T &result) const;
  bool co_reduce_max(T &result) const;
  bool co_reduce_min(T &result) const;
  bool co_reduce_checknan(T &result) const;
};




typedef Grid1DHost<float> Grid1DHostF;
typedef Grid1DHost<double> Grid1DHostD;
typedef Grid1DHost<int> Grid1DHostI;

typedef Grid1DDevice<float> Grid1DDeviceF;
typedef Grid1DDevice<double> Grid1DDeviceD;
typedef Grid1DDevice<int> Grid1DDeviceI;

typedef Grid1DDeviceCo<float> Grid1DDeviceCoF;
typedef Grid1DDeviceCo<double> Grid1DDeviceCoD;
typedef Grid1DDeviceCo<int> Grid1DDeviceCoI;

typedef Grid1DHostCo<float> Grid1DHostCoF;
typedef Grid1DHostCo<double> Grid1DHostCoD;
typedef Grid1DHostCo<int> Grid1DHostCoI;

} // end namespace

#endif

