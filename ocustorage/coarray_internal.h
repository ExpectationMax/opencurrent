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

#ifndef __OCU_STORAGE_COARRAY_INTERNAL_H__
#define __OCU_STORAGE_COARRAY_INTERNAL_H__

#include "ocustorage/coarray.h"

namespace ocu {

// A region may be host, local device, or remote device.
// At least one must be remote device
// That means there are 5 possibilities:
// 1a. host to remote device
// 2a. device to remote device
// 3a. remote device to host
// 4a. remote device to device
// 5a. remote device to remote device
// There are also local transfers which do not involve pushing requests
// onto remote q's, although they may require servicing requests.
// 1b. host to host
// 2b. host to device
// 3b. device to host
// 4b. device to device
// 5b. no-op (participate in the barrier, but do nothing)
// For now: only support 2a, 4a, (and later 5b).

enum ExchangeType {
  XT_SUPPORTED   = 0x10000000,
  XT_UNSUPPORTED = 0x00000000,

  // Exchanges involving remote devices:
  XT_HOST_TO_RDEVICE =    1 | XT_SUPPORTED,
  XT_DEVICE_TO_RDEVICE =  2 | XT_SUPPORTED,
  XT_RDEVICE_TO_HOST   =  3 | XT_SUPPORTED,
  XT_RDEVICE_TO_DEVICE =  4 | XT_SUPPORTED,
  XT_RDEVICE_TO_RDEVICE = 5,
  // Exchanges involving local-only:
  XT_HOST_TO_HOST       = 6 | XT_SUPPORTED,
  XT_HOST_TO_DEVICE     = 7 | XT_SUPPORTED,
  XT_DEVICE_TO_HOST     = 8 | XT_SUPPORTED,
  XT_DEVICE_TO_DEVICE   = 9 | XT_SUPPORTED,
  XT_NONE               = 10 | XT_SUPPORTED
};

ExchangeType determine_type(int dst_imageid, MemoryType dst_memtype, int src_imageid, MemoryType src_memtype);

// These are the commends that will be executed by a remote image.
enum TransferCommand {
  TRANSFER_DEVICE_TO_HOST,
  TRANSFER_HOST_TO_DEVICE,
  TRANSFER_DEVICE_TO_HOSTBUFFER,
  TRANSFER_HOSTBUFFER_TO_DEVICE,
  TRANSFER_ALLOCATE,
  TRANSFER_FREE,
};

enum TransferBufferMethod {
  TBM_INVALID,
  TBM_PACKED,
  TBM_CONTIGUOUS
};

TransferBufferMethod determine_method(const Region3D &dst, const ConstRegion3D &src);
size_t contiguous_distance(const ConstRegion3D &rgn);


struct TransferRequest1D {
  // Possible commands:
  // HOST_TO_DEVICE
  // DEVICE_TO_HOST
  const void *src;
  void *dst;
  size_t num_bytes;
  TransferCommand cmd;
};

struct TransferRequest3D {
  // info needed for transfer - this will have to be extended for 2d/3d xfers
  // Possible commands:
  // HOST_TO_DEVICE
  // HOSTBUFFER_TO_DEVICE
  // DEVICE_TO_HOST
  // DEVICE_TO_HOSTBUFFER
  ConstRegion3D src;
  Region3D dst;
  void *host_buffer; // either src or dst
  void *device_buffer; // for intermediate storage
  TransferCommand cmd;
  TransferBufferMethod method;
};

struct TransferRequestAlloc {
  // Possible commands:
  // ALLOCATE
  // FREE
  void *ptr;
  void **result;
  bool *valid;
  size_t num_bytes;
  TransferCommand cmd;
};

// Every exchange can have up to 1 host buffer and 1 device buffer allocated
struct ExchangeMemoryHandle
{
  bool valid;
  size_t d_local_size;
  void * d_local_bytes;
  size_t d_remote_size;
  void * d_remote_bytes;
  size_t h_size;
  void * h_bytes;
  int    remote_id;

  Region1D dst1d;
  ConstRegion1D src1d;
  Region3D dst3d;
  ConstRegion3D src3d;
  int dimension;
  ExchangeType type;
  TransferBufferMethod method;

  ExchangeMemoryHandle() : 
    valid(false), d_local_bytes(0), d_local_size(0), d_remote_bytes(0), d_remote_size(0), 
      h_bytes(0), h_size(0), remote_id(-1), dimension(0), type(XT_UNSUPPORTED), method(TBM_INVALID) { }
};


class ExchangeMemoryPool {
  std::vector<ExchangeMemoryHandle *> _handles;
public:
  ExchangeMemoryHandle * allocate_handle(size_t num_host_bytes, size_t num_local_device_bytes, size_t num_remote_device_bytes, bool host_write_combined, int remote_id, int &handle);
  void remove_handle(int handle);

  bool is_handle_valid(int h) const {
    if (h < 0 || h >= _handles.size()) return false;
    return _handles[h] && _handles[h]->valid;
  }

  const ExchangeMemoryHandle *get_handle(int h) const {
    if (!is_handle_valid(h)) return 0;
    return _handles[h];
  }

  void cleanup();
};



class TransferRequestQ {
  std::vector<TransferRequest1D> _q1d;
  std::vector<TransferRequest3D> _q3d;
  std::vector<TransferRequestAlloc> _qalloc;
  int _image_id;

#ifdef OCU_OMP
  omp_lock_t _lock;
#endif

public:
  TransferRequestQ(int image_id) :
      _image_id(image_id) {
#ifdef OCU_OMP
    omp_init_lock(&_lock);
#endif
    }

  ~TransferRequestQ() {
#ifdef OCU_OMP
    omp_destroy_lock(&_lock);
#endif
  }

  void push(const TransferRequest1D &request) {
#ifdef OCU_OMP
    omp_set_lock(&_lock);
#endif
    _q1d.push_back(request);
#ifdef OCU_OMP
    omp_unset_lock(&_lock);
#endif
  }

  void push(const TransferRequest3D &request) {
#ifdef OCU_OMP
    omp_set_lock(&_lock);
#endif
    _q3d.push_back(request);
#ifdef OCU_OMP
    omp_unset_lock(&_lock);
#endif
  }

  void push(const TransferRequestAlloc &request) {
#ifdef OCU_OMP
    omp_set_lock(&_lock);
#endif
    _qalloc.push_back(request);
#ifdef OCU_OMP
    omp_unset_lock(&_lock);
#endif
  }

  void process1d();
  void process3d();
  void processalloc();

  void process() {
    processalloc();
    process1d();
    process3d();
  }
};





// 1d methods are all contiguous
void remote_xfer_hregion1d_to_dregion1d(int image_id, void *d_dst, const void *h_src, size_t num_bytes);
void remote_xfer_dregion1d_to_hregion1d(int image_id, void *h_dst, const void *d_src, size_t num_bytes);

void remote_xfer_hbuffer_to_dregion3d(int image_id, const Region3D &dst, void *host_buffer, void *device_buffer, TransferBufferMethod tbm);
void remote_xfer_dregion3d_to_hbuffer(int image_id, void *host_buffer, void *device_buffer, const ConstRegion3D &src, TransferBufferMethod tbm);

// 1d methods are all contiguous
bool xfer_hregion1d_to_dregion1d(void *d_dst, const void *h_src, size_t num_bytes);
bool xfer_dregion1d_to_hregion1d(void *h_dst, const void *d_src, size_t num_bytes);
bool xfer_hregion1d_to_hregion1d(void *h_dst, const void *h_src, size_t num_bytes);
bool xfer_dregion1d_to_dregion1d(void *d_dst, const void *d_src, size_t num_bytes);

bool xfer_hbuffer_to_dregion3d(const Region3D &dst, void *host_buffer, void *device_buffer, TransferBufferMethod tbm);
bool xfer_dregion3d_to_hbuffer(void *host_buffer, void *device_buffer, const ConstRegion3D &src, TransferBufferMethod tbm);
bool xfer_dregion3d_to_dregion3d(const Region3D &d_dst, const ConstRegion3D &d_src, TransferBufferMethod tbm);
bool xfer_hregion3d_to_hregion3d(const Region3D &h_dst, const ConstRegion3D &h_src, TransferBufferMethod tbm);
bool xfer_hbuffer_to_hregion3d(const Region3D &h_dst, const void *h_src, TransferBufferMethod tbm);
bool xfer_hregion3d_to_hbuffer(void *h_dst, const ConstRegion3D &h_src, TransferBufferMethod tbm);

// TODO: enable fast paths:
//bool xfer_dregion3d_to_hregion3d(const Region3D &h_dst, const ConstRegion3D &d_src, TransferBufferMethod tbm);
//bool xfer_hregion3d_to_dregion3d(const Region3D &d_dst, const ConstRegion3D &h_src, TransferBufferMethod tbm);


} // end namespace


#endif

