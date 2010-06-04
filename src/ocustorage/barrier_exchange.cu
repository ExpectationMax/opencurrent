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

#include "ocustorage/coarray_internal.h"

namespace ocu {


ExchangeType determine_type(int dst_imageid, MemoryType dst_memtype, int src_imageid, MemoryType src_memtype)
{
  int imageid = ThreadManager::this_image();

  if (dst_imageid == imageid && dst_memtype == MEM_DEVICE && src_imageid != imageid && src_memtype == MEM_DEVICE)
    return XT_RDEVICE_TO_DEVICE;
  if (dst_imageid != imageid && dst_memtype == MEM_DEVICE && src_imageid == imageid && src_memtype == MEM_DEVICE)
    return XT_DEVICE_TO_RDEVICE;
  if (dst_imageid != imageid && dst_memtype == MEM_DEVICE && src_memtype == MEM_HOST)
    return XT_HOST_TO_RDEVICE;
  if (dst_memtype == MEM_HOST && src_imageid != imageid && src_memtype == MEM_DEVICE)
    return XT_RDEVICE_TO_HOST;
  if (dst_imageid != imageid && dst_memtype == MEM_DEVICE && src_imageid != imageid && src_memtype == MEM_DEVICE)
    return XT_RDEVICE_TO_RDEVICE;

  if (dst_memtype == MEM_HOST && src_memtype == MEM_HOST)
    return XT_HOST_TO_HOST;
  if (dst_imageid == imageid && dst_memtype == MEM_DEVICE && src_memtype == MEM_HOST)
    return XT_HOST_TO_DEVICE;
  if (dst_memtype == MEM_HOST && src_imageid == imageid && src_memtype == MEM_DEVICE)
    return XT_DEVICE_TO_HOST;
  if (dst_imageid == imageid && dst_memtype == MEM_DEVICE && src_imageid == imageid && src_memtype == MEM_DEVICE)
    return XT_DEVICE_TO_DEVICE;

  return XT_UNSUPPORTED;
}


TransferBufferMethod determine_method(const Region3D &dst, const ConstRegion3D &src)
{
  if (src.z0 == -src.grid()->gz() && src.z1 == (src.grid()->gz() + src.grid()->nz()-1) && 
      dst.z0 == -dst.grid()->gz() && dst.z1 == (dst.grid()->gz() + dst.grid()->nz()-1) && 
      src.y0 == -src.grid()->gy() && src.y1 == (src.grid()->gy() + src.grid()->ny()-1) &&
      dst.y0 == -dst.grid()->gy() && dst.y1 == (dst.grid()->gy() + dst.grid()->ny()-1) &&
      src.grid()->pnz() == dst.grid()->pnz() && 
      src.grid()->pny() == dst.grid()->pny() &&
      src.grid()->pnx() == dst.grid()->pnx()) {
    return TBM_CONTIGUOUS;
  }
  // TODO: there can also be a path for contiguous, but not congruent layouts

  return TBM_PACKED;

  // TODO: this is also the case that x0==x1, and the case x0==x1&&y0==y1
}

size_t contiguous_distance(const ConstRegion3D &rgn)
{
  const char *end   = (const char *)rgn.grid()->ptr_untyped(rgn.x1, rgn.y1, rgn.z1+1);
  const char *start = (const char *)rgn.grid()->ptr_untyped(rgn.x0, rgn.y0, rgn.z0);

  return end - start;
  //return static_cast<size_t>(end) - static_cast<size_t>(start);
}

void CoArrayManager::barrier_deallocate()
{
  ThreadManager::barrier();
  process_instance_requests();
}

void CoArrayManager::barrier_deallocate(int handle)
{
  if (handle != -1)
    mempool()->remove_handle(handle);
  ThreadManager::barrier();
  process_instance_requests();
}


int CoArrayManager::barrier_allocate()
{
  ThreadManager::barrier();
  process_instance_requests();
  ThreadManager::barrier();

  return -1;
}


int CoArrayManager::barrier_allocate(const Region1D &dst, const ConstRegion1D &src)
{
  if (!dst.is_valid() || !src.is_valid()) {
    printf("[ERROR] CoArrayManager::barrier_allocate - invalid regions\n");
    return -1;
  }

  if (src.x1 - src.x0 != dst.x1 - dst.x0) {
    printf("[ERROR] CoArrayManager::barrier_allocate - region dimensions mismatch\n");
    return -1;
  }

  if (src.grid()->atom_size() != dst.grid()->atom_size()) {
    printf("[ERROR] CoArrayManager::barrier_allocate - region type size mismatch\n");
    return -1;
  }

  // TODO: should add type checking as well

  ExchangeType xt = determine_type(dst.imageid(), dst.memtype(), src.imageid(), src.memtype());

  if (! (xt & XT_SUPPORTED)) {
    printf("[ERROR] CoArrayManager::barrier_allocate - unsupported operation\n");
    return -1;
  }

  int my_id = ThreadManager::this_image();
  int remote_id = (dst.imageid() != my_id) ? dst.imageid() : src.imageid();
  bool host_write_combined = false;


  size_t num_bytes = 0;
  switch(xt) {
    case XT_DEVICE_TO_RDEVICE:
    case XT_RDEVICE_TO_DEVICE:
      num_bytes = src.grid()->atom_size() * (src.x1 - src.x0 + 1);
      host_write_combined = true;
      break;
    case XT_HOST_TO_RDEVICE:
    case XT_RDEVICE_TO_HOST:
    case XT_HOST_TO_HOST:
    case XT_DEVICE_TO_DEVICE:
    case XT_HOST_TO_DEVICE:
    case XT_DEVICE_TO_HOST:
      num_bytes = 0;
      break;
  }

  // create handle
  int handle;
  ExchangeMemoryHandle *hdl = mempool()->allocate_handle(num_bytes, 0, 0, host_write_combined, remote_id, handle);
  hdl->dst1d = dst;
  hdl->src1d = src;
  hdl->dimension = 1;
  hdl->type = xt;

  ThreadManager::barrier();
  process_instance_requests();
  ThreadManager::barrier();

  if (!mempool()->is_handle_valid(handle)) {
    printf("[ERROR] CoArrayManager::barrier_allocate - allocation failed\n");
    return -1;
  }
  else
    return handle;
}

int CoArrayManager::barrier_allocate(const Region3D &dst, const ConstRegion3D &src)
{
  if (!dst.is_valid() || !src.is_valid()) {
    printf("[ERROR] CoArrayManager::barrier_allocate - invalid regions\n");
    return -1;
  }

  if (src.x1 - src.x0 != dst.x1 - dst.x0 ||
      src.y1 - src.y0 != dst.y1 - dst.y0 ||
      src.z1 - src.z0 != dst.z1 - dst.z0) {
    printf("[ERROR] CoArrayManager::barrier_allocate - region dimensions mismatch\n");
    return -1;
  }

  if (src.grid()->atom_size() != dst.grid()->atom_size()) {
    printf("[ERROR] CoArrayManager::barrier_allocate - region type size mismatch\n");
    return -1;
  }

  // TODO: should add type checking as well

  ExchangeType xt = determine_type(dst.imageid(), dst.memtype(), src.imageid(), src.memtype());

  if (! (xt & XT_SUPPORTED)) {
    printf("[ERROR] CoArrayManager::barrier_allocate - unsupported operation\n");
    return false;
  }

  // TODO: figure out how big an intermediate storage buffer is needed on host & device.
  // for now, just go with volume * atom_size.  Later implement more sophisticated options.
  int my_id = ThreadManager::this_image();
  int remote_id = (dst.imageid() != my_id) ? dst.imageid() : src.imageid();

  TransferBufferMethod tbm = determine_method(dst, src);

  size_t num_h_bytes = 0, num_d_bytes = 0, num_rd_bytes = 0;
  bool host_write_combined = false;
  switch(xt) {
    case XT_DEVICE_TO_RDEVICE:
    case XT_RDEVICE_TO_DEVICE:
      if (tbm == TBM_CONTIGUOUS) {
        num_d_bytes = num_rd_bytes = 0;
        num_h_bytes = contiguous_distance(src);
        host_write_combined = true;
      }
      else {
        num_h_bytes = num_d_bytes = num_rd_bytes = dst.volume() * dst.grid()->atom_size();
      }
      break;
    case XT_HOST_TO_RDEVICE:
    case XT_RDEVICE_TO_HOST:
      if (tbm == TBM_CONTIGUOUS) {
        num_h_bytes = contiguous_distance(src);
        num_d_bytes = num_rd_bytes = 0;
      }
      else {
        num_h_bytes = num_rd_bytes = dst.volume() * dst.grid()->atom_size();
        num_d_bytes = 0;
      }

      break;
    case XT_HOST_TO_HOST:
    case XT_DEVICE_TO_DEVICE:
      num_h_bytes = num_d_bytes = num_rd_bytes = 0;
      break;
    case XT_HOST_TO_DEVICE:
    case XT_DEVICE_TO_HOST:
      if (tbm == TBM_CONTIGUOUS) {
        num_d_bytes = num_rd_bytes = 0;
        num_h_bytes = contiguous_distance(src);   
      }
      else {
        num_h_bytes = num_d_bytes = dst.volume() * dst.grid()->atom_size();
        num_rd_bytes = 0;
      }
      break;
  }

  // create handle
  int handle;
  ExchangeMemoryHandle *hdl = mempool()->allocate_handle(num_h_bytes, num_d_bytes, num_rd_bytes, host_write_combined, remote_id, handle);
  hdl->dst3d = dst;
  hdl->src3d = src;
  hdl->dimension = 3;
  hdl->type = xt;
  hdl->method = tbm;

  ThreadManager::barrier();
  process_instance_requests();
  ThreadManager::barrier();

  if (!mempool()->is_handle_valid(handle)) {
    printf("[ERROR] CoArrayManager::barrier_allocate - allocation failed\n");
    return -1;
  }
  else
    return handle;
}



bool CoArrayManager::barrier_exchange()
{
  // participate in the barrier, handling requests to this thread, but otherwise doing nothing.
  // all previous exchanges must finish
  ThreadManager::io_fence();
  ThreadManager::barrier();

  ThreadManager::barrier();

  process_instance_requests();

  ThreadManager::io_fence();
  ThreadManager::barrier();

  ThreadManager::barrier();
  process_instance_requests();

  return true;
}

bool CoArrayManager::barrier_exchange(int handle)
{
  if (handle == -1)
    return barrier_exchange();

  bool ok = true;
  const ExchangeMemoryHandle *hdl = mempool()->get_handle(handle);
  if (!hdl) {
    printf("[ERROR] CoArrayManager::barrier_exchange - invalid handle\n");
    ok = false;
  }
  
  if (hdl->dimension != 1 && hdl->dimension != 3) {
    printf("[ERROR] CoArrayManager::barrier_exchange - handle dimension %d invalid\n", hdl->dimension);
    ok = false;
  }

  if (ok && hdl->dimension == 1) {
    return _barrier_exchange_1d(hdl->dst1d, hdl->src1d, hdl);
  }
  else if (ok && hdl->dimension == 3) {
    return _barrier_exchange_3d(hdl->dst3d, hdl->src3d, hdl);
  }
  else {
    // still participate in exchange to prevent deadlock.
    printf("[WARNING] CoArrayManager::barrier_exchange - invalid handle, participating in exchange anyway\n");
    barrier_exchange();
    return false;
  }
}


bool CoArrayManager::_barrier_exchange_3d(const Region3D &dst, const ConstRegion3D &src, const ExchangeMemoryHandle *hdl)
{
  // all previous exchanges must finish
  // This acts as a barrier.
  ThreadManager::io_fence();
  ThreadManager::barrier();

  // push requests for remote DtoH transfers first
  switch(hdl->type) {
    case XT_RDEVICE_TO_DEVICE:
      remote_xfer_dregion3d_to_hbuffer(src.imageid(), hdl->h_bytes, hdl->d_remote_bytes, src, hdl->method);
      break;
    case XT_RDEVICE_TO_HOST:
      remote_xfer_dregion3d_to_hbuffer(src.imageid(), hdl->h_bytes, hdl->d_remote_bytes, src, hdl->method);
      break;
  }

  ThreadManager::barrier();

  process_instance_requests();

  // do local xfers
  switch(hdl->type) {
    case XT_DEVICE_TO_RDEVICE:
      xfer_dregion3d_to_hbuffer(hdl->h_bytes, hdl->d_local_bytes, src, hdl->method);
      break;
    case XT_HOST_TO_RDEVICE:
      xfer_hregion3d_to_hbuffer(hdl->h_bytes, src, hdl->method);
      break;
    case XT_HOST_TO_HOST:
      xfer_hregion3d_to_hregion3d(dst, src, hdl->method);
      break;
    case XT_HOST_TO_DEVICE:
      //TODO: enable fast path
      xfer_hregion3d_to_hbuffer(hdl->h_bytes, src,hdl->method);
      xfer_hbuffer_to_dregion3d(dst, hdl->h_bytes, hdl->d_local_bytes, hdl->method);
      break;
    case XT_DEVICE_TO_HOST:
      // TODO: enable fast path
      xfer_dregion3d_to_hbuffer(hdl->h_bytes, hdl->d_local_bytes, src, hdl->method);
      break;
    case XT_DEVICE_TO_DEVICE:
      xfer_dregion3d_to_dregion3d(dst, src, hdl->method);
      break;

  }

  ThreadManager::io_fence();
  ThreadManager::barrier();

  // by this point, data is now transfered to host, and we have a pointer to this buffer

  // push requests for remote DtoH transfers
  switch(hdl->type) {
    case XT_DEVICE_TO_RDEVICE:
      remote_xfer_hbuffer_to_dregion3d(dst.imageid(), dst, hdl->h_bytes, hdl->d_remote_bytes, hdl->method);
      break;
    case XT_HOST_TO_RDEVICE:
      remote_xfer_hbuffer_to_dregion3d(dst.imageid(), dst, hdl->h_bytes, hdl->d_remote_bytes, hdl->method);
      break;
  }

  ThreadManager::barrier();


  process_instance_requests();

  // do local xfers
  switch(hdl->type) {
    case XT_RDEVICE_TO_DEVICE:
      xfer_hbuffer_to_dregion3d(dst, hdl->h_bytes, hdl->d_local_bytes, hdl->method);
      break;
    case XT_RDEVICE_TO_HOST:
      xfer_hbuffer_to_hregion3d(dst, hdl->h_bytes, hdl->method);
      break;
    case XT_DEVICE_TO_HOST:
      xfer_hbuffer_to_hregion3d(dst, hdl->h_bytes, hdl->method);
      break;
  }

  return true;
}


bool CoArrayManager::_barrier_exchange_1d(const Region1D &dst, const ConstRegion1D &src, const ExchangeMemoryHandle *hdl)
{
  // all previous exchanges must finish
  ThreadManager::io_fence();
  ThreadManager::barrier();

  // This acts as a barrier.

  // push requests for remote DtoH transfers first
  switch(hdl->type) {
    case XT_RDEVICE_TO_DEVICE:
      remote_xfer_dregion1d_to_hregion1d(src.imageid(), hdl->h_bytes, src.grid()->ptr_untyped(src.x0), hdl->h_size);
      break;
    case XT_RDEVICE_TO_HOST:
      remote_xfer_dregion1d_to_hregion1d(src.imageid(), dst.grid()->ptr_untyped(dst.x0), src.grid()->ptr_untyped(src.x0), dst.volume() * dst.grid()->atom_size());
      break;
  }

  ThreadManager::barrier();

  process_instance_requests();

  switch(hdl->type) {
    case XT_DEVICE_TO_RDEVICE:
      xfer_dregion1d_to_hregion1d(hdl->h_bytes, src.grid()->ptr_untyped(src.x0), hdl->h_size);
      break;
    case XT_HOST_TO_HOST:
      xfer_hregion1d_to_hregion1d(dst.grid()->ptr_untyped(dst.x0), src.grid()->ptr_untyped(src.x0), dst.volume() * dst.grid()->atom_size());
      break;
    case XT_HOST_TO_DEVICE:
      xfer_hregion1d_to_dregion1d(dst.grid()->ptr_untyped(dst.x0), src.grid()->ptr_untyped(src.x0), dst.volume() * dst.grid()->atom_size());
      break;
    case XT_DEVICE_TO_HOST:
      xfer_dregion1d_to_hregion1d(dst.grid()->ptr_untyped(dst.x0), src.grid()->ptr_untyped(src.x0), dst.volume() * dst.grid()->atom_size());
      break;
    case XT_DEVICE_TO_DEVICE:
      xfer_dregion1d_to_dregion1d(dst.grid()->ptr_untyped(dst.x0), src.grid()->ptr_untyped(src.x0), dst.volume() * dst.grid()->atom_size());
      break;
  }

  ThreadManager::io_fence();
  ThreadManager::barrier();

  // by this point, data is now transfered to host, and we have a pointer to this buffer

  // now transfer to appropriate device
  // push requests for remote DtoH transfers first
  switch(hdl->type) {
    case XT_DEVICE_TO_RDEVICE:
      remote_xfer_hregion1d_to_dregion1d(dst.imageid(), dst.grid()->ptr_untyped(dst.x0), hdl->h_bytes, hdl->h_size);
      break;
    case XT_HOST_TO_RDEVICE:
      remote_xfer_hregion1d_to_dregion1d(dst.imageid(), dst.grid()->ptr_untyped(dst.x0), src.grid()->ptr_untyped(src.x0), dst.volume() * dst.grid()->atom_size());
      break;
  }

  ThreadManager::barrier();

  process_instance_requests();

  switch(hdl->type) {
    case XT_RDEVICE_TO_DEVICE:
      xfer_hregion1d_to_dregion1d(dst.grid()->ptr_untyped(dst.x0), hdl->h_bytes, hdl->h_size);
      break;
  }

  return true;
}


void CoArrayManager::barrier_exchange_fence()
{
  ThreadManager::io_fence();
  ThreadManager::barrier();
}

} // end namespace 

