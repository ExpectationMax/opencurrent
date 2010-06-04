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





ExchangeMemoryHandle * ExchangeMemoryPool::allocate_handle(size_t num_host_bytes, size_t num_local_device_bytes, size_t num_remote_device_bytes, bool host_write_combined, int remote_id, int &handle)
{
  ExchangeMemoryHandle *hdl = new ExchangeMemoryHandle();
  bool is_valid = true;

  hdl->h_size = num_host_bytes;
  if (hdl->h_size > 0) {
    hdl->h_bytes = host_malloc(hdl->h_size, true, host_write_combined); 
    if (!hdl->h_bytes) {
      printf("[ERROR] ExchangeMemoryPool::create_handle - host_malloc failed\n");
      is_valid = false;
    }
  }
  else
    hdl->h_bytes = 0;

  hdl->d_local_size = num_local_device_bytes;
  if (hdl->d_local_size > 0) {
    cudaError_t ok = cudaMalloc((void **)&hdl->d_local_bytes, hdl->d_local_size); 
    if (ok != cudaSuccess) {
      printf("[ERROR] ExchangeMemoryPool::create_handle - cudaMalloc() failed: %s\n", cudaGetErrorString(ok));
      is_valid = false;
    }
  }
  else
    hdl->d_local_bytes = 0;

  hdl->d_remote_size = num_remote_device_bytes;
  hdl->d_remote_bytes = 0;
  if (hdl->d_remote_size > 0) {
    
    TransferRequestAlloc req;
    req.result = &hdl->d_remote_bytes;
    req.valid = &hdl->valid;
    req.num_bytes = hdl->d_remote_size;
    req.cmd = TRANSFER_ALLOCATE;
 
    CoArrayManager::xfer_q(remote_id)->push(req);

    // if we requested a remote allocation, we are not valid until it is completed.
    is_valid = false;
    hdl->remote_id = remote_id;
  }

  hdl->valid = is_valid;
  _handles.push_back(hdl);

  handle = _handles.size() - 1;
  return hdl;
}

void ExchangeMemoryPool::remove_handle(int handle)
{

  const ExchangeMemoryHandle *hdl = get_handle(handle);
  if (!hdl) {
    printf("[WARNING] ExchangeMemoryPool::remove_handle - invalid handle\n");
  }

  if (hdl && hdl->h_size > 0) {
    host_free(hdl->h_bytes, true);
  }
  if (hdl && hdl->d_local_size > 0) {
    cudaFree(hdl->d_local_bytes);
  }
  if (hdl && hdl->d_remote_size > 0 && hdl->remote_id != -1) {
    TransferRequestAlloc req;
    req.cmd = TRANSFER_FREE;
    req.ptr = hdl->d_remote_bytes;
 
    CoArrayManager::xfer_q(hdl->remote_id)->push(req);
  }

  if (hdl) {
    _handles[handle] = 0;
    delete hdl;
  }
}


void ExchangeMemoryPool::cleanup() 
{
  // everything must be finished
  ThreadManager::io_fence();
  ThreadManager::barrier();

  // this will leave memory on remote devices hanging.  But app should free all handles explicitly to prevent this anyway.
  // TODO: can we do better? Yes, if each image maintains a list of remote-requested allocations so they can be cleaned up later.

  for (int i=0; i < _handles.size(); i++) {
    if (_handles[i] && _handles[i]->valid) {
      host_free(_handles[i]->h_bytes, true);
      cudaFree(_handles[i]->d_local_bytes);
      delete _handles[i];
      _handles[i] = 0;
    }
  }

  _handles.clear();
}

} // end namespace
