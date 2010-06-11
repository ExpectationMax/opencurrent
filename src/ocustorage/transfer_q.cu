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


void 
TransferRequestQ::process1d() 
{
#ifdef OCU_OMP
  omp_set_lock(&_lock);
#endif

  for (int i=0; i < _q1d.size(); i++) {
    TransferRequest1D d = _q1d[i];

    // handle the request
    switch(d.cmd) {
      case TRANSFER_HOST_TO_DEVICE:
        xfer_hregion1d_to_dregion1d(d.dst, d.src, d.num_bytes);
        break;
      case TRANSFER_DEVICE_TO_HOST:
        xfer_dregion1d_to_hregion1d(d.dst, d.src, d.num_bytes);
        break;
      default:
        printf("[ERROR] ransferRequestQ::process1d - invalid cmd %d\n", (unsigned int)d.cmd);
    }
  }

  _q1d.clear();
#ifdef OCU_OMP
  omp_unset_lock(&_lock);
#endif
}

void 
TransferRequestQ::process3d() 
{
#ifdef OCU_OMP 
  omp_set_lock(&_lock);
#endif

  for (int i=0; i < _q3d.size(); i++) {
    TransferRequest3D d = _q3d[i];
    // handle the request
    switch(d.cmd) {
      case TRANSFER_HOSTBUFFER_TO_DEVICE:
        xfer_hbuffer_to_dregion3d(d.dst, d.host_buffer, d.device_buffer, d.method);
        break;
      case TRANSFER_DEVICE_TO_HOSTBUFFER:
        xfer_dregion3d_to_hbuffer(d.host_buffer, d.device_buffer, d.src, d.method);
        break;
      default:
        printf("[ERROR] ransferRequestQ::process3d - invalid cmd %d\n", (unsigned int)d.cmd);
    }
  }

  _q3d.clear();

#ifdef OCU_OMP 
  omp_unset_lock(&_lock);
#endif
}

void 
TransferRequestQ::processalloc() 
{
#ifdef OCU_OMP 
  omp_set_lock(&_lock);
#endif

  for (int i=0; i < _qalloc.size(); i++) {
    TransferRequestAlloc d = _qalloc[i];
    // handle the request
    switch(d.cmd) {
      case TRANSFER_ALLOCATE:
        {
          cudaError_t ok = cudaMalloc(d.result, d.num_bytes);
          if (ok != cudaSuccess)
            printf("[ERROR] TransferRequestQ::processalloc - cudaMalloc failed: %s\n", cudaGetErrorString(ok));
          else
            *d.valid = true;
        }
        break;
      case TRANSFER_FREE:
        {
          cudaFree(d.ptr);
        }
        break;
      default:
        printf("[ERROR] ransferRequestQ::processalloc - invalid cmd %d\n", (unsigned int)d.cmd);
    }
  }

  _qalloc.clear();
#ifdef OCU_OMP 
  omp_unset_lock(&_lock);
#endif
}



} // end namespace
