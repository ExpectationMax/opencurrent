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

#include "cuda.h"
#include <cstdio>
#include "ocuutil/memory.h"



namespace ocu {


void *host_malloc(size_t bytes, bool pinned)
{
  if (!pinned)
    return malloc(bytes);
  else {
    void *result;
    
    if (cudaMallocHost(&result, bytes) != (unsigned int)CUDA_SUCCESS) {
      printf("[ERROR] host_malloc - failed with cudaError \"%s\"\n", cudaGetErrorString(cudaGetLastError()));
      return 0;
    }

    return result;
  }
    
}

void host_free(void *ptr, bool pinned)
{
  if (!pinned)
    free(ptr);
  else {
    cudaFreeHost(ptr);
  }
}



} // end namespace

