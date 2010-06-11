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


#include <cstdio>

#include "ocuutil/thread.h"

namespace ocu {



ThreadManager ThreadManager::s_manager;

ThreadManager::ThreadManager()
{
  _valid = false;
  _num_images = 1;
  for (int i=0; i < OCU_MAX_IMAGES; i++) {
    _compute_streams[i] = 0;
    _io_streams[i] = 0;
  }
}

ThreadManager::~ThreadManager()
{
}


bool ThreadManager::_initialize(int num_images_val)
{
  if (num_images_val > OCU_MAX_IMAGES) {
    printf("[ERROR] ThreadManager::_initialize - cannot initialize %d threads, maximum value is %d\n", num_images_val, OCU_MAX_IMAGES);
    return false;
  }

#ifdef OCU_OMP
  omp_set_num_threads(num_images_val);
#endif

  _num_images = num_images_val;
  _valid = true;

  return true;
}


bool ThreadManager::_initialize_image(int gpu_id)
{
  if (!_valid) {
    printf("[ERROR] ThreadManager::initialize_image - ThreadManager must be initialized first\n");
    return false;
  }

  cudaError_t ok = cudaSetDevice(gpu_id);
  if (ok != cudaSuccess) {
    printf("[ERROR] ThreadManager::initialize_image - cudaSetDevice(%d) failed on image %d: %s\n", gpu_id, this_image(), cudaGetErrorString(ok));
    return false;
  }

  // initialize streams
  if (cudaStreamCreate(&_compute_streams[this_image()]) != cudaSuccess) {
    printf("[ERROR] ThreadManager::_initialize_image - cudaStreamCreate failed on compute stream: %s\n", cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  if (cudaStreamCreate(&_io_streams[this_image()]) != cudaSuccess) {
    printf("[ERROR] ThreadManager::_initialize_image - cudaStreamCreate failed on io stream: %s\n", cudaGetErrorString(cudaGetLastError()));
    return false;
  }

  return true;
}


int ThreadManager::this_image()
{
#ifdef OCU_OMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

int ThreadManager::num_images()
{
  return manager()._num_images;
}

void ThreadManager::barrier()
{
#ifdef OCU_OMP

#pragma omp barrier
  {
  }

#endif
}

void ThreadManager::_compute_fence()
{
  if (cudaStreamSynchronize(get_compute_stream()) != cudaSuccess)
    printf("[ERROR] ThreadManager::_compute_fence - cudaStreamSynchronize failed with '%s'\n", cudaGetErrorString(cudaGetLastError()));

}

void ThreadManager::_io_fence()
{
  if (cudaStreamSynchronize(get_io_stream()) != cudaSuccess)
    printf("[ERROR] ThreadManager::_io_fence - cudaStreamSynchronize failed with '%s'\n", cudaGetErrorString(cudaGetLastError()));
}


} // end namespace

