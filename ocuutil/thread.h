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


#ifndef __OCU_UTIL_THREAD_H__
#define __OCU_UTIL_THREAD_H__

#ifdef OCU_OMP
#include <omp.h>
#endif

#include <driver_types.h>

#include "ocuutil/defines.h"


namespace ocu {


class ThreadManager {
private:

  static ThreadManager s_manager;

  bool _valid;
  int _num_images;

  cudaStream_t _io_streams[OCU_MAX_IMAGES];
  cudaStream_t _compute_streams[OCU_MAX_IMAGES];
  double _scratch_space[OCU_MAX_IMAGES];


  ThreadManager();
  ~ThreadManager();

  bool _initialize(int num_images_val);
  bool _initialize_image(int gpu_id=this_image());
  void _shutdown();

  void _compute_fence();
  void _io_fence();

  cudaStream_t _get_compute_stream(int image_id) const { return _compute_streams[image_id]; }
  cudaStream_t _get_io_stream(int image_id) const { return _io_streams[image_id]; }

  template <typename T>
  void read_scratch_space(T &result, int image_id) const {
    result = * reinterpret_cast<const T *>(& (_scratch_space[image_id]));
  }

  template <typename T>
  void write_scratch_space(const T &result, int image_id) {
    * reinterpret_cast<T *>(& (_scratch_space[image_id])) = result;
  }

public:


  static ThreadManager &manager() { return s_manager; }

  static int this_image();
  static int num_images();
  static void barrier();
  static void barrier_and_fence() { io_fence(); compute_fence(); barrier(); }

  static bool initialize(int num_images_val) { return manager()._initialize(num_images_val); }
  static bool initialize_image(int gpu_id=this_image()) { return manager()._initialize_image(gpu_id); }
  static void shutdown() { manager()._shutdown(); }

  static void compute_fence() { manager()._compute_fence(); }
  static void io_fence() { manager()._io_fence(); }

  static cudaStream_t get_io_stream(int image_id=this_image()) { return manager()._get_io_stream(image_id); }
  static cudaStream_t get_compute_stream(int image_id=this_image()) { return manager()._get_compute_stream(image_id); }

  template<typename T, typename REDUCE>
  static T barrier_reduce(const T &val, REDUCE reduce) {
    manager().write_scratch_space(val, this_image());
    barrier();
    T result, tmp;
    manager().read_scratch_space(result, 0);
    result = reduce.process(result);
    for (int i=1; i < num_images(); i++) {
      manager().read_scratch_space(tmp, i);
      result = reduce(reduce.process(tmp), result);
    }
    barrier();
    return result;
  }
};


} // end namespace


#endif
