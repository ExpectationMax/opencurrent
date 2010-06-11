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

#ifndef __OCU_STORAGE_COARRAY_H__
#define __OCU_STORAGE_COARRAY_H__

#include <map>
#include <string>
#include <vector>
#include "ocuutil/thread.h"
#include "ocustorage/grid1d.h"
#include "ocustorage/grid3d.h"


namespace ocu {



struct ExchangeMemoryHandle;
class ExchangeMemoryPool;
class TransferRequestQ;


struct CoArrayTable {
  // do I want to store type info also?
  CoArrayTable() { 
    for (int i=0; i < OCU_MAX_IMAGES; i++)
      table[i] = 0;
  }

  std::string name;
  void *table[OCU_MAX_IMAGES];
};


// TODO: maybe this should be subclass of ThreadManager - how to handle singleton?
class CoArrayManager {
private:

  static CoArrayManager s_manager;
  
  std::map<std::string, CoArrayTable *> _coarrays;
  TransferRequestQ  *_transfers[OCU_MAX_IMAGES];
  ExchangeMemoryPool *_mempools[OCU_MAX_IMAGES];

#ifdef OCU_OMP
  omp_lock_t _lock;
#endif

  bool _valid;
  int _num_images;

  CoArrayManager();
  ~CoArrayManager();

  bool          _initialize(int num_images);
  bool          _initialize_image(int gpu_id);
  bool          _shutdown();

  CoArrayTable *_register_coarray(const char *name, int image_id, void *coarray);
  void          _unregister_coarray(const std::string &name, int image_id);

  void          _process_instance_requests();

  ExchangeMemoryPool *_mempool(int image_id) { return _mempools[image_id]; }
  TransferRequestQ   *_xfer_q(int image_id) { return _transfers[image_id]; }

  static CoArrayManager &manager() { return s_manager; }

  static bool _barrier_exchange_3d(const Region3D &dst, const ConstRegion3D &src, const ExchangeMemoryHandle *hdl);
  static bool _barrier_exchange_1d(const Region1D &dst, const ConstRegion1D &src, const ExchangeMemoryHandle *hdl);

public:

  static CoArrayTable *register_coarray(const char *name, int image_id, void *coarray) { return manager()._register_coarray(name, image_id, coarray); }
  static void          unregister_coarray(const std::string &name, int image_id) { manager()._unregister_coarray(name, image_id); }

  static void          process_instance_requests() { manager()._process_instance_requests(); }

  static TransferRequestQ   *xfer_q(int image_id)  { return manager()._xfer_q(image_id); }
  static ExchangeMemoryPool *mempool()             { return manager()._mempool(ThreadManager::this_image()); }
  static bool          initialize(int num_images)  { return manager()._initialize(num_images); }
  static bool          initialize_image(int gpu_id=ThreadManager::this_image()) { return manager()._initialize_image(gpu_id); }
  static void          shutdown() { manager()._shutdown(); }


  static int  barrier_allocate();
  static int  barrier_allocate(const Region1D &dst, const ConstRegion1D &src);
  static int  barrier_allocate(const Region3D &dst, const ConstRegion3D &src);

  static bool barrier_exchange();
  static bool barrier_exchange(int handle);

  static void barrier_deallocate();
  static void barrier_deallocate(int handle);

  static void barrier_exchange_fence();


};




} // end namespace

#endif
