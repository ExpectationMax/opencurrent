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

#include <vector>

#include "ocustorage/coarray_internal.h"

namespace ocu {



CoArrayManager CoArrayManager::s_manager;

CoArrayManager::CoArrayManager()
{
#ifdef OCU_OMP
  omp_init_lock(&_lock);
#endif
  _valid = false;
  _num_images = 1;
}

CoArrayManager::~CoArrayManager()
{
  // clean up tables?  Or memory leak?
#ifdef OCU_OMP
  omp_destroy_lock(&_lock);
#endif
}


bool CoArrayManager::_initialize(int num_images)
{

  for (int i=0; i < num_images; i++) {
    _transfers[i] = new TransferRequestQ(i);
    _mempools[i] = new ExchangeMemoryPool();
  }

  _num_images = num_images;
  _valid = true;


  return true;
}

bool CoArrayManager::_initialize_image(int gpu_id)
{
  // nothing to do yet...


  return true;
}

CoArrayTable *CoArrayManager::_register_coarray(const char *name, int image_id, void *coarray)
{
  // grab mutex
  // check if table exists
  // if not, add it
  // release mutex
  // add ptr to slot
  // barrier - implicit when co array is created
#ifdef OCU_OMP
  omp_set_lock(&_lock);
#endif
  std::map<std::string, CoArrayTable *>::iterator iter = _coarrays.find(name);
  if (iter == _coarrays.end()) {
    iter = _coarrays.insert(std::pair<std::string, CoArrayTable *>(name, new CoArrayTable())).first;
    iter->second->name = name;
  }
  iter->second->table[image_id] = coarray;
#ifdef OCU_OMP
  omp_unset_lock(&_lock);
#endif

  ThreadManager::barrier();

  return iter->second;
}

void CoArrayManager::_unregister_coarray(const std::string &name, int image_id)
{
  // zero ptr
  // grab mutex
  // if entry is all empty, remove it
  // release mutex
#ifdef OCU_OMP
  omp_set_lock(&_lock);
#endif
  std::map<std::string, CoArrayTable *>::iterator iter = _coarrays.find(name);
  bool not_found = iter == _coarrays.end();

#ifdef OCU_OMP
  omp_unset_lock(&_lock);
#endif

  if (not_found) {
    return;
  }

#ifdef OCU_OMP
  omp_set_lock(&_lock);
#endif

  iter->second->table[image_id] = 0;
  bool empty = true;
  for (int i=0; i < OCU_MAX_IMAGES; i++)
    if (iter->second->table[i])
      empty = false;
  if (empty) {
    delete iter->second;
    _coarrays.erase(iter);
  }
#ifdef OCU_OMP
  omp_unset_lock(&_lock);
#endif
}


void CoArrayManager::_process_instance_requests()
{
  _transfers[ThreadManager::this_image()]->process();
}

} // end namespace

