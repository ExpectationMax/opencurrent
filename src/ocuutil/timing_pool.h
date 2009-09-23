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

#ifndef __OCU_UTIL_TIMING_POOL_H__
#define __OCU_UTIL_TIMING_POOL_H__


namespace ocu {


void global_timer_clear_all();
void global_timer_add_timing(const char *name, float ms);
void global_timer_print();
void global_timer_add_flops(const char *name, float ms, int opcount);

void global_counter_clear_all();
void global_counter_add(const char *name, int value);
void global_counter_print();

} // end namespace


#endif

