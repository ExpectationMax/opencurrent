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

#include "ocuutil/defines.h"
#include "ocuutil/timer.h"
#include "ocuutil/timing_pool.h"

#include <map>
#include <string>
#include <cstdio>


namespace ocu {

class counter_pool {
    struct counter {
      counter() { total = 0; count = 0; }
      int total;
      int count;
    };

    std::map<std::string, counter> _ctr;

public:

    void clear_all() {
      _ctr.clear();
    }

    void add_counter(const char *name, int value) {
        if (_ctr.count(name)) {
            _ctr[name].total += value;
            _ctr[name].count ++;
        }
        else {
            _ctr[name].total = value;
            _ctr[name].count = 1;
        }
    }

    void print() {        
        printf("\nCounters\n");
        for (std::map<std::string, counter>::iterator f = _ctr.begin(); f != _ctr.end(); ++f) {
            printf("%s: avg %f (total %d, count %d)\n", f->first.c_str(), ((float)f->second.total) / ((float)f->second.count), f->second.total, f->second.count);
        }
    }
};

class timing_pool
{
    struct entry {            
      float total;
      int count;

      entry() { total = 0; count = 0; }
    };

    struct flopsentry {
      double total;
      double count;

      flopsentry() { total = 0; count = 0; }
    };



    std::map<std::string, entry> _table;
    std::map<std::string, flopsentry> _flops;

    float _alltime;

public:

    timing_pool() : _alltime(0) { }

    void clear_all() {
      _table.clear();
      _flops.clear();
      _alltime = 0;
    }

    void add_flops(const char *name, float ms, int opcount) {
        if (_flops.count(name)) {
            _flops[name].total += (ms*.001);
            _flops[name].count += opcount;
        }
        else {
            _flops[name].total = (ms*.001);
            _flops[name].count = opcount;
        }
    }



    void add_timing(const char *name, float ms) {
        if (_table.count(name)) {
            _table[name].total += ms;
            _table[name].count++;
        }
        else {
            _table[name].total = ms;
            _table[name].count = 1;
        }
        _alltime += ms;
    }

    void print() {
        std::multimap<float, std::string> fraction;
        printf("\n");
        for (std::map<std::string, entry>::iterator i = _table.begin(); i != _table.end(); ++i) {
            printf("%s: \tTotal %f, Count %d, Avg %f\n", i->first.c_str(), i->second.total, i->second.count, i->second.total / i->second.count);
            fraction.insert(std::make_pair(100.0f * i->second.total / _alltime, i->first));
        }
        
        printf("\nFLOPS\n");
        for (std::map<std::string, flopsentry>::iterator f = _flops.begin(); f != _flops.end(); ++f) {
            printf("%s: \tGFLOPS %f\n", f->first.c_str(), 1e-9 * (f->second.count / f->second.total));
        }

        printf("\nTotal %fms\n---------------------\n", _alltime);
        for (std::multimap<float, std::string>::reverse_iterator it = fraction.rbegin(); it != fraction.rend(); ++it) {
            printf(" %.02f%% - %s\n", it->first, it->second.c_str()); 
        }
    }

};

counter_pool g_global_ctr;
timing_pool g_global_timer;
static bool g_timing_enabled = true;

void global_timer_clear_all() {
  g_global_timer.clear_all();
}

void global_timer_add_timing(const char *name, float ms) {
  if (g_timing_enabled)
    g_global_timer.add_timing(name, ms);
}

void global_timer_print() {
  g_global_timer.print();
#ifndef OCU_ENABLE_GPU_TIMING_BY_DEFAULT
  printf("[WARNING] global_timer_print - Data may be incomplete - GPU timing is disabled.  #define OCU_ENABLE_GPU_TIMING_BY_DEFAULT to enable.\n");
#endif //OCU_ENABLE_GPU_TIMING_BY_DEFAULT
}

void global_timer_add_flops(const char *name, float ms, int opcount) {
  if (g_timing_enabled)
    g_global_timer.add_flops(name, ms, opcount);
}

void global_counter_add(const char *name, int value)
{
  g_global_ctr.add_counter(name, value);
}

void global_counter_clear_all() {
  g_global_ctr.clear_all();
}

void global_counter_print() {
  g_global_ctr.print();
}

void disable_timing()
{
    g_timing_enabled = false;
}


void enable_timing()
{
    g_timing_enabled = true;
}

} // end namespace

