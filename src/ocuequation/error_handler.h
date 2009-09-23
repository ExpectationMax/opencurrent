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

#ifndef __OCU_EQUATION_ERROR_HANDLER_H__
#define __OCU_EQUATION_ERROR_HANDLER_H__

#include <cstdio>

namespace ocu {




class ErrorHandler {

  bool _any_error;

protected:

  void check_ok(bool ok)  {
    _any_error = _any_error || !ok; 
  }
  
  void check_ok(bool ok, const char *msg) { 
    _any_error = _any_error || !ok; 
    if (!ok)
      printf("[WARNING] Failure: %s\n", msg);
  }

  void check_ok(bool ok, const char *filename, int lineno) {
    _any_error = _any_error || !ok; 
    if (!ok)
      printf("[WARNING] Failure in file \"%s\" at line %d\n", filename, lineno);
  }

  void add_error() { _any_error = true; }

public:

  ErrorHandler() : _any_error(false) { }

  void clear_error() { _any_error = false; }
  bool any_error() const { return _any_error; }
};

#define CHECK_OK(X) check_ok((X), __FILE__, __LINE__)

} // end namespace

#endif

