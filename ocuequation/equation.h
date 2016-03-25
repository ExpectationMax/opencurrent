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

#ifndef __OCU_EQUATION_EQUATION_H__
#define __OCU_EQUATION_EQUATION_H__

#include <vector>

#include "ocuequation/error_handler.h"

namespace ocu {



class Equation : public ErrorHandler {

public:

  Equation(){ }

  virtual ~Equation();

  virtual bool advance(double dt);

  virtual bool advance_one_step(double dt) = 0;
  virtual double get_max_stable_timestep() const = 0;  

};





} // end namespace

#endif

