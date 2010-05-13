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

#ifndef __OCU_EQUATION_GRADIENT_3D_H__
#define __OCU_EQUATION_GRADIENT_3D_H__

#include "ocustorage/grid3d.h"
#include "ocuequation/solver.h"

namespace ocu {

template<typename T>
class Sol_Gradient3DDevice : public Solver {

  //**** MEMBER VARIABLES ****
  double _hx, _hy, _hz;

public:

  Sol_Gradient3DDevice();

  Grid3DDevice<T> *u, *v, *w; // updated in place
  Grid3DDevice<T> *phi;
  T coefficient;

  //**** PUBLIC INTERFACE ****
  // enforce_boundary_conditions must have been called first
  bool solve();

  bool initialize_storage(int nx, int ny, int nz, double hx, double hy, double hz, 
    Grid3DDevice<T> *u_val, Grid3DDevice<T> *v_val, Grid3DDevice<T> *w_val, Grid3DDevice<T> *phi_val);
};

} // end namespace


#endif

