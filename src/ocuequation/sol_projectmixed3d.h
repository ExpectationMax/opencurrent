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

#ifndef __OCU_EQUATION_PROJECT_MIXED_3D_H__
#define __OCU_EQUATION_PROJECT_MIXED_3D_H__

#include "ocuequation/sol_mgpressuremixed3d.h"
#include "ocuequation/sol_project3d.h"

namespace ocu {

#ifdef OCU_DOUBLESUPPORT

class Sol_ProjectDivergenceMixed3DDeviceD : public Sol_ProjectDivergence3DDeviceStorage<double> {
public:

  //**** PUBLIC STATE ****
  Sol_MultigridPressureMixed3DDeviceD pressure_solver;
  Sol_Divergence3DDevice<double> divergence_solver;
  Sol_Gradient3DDevice<double> gradient_solver;

  //**** MANAGERS ****
  Sol_ProjectDivergenceMixed3DDeviceD() { }
  ~Sol_ProjectDivergenceMixed3DDeviceD() { }

  //**** PUBLIC INTERFACE ****
  bool solve(double tolerance=1e-5);
  bool solve_divergence_only();

  bool initialize_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDeviceD *u, Grid3DDeviceD *v, Grid3DDeviceD *w);
};


#endif // OCU_DOUBLESUPPORT


} // end namespace



#endif

