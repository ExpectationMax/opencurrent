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

#ifndef __OCU_EQUATION_MULTIGRID_PRESSURE_MIXED_3D_H__
#define __OCU_EQUATION_MULTIGRID_PRESSURE_MIXED_3D_H__

#include "ocuequation/sol_mgpressure3d.h"


namespace ocu {

#ifdef OCU_DOUBLESUPPORT

// mixed precision, "drop in" replacement for DP version.
class Sol_MultigridPressureMixed3DDeviceD : public Sol_MultigridPressure3DBase {
  Sol_MultigridPressure3DDevice<float> _mg_f;
  Sol_MultigridPressure3DDevice<double> _mg_d;
  Grid3DDevice<float> _rhs_f;

  bool  _double_mode; // true = double, false = float

  //***** OVERRIDES ****
  virtual void apply_boundary_conditions(int level);
  virtual void relax(int level, int iterations);            
  virtual void restrict_residuals(int fine_level, int coarse_level, double *l2, double *linf);
  virtual void prolong(int coarse_level, int fine_level);
  virtual void clear_zero(int level);
  virtual bool do_fmg(double tolerance, int max_iter, double &result_l2, double &result_linf);

public:

  //**** MANAGERS ****
  Sol_MultigridPressureMixed3DDeviceD();
  ~Sol_MultigridPressureMixed3DDeviceD();

  //**** PUBLIC INTERFACE ****
  bool initialize_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDevice<double> *rhs);
  //bool solve(double &residual, double tolerance = 1e-6, int max_iter = 4);

  Grid3DDevice<double>       &pressure()       { return _mg_d.pressure(); }
  const Grid3DDevice<double> &pressure() const { return _mg_d.pressure(); }
};

#endif // OCU_DOUBLESUPPORT


} // end namespace

#endif

