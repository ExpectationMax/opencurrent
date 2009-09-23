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

#include <cstdio>
#include <algorithm>

#include "ocuutil/float_routines.h"
#include "ocuequation/sol_mgpressuremixed3d.h"



namespace ocu {

#ifdef OCU_DOUBLESUPPORT

void Sol_MultigridPressureMixed3DDeviceD::apply_boundary_conditions(int level)
{
  if (_double_mode)
    _mg_d.apply_boundary_conditions(level);
  else
    _mg_f.apply_boundary_conditions(level);
}

void Sol_MultigridPressureMixed3DDeviceD::relax(int level, int iterations)
{
  if (_double_mode)
    _mg_d.relax(level, iterations);
  else
    _mg_f.relax(level, iterations);
}

void Sol_MultigridPressureMixed3DDeviceD::restrict_residuals(int fine_level, int coarse_level, double *l2, double *linf)
{
  if (_double_mode)
    _mg_d.restrict_residuals(fine_level, coarse_level, l2, linf);
  else
    _mg_f.restrict_residuals(fine_level, coarse_level, l2, linf);
}

void Sol_MultigridPressureMixed3DDeviceD::prolong(int coarse_level, int fine_level)
{
  if (_double_mode)
    _mg_d.prolong(coarse_level, fine_level);
  else
    _mg_f.prolong(coarse_level, fine_level);
}

void Sol_MultigridPressureMixed3DDeviceD::clear_zero(int level)
{
  if (_double_mode)
    _mg_d.clear_zero(level);
  else
    _mg_f.clear_zero(level);
}

Sol_MultigridPressureMixed3DDeviceD::Sol_MultigridPressureMixed3DDeviceD()
{
  _double_mode = false;
}


Sol_MultigridPressureMixed3DDeviceD::~Sol_MultigridPressureMixed3DDeviceD()
{
}

bool Sol_MultigridPressureMixed3DDeviceD::do_fmg(double tolerance, int max_iter, double &result_l2, double &result_linf)
{
  // do float first
  if (!_rhs_f.copy_all_data(_mg_d.get_b(0)))
    add_failure();

  _double_mode = false;

  //Sol_MultigridPressure3DBase::do_fmg(std::max(tolerance, 1e-3), 4, result_l2, result_linf);
  Sol_MultigridPressure3DBase::do_fmg(std::max(tolerance, 1e-2), 4, result_l2, result_linf);
  _mg_d.get_u(0).copy_all_data(_mg_f.get_u(0));
  _double_mode = true;

  if (tolerance < 1e-2 || convergence == CONVERGENCE_L2) {
    bool ok = Sol_MultigridPressure3DBase::do_fmg(tolerance, max_iter, result_l2, result_linf);    
//    bool ok = Sol_MultigridPressure3DBase::do_vcycle(tolerance, max_iter, result_l2, result_linf);    
    _mg_f.get_u(0).copy_all_data(_mg_d.get_u(0));
    return ok;
  }

  return true;
}

bool Sol_MultigridPressureMixed3DDeviceD::initialize_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDevice<double> *rhs)
{
  _mg_f.bc = bc;
  _mg_d.bc = bc;

  _mg_f.nu1 = nu1;
  _mg_f.nu2 = nu2;

  _mg_d.nu1 = nu1;
  _mg_d.nu2 = nu2;

  _mg_f.convergence = CONVERGENCE_LINF;
  _mg_d.convergence = convergence;

  if (!_mg_d.initialize_storage(nx_val, ny_val, nz_val, hx_val, hy_val, hz_val, rhs)) {
    printf("[ERROR] Sol_MultigridPressureMixed3DDeviceD::initialize_storage - failed to initialize double solver\n");
    return false;
  }

  // init float version of RHS
  if (!_rhs_f.init_congruent(*rhs)) {
    printf("[ERROR] Sol_MultigridPressureMixed3DDeviceD::initialize_storage - could not initialize float rhs\n");
    return false;
  }

  if (!_mg_f.initialize_storage(nx_val, ny_val, nz_val, hx_val, hy_val, hz_val, &_rhs_f)) {
    printf("[ERROR] Sol_MultigridPressureMixed3DDeviceD::initialize_storage - failed to initialize float solver\n");
    return false;
  }

  if (!initialize_base_storage(nx_val, ny_val, nz_val, hx_val, hy_val, hz_val)) {
    printf("[ERROR] Sol_MultigridPressureMixed3DDeviceD::initialize_storage - failed to initialize base storage\n");
    return false;  
  }



  return true;
}

#endif // OCU_DOUBLESUPPORT


} // end namespace


