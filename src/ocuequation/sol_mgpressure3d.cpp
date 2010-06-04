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

#include "ocuutil/float_routines.h"
#include "ocuutil/timer.h"
#include "ocuutil/timing_pool.h"
#include "ocuequation/sol_mgpressure3d.h"

namespace ocu {

Sol_MultigridPressure3DBase::Sol_MultigridPressure3DBase()
{
  _num_levels = 0;

  _h = 0;
  _dim = 0;
  _omega = 1.0f;

  nu1 = 2;
  nu2 = 2;

  convergence = CONVERGENCE_L2;
  make_symmetric_operator = false;

}

Sol_MultigridPressure3DBase::~Sol_MultigridPressure3DBase()
{
  delete[] _h;
  delete[] _dim;
}


double 
Sol_MultigridPressure3DBase::bc_diag_mod(const BoundaryCondition &local_bc, double factor) const
{
  if (local_bc.type == BC_PERIODIC)
    return 0;
  else if (local_bc.type == BC_NONE)
    return 0;
  else if (local_bc.type == BC_DIRICHELET) 
    return factor;
  else if (local_bc.type == BC_NEUMANN)
    return -factor;
  else {
    printf("[WARNING] Sol_MultigridPressure3DBase::bc_diag_mod - unknown boundary condition type %d\n", local_bc.type);
    return 0;
  }
}


int 
Sol_MultigridPressure3DBase::num_levels(int nx_val, int ny_val, int nz_val)
{
  int cnx = nx_val/2;
  int cny = ny_val/2;
  int cnz = nz_val/2;
  int levels = 0;
  while (2*cnx == nx_val && 2*cny == ny_val && 2*cnz == nz_val) {
    levels++;
    nx_val = cnx;
    ny_val = cny;
    nz_val = cnz;

    cnx /= 2;
    cny /= 2;
    cnz /= 2;

    if (cnx < 2 || nx_val == 5 || nx_val == 7)
      break;
    if (cny < 2 || ny_val == 5 || ny_val == 7)
      break;
    if (cnz < 2 || nz_val == 5 || nz_val == 7)
      break;
  }
  return levels;
}

double 
Sol_MultigridPressure3DBase::optimal_omega(double fx, double fy, double fz)
{
  // Equation is from:
  // Irad Yavneh. On red-black SOR smoothing in multigrid. SIAM J. Sci. Comput., 17(1):180-192, 1996.

  // The paper is written in terms of a Poisson equation f_x u_xx + f_y u_yy + f_z u_zz = b
  // In our case, f_x = f_y = f_z = 1, but the grid spacings are different.
  // Via a change of coordinate transform, this is equivalent to making the grid
  // spacings all the same (h), and setting f_x = (h/hx)^2, f_y = (h/hy)^2, f_z = (h/hz)^2,

  double f_min = min3(fx, fy, fz);
  double c_max = (1.0 - f_min/(fx+fy+fz));
  double c_max_sqr = c_max * c_max;
  return 2.0/(1.0+sqrt(1.0f-c_max_sqr));
}



bool
Sol_MultigridPressure3DBase::do_fmg(double tolerance, int max_iter, double &result_l2, double &result_linf)
{
  CPUTimer timer;
  timer.start();

  clear_error();

  int level_ncyc;
  int level;

  result_l2 = result_linf = 0;

  apply_boundary_conditions(0);
  double orig_l2=0, orig_linf=0;
  restrict_residuals(0,0, 
    (convergence & CONVERGENCE_CALC_L2) ? &orig_l2 : 0, 
    (convergence & CONVERGENCE_CALC_LINF) ? &orig_linf : 0);
  //printf("Error Before: l2 = %f, linf = %f\n", orig_l2, orig_linf);

  double orig_error = (convergence & CONVERGENCE_CRITERIA_L2) ? orig_l2 : 
                      (convergence & CONVERGENCE_CRITERIA_LINF) ? orig_linf : 1e20;
  if (orig_error < tolerance) {
    if (convergence & CONVERGENCE_CALC_L2)   result_l2 = orig_l2;
    if (convergence & CONVERGENCE_CALC_LINF) result_linf = orig_linf;
    return true;
  }

#if 0
  // for testing relaxation only, enable this code block
  double iter_l2, iter_linf;
  for (int o=0; o < 100; o++) {
    relax(0, 10, RO_SYMMETRIC);
    restrict_residuals(0, 0, &iter_l2, &iter_linf);
    printf("error: l2 = %.12f, linf = %.12f\n", iter_l2, iter_linf);
  }
  printf("reduction: l2 = %f, linf = %f\n", orig_l2/iter_l2, orig_linf/iter_linf);
  result_l2 = iter_l2;
  result_linf = iter_linf;

  return true;
#endif

  // initialize all the residuals.
  // we need this because in the FMG loop below, we don't necessarily start at level 0, but 
  // rather 2 levels from the finest.  Which means we first need to propagate the errors all the way down first before
  // beginning FMG.

  int coarse_level = _num_levels-1;
  int num_vcyc = 0;

  for (level = 0; level < _num_levels-1; level++) {
    // initialize U (solution) at next level to zero
    clear_zero(level+1);
    apply_boundary_conditions(level+1);

    // restrict residuals to the next level.
    restrict_residuals(level, level+1,0,0);
  }

  // do the full-multigrid loop
  for (int fine_level = _num_levels-1; fine_level >= 0 ; fine_level--) {
  //{ int fine_level = 0; // do a single v-cycle instead

    // we always do one extra v-cycle
    level_ncyc = (fine_level == 0) ? max_iter+1 : 1;

    // do ncyc v-cycle's
    for (int i_cyc = 0; i_cyc < level_ncyc; i_cyc++) {

      if (fine_level == 0)
        num_vcyc++;

      // going down
      for (level = fine_level; level < coarse_level; level++) {
        relax(level, nu1, RO_RED_BLACK);

        clear_zero(level+1);
        apply_boundary_conditions(level+1);


        if (level == 0) {
          restrict_residuals(0, 1, 
            (convergence & CONVERGENCE_CALC_L2) ? &result_l2 : 0, 
            (convergence & CONVERGENCE_CALC_LINF) ? &result_linf : 0);
          double residual = (convergence & CONVERGENCE_CRITERIA_L2) ? result_l2 : 
                            (convergence & CONVERGENCE_CRITERIA_LINF) ? result_linf : 1e20;          

          if (ThreadManager::this_image() == 0)
            printf("%d: residual = %.12f,%.12f\n", i_cyc, result_linf, result_l2);

          // if we're below tolerance, or we're no longer converging, bail out
          if (residual < tolerance) {

            // last time through, we need to apply boundary condition to u[0], since we just relaxed (above), but haven't propagated changes to ghost cells.
            // in the case we are not finished, by the time we get back to level 0 (via coarsening & then prolongation), ghost cells would be filled in.
            // but since we are bailing here, we need to explicitly make ghost cells up-to-date with u.
            apply_boundary_conditions(0);

            timer.stop();
            //printf("[ELAPSED] Sol_MultigridPressure3DBase::do_fmg - converged in %fms\n", timer.elapsed_ms());
            printf("[INFO] Sol_MultigridPressure3DBase::do_fmg - error after %d iterations: L2 = %f (%fx), Linf = %f (%fx)\n", i_cyc, result_l2, orig_l2 / result_l2, result_linf, orig_linf / result_linf);
            global_counter_add("vcycles", num_vcyc);
            return !any_error();
          }
        }
        else
          restrict_residuals(level, level+1, 0, 0);
      }

      // these relaxation steps are essentially free, so do lots of them
      // (reference implementation uses nu1+nu2) - this is probably overkill, i need to revisit this
      // with a good heuristic.  Inhomogeneous conditions require more iterations.
      int coarse_iters = max3(nx(coarse_level)*ny(coarse_level), ny(coarse_level)*nz(coarse_level), nx(coarse_level)*nz(coarse_level))/2;      
      relax(coarse_level, coarse_iters, make_symmetric_operator ? RO_SYMMETRIC : RO_RED_BLACK);
      //relax(coarse_level, (nx(coarse_level)*ny(coarse_level)*nz(coarse_level))/2);

      // going up
      for (level = coarse_level-1; level >= fine_level; level--) {
        prolong(level+1, level);
        // don't need to relax finest grid since it will get relaxed at the beginning of the next v-cycle
        if (level > 0) 
          relax(level, nu2, make_symmetric_operator ? RO_BLACK_RED : RO_RED_BLACK);
      }
    }

    if (fine_level > 0) {
      // if not at finest level, need to prolong once more to next finer level for the next fine_level value
      prolong(fine_level, fine_level-1);
    }
  }

  timer.stop();
  //printf("[ELAPSED] Sol_MultigridPressure3DBase::do_fmg - stopped iterations after %fms\n", timer.elapsed_ms());
  if (!(convergence & CONVERGENCE_CRITERIA_NONE))
    printf("[WARNING] Sol_MultigridPressure3DBase::do_fmg - Failed to converge, error after: L2 = %.12f (%fx), Linf = %.12f (%fx)\n", result_l2, orig_l2 / result_l2, result_linf, orig_linf / result_linf);

  return false;
}

bool
Sol_MultigridPressure3DBase::do_vcycle(double tolerance, int max_iter, double &result_l2, double &result_linf)
{
  clear_error();

  int level;
  int coarse_level = _num_levels-1;

  apply_boundary_conditions(0);

  double orig_l2=0, orig_linf=0;
  restrict_residuals(0,0, 
    (convergence & CONVERGENCE_CALC_L2) ? &orig_l2 : 0, 
    (convergence & CONVERGENCE_CALC_LINF) ? &orig_linf : 0);
  //printf("Error Before: l2 = %f, linf = %f\n", orig_l2, orig_linf);

  double orig_error = (convergence & CONVERGENCE_CRITERIA_L2) ? orig_l2 : 
                      (convergence & CONVERGENCE_CRITERIA_LINF) ? orig_linf : 1e20;
  if (orig_error < tolerance) {
    if (convergence & CONVERGENCE_CALC_L2)   result_l2 = orig_l2;
    if (convergence & CONVERGENCE_CALC_LINF) result_linf = orig_linf;
    return true;
  }

  for (int i_cyc = 0; i_cyc < max_iter; i_cyc++) {

    // going down
    for (level = 0; level < coarse_level; level++) {
      relax(level, nu1, RO_RED_BLACK);

      clear_zero(level+1);
      apply_boundary_conditions(level+1);
      if (level == 0) {
        restrict_residuals(0, 1, 
          (convergence & CONVERGENCE_CALC_L2) ? &result_l2 : 0, 
          (convergence & CONVERGENCE_CALC_LINF) ? &result_linf : 0);
        double residual = (convergence & CONVERGENCE_CRITERIA_L2) ? result_l2 : 
                          (convergence & CONVERGENCE_CRITERIA_LINF) ? result_linf : 1e20;      

        //printf("%d: residual = %.12f\n", i_cyc, result_linf);

        // if we're below tolerance, or we're no longer converging, bail out
        if (residual < tolerance) {
          // last time through, we need to apply boundary condition to u[0], since we just relaxed (above), but haven't propagated changes to ghost cells.
          // in the case we are not finished, by the time we get back to level 0 (via coarsening & then prolongation), ghost cells would be filled in.
          // but since we are bailing here, we need to explicitly make ghost cells up-to-date with u.
          apply_boundary_conditions(0);

          //printf("[INFO] Sol_MultigridPressure3DBase::do_vcycle - error after %d iterations: L2 = %f (%fx), Linf = %f (%fx)\n", i_cyc, result_l2, orig_l2 / result_l2, result_linf, orig_linf / result_linf);
          return !any_error();
        }
      }
      else
        restrict_residuals(level, level+1, 0, 0);
    }

    // these relaxation steps are essentially free, so do lots of them
    // (reference implementation uses nu1+nu2) - this is probably overkill, i need to revisit this
    // with a good heuristic.  Inhomogeneous conditions require more iterations.
    int coarse_iters = max3(nx(coarse_level)*ny(coarse_level), ny(coarse_level)*nz(coarse_level), nx(coarse_level)*nz(coarse_level))/2;      
    relax(coarse_level, coarse_iters, make_symmetric_operator ? RO_SYMMETRIC : RO_RED_BLACK);
    //relax(coarse_level, (nx(coarse_level)*ny(coarse_level)*nz(coarse_level))/2);

    // going up
    for (level = coarse_level-1; level >= 0; level--) {
      prolong(level+1, level);
      // don't need to relax finest grid since it will get relaxed at the beginning of the next v-cycle
      if (level > 0) 
        relax(level, nu2, make_symmetric_operator ? RO_BLACK_RED : RO_RED_BLACK);
    }
  }

  if (!(convergence & CONVERGENCE_CRITERIA_NONE))
    printf("[WARNING] Sol_MultigridPressure3DBase::do_vcycle - Failed to converge, error after: L2 = %f (%fx), Linf = %f (%fx)\n", result_l2, orig_l2 / result_l2, result_linf, orig_linf / result_linf);

  return false;
}


bool 
Sol_MultigridPressure3DBase::solve(double &residual, double tolerance, int max_iter)
{
  if (!check_float(tolerance)) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::solve - garbage tolerance value %f\n", tolerance);
    return false;
  }

  if (tolerance < 0) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::solve - negative tolerance value %f\n", tolerance);
    return false;
  }

  if (max_iter <= 0) {
    printf("[WARNING] Sol_MultigridPressure3DDevice::solve - non-positive max_iter %d\n", max_iter);
    return false;
  }

  double l2, linf;
  if (!do_fmg(tolerance, max_iter, l2, linf)) {

    // try again with initual solution vector set to 0.  This can fix the problem
    // of certain error modes growing when resuing the previous solution as a
    // starting point.
    printf("[WARNING] Sol_MultigridPressure3DBase::solve - do_fmg did not converge, retrying with zeroed initial search vector\n");
    clear_zero(0);
    if (!do_fmg(tolerance, max_iter, l2, linf)) {

      printf("[WARNING] Sol_MultigridPressure3DBase::solve - do_fmg failed\n");
      residual = (convergence == CONVERGENCE_L2) ? l2 : linf;
      return false;

    }
  }
  residual = (convergence == CONVERGENCE_L2) ? l2 : linf;
  //printf("residual = %g\n", residual);

  return true;
}

bool 
Sol_MultigridPressure3DBase::run_vcycles(int count)
{
  if (count <= 0) {
    printf("[WARNING] Sol_MultigridPressure3DDevice::run_vcycles - non-positive count %d\n", count);
    return false;
  }

  double l2, linf;
  do_vcycle(0, count, l2, linf);

  return !any_error();
}

bool 
Sol_MultigridPressure3DBase::run_fmg(int count)
{
  if (count < 0) {
    printf("[WARNING] Sol_MultigridPressure3DDevice::run_fmg - negative count %d\n", count);
    return false;
  }

  double l2, linf;
  do_fmg(0, count, l2, linf);

  return !any_error();
}


bool 
Sol_MultigridPressure3DBase::initialize_base_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val)
{
  // pre-validate
  if (!check_float(hx_val) || !check_float(hy_val) || !check_float(hz_val)) {
    printf("[ERROR] Sol_MultigridPressure3DBase::initialize_storage - garbage hx,hy,hz value %f %f %f\n", hx_val, hy_val, hz_val);
    return false;
  }

  // do allocation and initialization
  _num_levels = num_levels(nx_val, ny_val, nz_val);

  _h = new double[_num_levels];
  _dim = new int3[_num_levels];
  _h[0] = min3(hx_val, hy_val, hz_val);
  _fx = (_h[0] * _h[0]) / (hx_val * hx_val);
  _fy = (_h[0] * _h[0]) / (hy_val * hy_val);
  _fz = (_h[0] * _h[0]) / (hz_val * hz_val);
  _omega = optimal_omega(_fx, _fy, _fz);
  _dim[0].x = nx_val;
  _dim[0].y = ny_val;
  _dim[0].z = nz_val;

  int level;
  for (level=1; level < _num_levels; level++) {
    int this_nx = nx(level-1)/2;
    int this_ny = ny(level-1)/2;
    int this_nz = nz(level-1)/2;

    _h[level] = get_h(level-1)*2;
    _dim[level].x = this_nx;
    _dim[level].y = this_ny;
    _dim[level].z = this_nz;
  }

  return true;
}


} // namespace

