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

#include <cstdlib>
#include "tests/testframework.h"
#include "ocuutil/timer.h"
#include "ocuutil/timing_pool.h"
#include "ocustorage/grid3dops.h"
#include "ocuequation/sol_mgpressure3d.h"

using namespace ocu;




DECLARE_UNITTEST_DOUBLE_BEGIN(MultigridDoubleTest);

void init_solver(Sol_MultigridPressure3DDeviceD &solver, Grid3DDeviceD &rhs,
                 BoundaryConditionSet bc, 
                 int nx, int ny, int nz, float hx, float hy, float hz, bool sym)
{
  solver.nu1 = 2;
  solver.nu2 = 2;
  solver.bc = bc;
  solver.make_symmetric_operator = sym;

  UNITTEST_ASSERT_TRUE(solver.initialize_storage(nx,ny,nz,hx,hy,hz,&rhs));
}

double init_rhs(Grid3DDeviceD &rhs, int nx, int ny, int nz, float hx, float hy, float hz, int axis, bool zero_rhs)
{
  UNITTEST_ASSERT_TRUE(rhs.init(nx,ny,nz,1,1,1));


  if (zero_rhs) {
    UNITTEST_ASSERT_TRUE(rhs.clear_zero());
    return 0;
  }
  else {
    Grid3DHostD rhs_host;
    UNITTEST_ASSERT_TRUE(rhs_host.init(nx,ny,nz,1,1,1));
    double pi = acos(-1.0);
    double integral = 0;

    int i,j,k;
    for (k=0; k < nz; k++) {
      double z = ((k-.5) * 2 * pi * hz);
      if (axis == 0 || axis == 1) z = 0;
      for (j=0; j < ny; j++) {
        double y = ((j-.5) * 2 * pi * hy);
        if (axis == 0 || axis == 2) y = pi/2;
        for (i=0; i < nx; i++) {
          double x = ((i-.5) * 2 * pi * hx);
          if (axis == 1 || axis == 2) x = 0;
          integral += (cos(x) * sin(y) * cos(z));
        }
      }
    }

    double adjustment = integral / (nx * ny * nz);

    for (k=0; k < nz; k++) {
      double z = ((k-.5) * 2 * pi * hz);
      if (axis == 0 || axis == 1) z = 0;
      for (j=0; j < ny; j++) {
        double y = ((j-.5) * 2 * pi * hy);
        if (axis == 0 || axis == 2) y = pi/2;
        for (i=0; i < nx; i++) {
          double x = ((i-.5) * 2 * pi * hx);
          if (axis == 1 || axis == 2) x = 0;
          rhs_host.at(i,j,k) = (double) (cos(x) * sin(y) * cos(z) - adjustment);
        }
      }
    }

    double final_integral;
    rhs_host.reduce_sum(final_integral);

    UNITTEST_ASSERT_TRUE(rhs.copy_all_data(rhs_host));  
    return final_integral;
  }
}

// make test repeatable using psuedo-random hash
// TODO: move this into util library
float hash_seed(unsigned int seed) {
  seed = (seed+0x7ed55d16) + (seed<<12);
  seed = (seed^0xc761c23c) ^ (seed>>19);
  seed = (seed+0x165667b1) + (seed<<5);
  seed = (seed+0xd3a2646c) ^ (seed<<9);
  seed = (seed+0xfd7046c5) + (seed<<3);
  seed = (seed^0xb55a4f09) ^ (seed>>16);

  return (seed + 1.0f) / 4294967296.0f;
}

void init_search_vector(Sol_MultigridPressure3DDeviceD &solver, int nx, int ny, int nz, bool init_random)
{
  int i,j,k;

  if (init_random) {
    Grid3DHostD pinit;
    pinit.init_congruent(solver.pressure());
    for (i=0; i < nx; i++)
      for (j=0; j < ny; j++)
        for (k=0; k < nz; k++)
          pinit.at(i,j,k) = .5 - hash_seed(i*nz*ny+j*nz+nz);

    UNITTEST_ASSERT_TRUE(solver.pressure().copy_all_data(pinit));
  }
  else {
    UNITTEST_ASSERT_TRUE(solver.pressure().clear_zero());
  }
}

void plot_error(Sol_MultigridPressure3DDeviceD &solver)
{
  /*
  std::vector<ImageFile> slices;
  Grid3DHostD host_error;
  host_error.init_congruent(solver.read_r(0));
  host_error.copy_all_data(solver.read_r(0));
  plot_scalar_value(host_error, slices);
  for (int s=0; s < slices.size(); s++) {
    char buf[1024];
    sprintf(buf, "error.%04d.ppm", s);
    slices[s].write_ppm(buf);
  }
  */
}


void set_bc(
  BoundaryConditionSet &bc, 
  BoundaryConditionType type, float value) 
{
  BoundaryCondition example;
  example.type = type;
  example.value = value;
  bc = BoundaryConditionSet(example);
}

void run_isotropic_test(int nx, int ny, int nz, float hx, float hy, float hz, int axis, float value, double tol, bool sym)
{
  BoundaryConditionSet bc;
  set_bc(bc, BC_PERIODIC, 0);

  if (axis == 0) {
    bc.xpos.type = BC_DIRICHELET;
    bc.xneg.type = BC_DIRICHELET;
    bc.xpos.value = bc.xneg.value = value;
  }
  else if (axis == 1) {
    bc.ypos.type = BC_DIRICHELET;
    bc.yneg.type = BC_DIRICHELET;
    bc.ypos.value = bc.yneg.value = value;
  }
  else {
    bc.zpos.type = BC_DIRICHELET;
    bc.zneg.type = BC_DIRICHELET;
    bc.zpos.value = bc.zneg.value = value;
  }

  Sol_MultigridPressure3DDeviceD solver;
  Grid3DDeviceD rhs;

  init_rhs(rhs, nx, ny, nz, hx, hy, hz, axis, false); // don't init to zero
  init_solver(solver, rhs, bc, nx, ny, nz, hx, hy, hz, sym);
  init_search_vector(solver, nx, ny, nz, true); // init to random search vector
    
  double residual;

  CPUTimer timer;
  timer.start();

  UNITTEST_ASSERT_TRUE(solver.solve(residual,tol,16));
  timer.stop();
  global_timer_add_timing("solver", timer.elapsed_ms());

  UNITTEST_ASSERT_EQUAL_DOUBLE(residual, 0, tol);
}




void run_all_dirichelet_test(int nx, int ny, int nz, float hx, float hy, float hz, float value, double tol, bool sym)
{
  BoundaryConditionSet bc;
  set_bc(bc, BC_DIRICHELET, value);
  
  Sol_MultigridPressure3DDeviceD solver;
  Grid3DDeviceD rhs;

  init_rhs(rhs, nx, ny, nz, hx, hy, hz, -1, true); // init to zero, no 
  init_solver(solver, rhs, bc, nx, ny, nz, hx, hy, hz, sym);
  init_search_vector(solver, nx, ny, nz, true); // init to random search vector
    
  double residual;
  CPUTimer timer;
  timer.start();

  UNITTEST_ASSERT_TRUE(solver.solve(residual,tol, 15));
  timer.stop();
  global_timer_add_timing("solver", timer.elapsed_ms());

  UNITTEST_ASSERT_EQUAL_FLOAT(residual, 0, tol);

  double max_val, min_val;

  UNITTEST_ASSERT_TRUE(solver.pressure().reduce_max(max_val));
  UNITTEST_ASSERT_TRUE(solver.pressure().reduce_min(min_val));
  UNITTEST_ASSERT_EQUAL_DOUBLE(max_val, value, tol);
  UNITTEST_ASSERT_EQUAL_DOUBLE(min_val, value, tol);
}

void run_neumann_test(int nx, int ny, int nz, float hx, float hy, float hz, double tol, bool sym)
{
  BoundaryConditionSet bc;
  set_bc(bc, BC_NEUMANN, 0);
  
  Sol_MultigridPressure3DDeviceD solver;
  Grid3DDeviceD rhs;

  init_rhs(rhs, nx, ny, nz, hx, hy, hz, -1, false); // init to sin waves, no axis
  init_solver(solver, rhs, bc, nx, ny, nz, hx, hy, hz, sym);
  init_search_vector(solver, nx, ny, nz, false); // init to zero
    
  double residual;
  CPUTimer timer;
  timer.start();

  UNITTEST_ASSERT_TRUE(solver.solve(residual,tol, 15));
  timer.stop();
  global_timer_add_timing("solver", timer.elapsed_ms());

  UNITTEST_ASSERT_EQUAL_DOUBLE(residual, 0, tol);
}

void run_all_periodic_test(int nx, int ny, int nz, float hx, float hy, float hz, double tol, bool sym)
{
  BoundaryConditionSet bc;
  set_bc(bc, BC_PERIODIC, 0);
  
  Sol_MultigridPressure3DDeviceD solver;
  Grid3DDeviceD rhs;

  init_rhs(rhs, nx, ny, nz, hx, hy, hz, -1, false); // init to sin waves, no axis
  init_solver(solver, rhs, bc, nx, ny, nz, hx, hy, hz, sym);
  init_search_vector(solver, nx, ny, nz, false); // init to zero
    
  double residual;
  CPUTimer timer;
  timer.start();
  UNITTEST_ASSERT_TRUE(solver.solve(residual,tol, 20));
  timer.stop();
  global_timer_add_timing("solver", timer.elapsed_ms());

  UNITTEST_ASSERT_EQUAL_DOUBLE(residual, 0, tol);
}

void run_all_bcs(int nx, int ny, int nz, float hx, float hy, float hz, float tol, bool sym)
{
  run_isotropic_test(nx, ny, nz, hx, hy, hz, 0, 0, tol, sym);
  run_isotropic_test(nx, ny, nz, hx, hy, hz, 1, 0, tol, sym);
  run_isotropic_test(nx, ny, nz, hx, hy, hz, 2, 0, tol, sym);
  run_isotropic_test(nx, ny, nz, hx, hy, hz, 0, 1, tol, sym);
  run_isotropic_test(nx, ny, nz, hx, hy, hz, 1, -1, tol, sym);
  run_isotropic_test(nx, ny, nz, hx, hy, hz, 2, -1, tol, sym);
  run_isotropic_test(nx, ny, nz, hx, hy, hz, 2, .321, tol, sym); 
  run_all_dirichelet_test(nx, ny, nz, hx, hy, hz, 0, tol, sym);
  run_all_dirichelet_test(nx, ny, nz, hx, hy, hz, .5, tol, sym);
  run_all_dirichelet_test(nx, ny, nz, hx, hy, hz, 1, tol, sym);
  run_all_periodic_test(nx, ny, nz, hx, hy, hz, tol, sym);
  run_neumann_test(nx, ny, nz, hx, hy, hz, tol, sym); 
}

void run()
{
  run_all_bcs(256,256,256, 4.0 / 128, 4.0 / 128, 4.0 / 128, 1e-8, false);
  run_all_bcs(128, 128, 128, 4.0 / 128, 4.0 / 128, 4.0 / 128, 1e-8, false);
  run_all_bcs(128, 128, 128, 4.0 / 128, 5.0 / 128, 6.0 / 128, 1e-8, false);
  run_all_bcs(128, 128, 128, 5.0 / 128, 4.0 / 128, 6.0 / 128, 1e-8, false);
  run_all_bcs(128, 128, 128, 5.0 / 128, 6.0 / 128, 4.0 / 128, 1e-8, false);
  run_all_bcs(64, 128, 128, 4.0 / 64, 4.0 / 128, 4.0 / 128, 1e-8, false);
  run_all_bcs(128, 64, 128, 4.0 / 128, 4.0 / 64, 4.0 / 128, 1e-8, false);
  run_all_bcs(128, 128, 64, 4.0 / 128, 4.0 / 128, 4.0 / 64, 1e-8, false);
  run_all_bcs(96, 128, 80, 4.0 / 128, 4.0 / 128, 4.0 / 64, 1e-8, false);

  run_all_bcs(64, 128, 128, 4.0 / 64, 4.0 / 128, 4.0 / 128, 1e-8, true);
  run_all_bcs(128, 64, 128, 4.0 / 128, 4.0 / 64, 4.0 / 128, 1e-8, true);
  run_all_bcs(128, 128, 64, 4.0 / 128, 4.0 / 128, 4.0 / 64, 1e-8, true);
  run_all_bcs(96, 128, 80, 4.0 / 128, 4.0 / 128, 4.0 / 64, 1e-8, true);

  global_timer_print();
}


DECLARE_UNITTEST_DOUBLE_END(MultigridDoubleTest);
