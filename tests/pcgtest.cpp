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
#include "ocuequation/sol_pcgpressure3d.h"

using namespace ocu;


DECLARE_UNITTEST_DOUBLE_BEGIN(PCGDoubleTest);

void init_solver(Sol_PCGPressure3DDeviceD &solver, Grid3DDeviceD &rhs, Grid3DDeviceD &coeff,
                 BoundaryConditionSet bc, 
                 int nx, int ny, int nz, float hx, float hy, float hz)
{
  solver.convergence = CONVERGENCE_L2;
  solver.bc = bc;
  solver.preconditioner = PRECOND_JACOBI;

  UNITTEST_ASSERT_TRUE(solver.initialize_storage(nx,ny,nz,hx,hy,hz,&rhs, &coeff));
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

void init_coeff(Grid3DDeviceD &coeff, int nx, int ny, int nz, float hx, float hy, float hz)
{
  coeff.init(nx,ny,nz,1,1,1);

  Grid3DHostD coeff_host;
  coeff_host.init(nx,ny,nz,1,1,1);
  int i,j,k;
  for (k=0; k < nz; k++) {
    for (j=0; j < ny; j++) {
      for (i=0; i < nx; i++) {
        float j_var = (j < ny / 3) ? 0.01 : 1.0;
        float i_var = (i < nx / 2) ? 0.01 : 1.0;
        float k_var = (k < nz / 4) ? 0.01 : 1.0;
        coeff_host.at(i,j,k) = k_var * j_var * k_var;
      }
    }
  }
  coeff.copy_all_data(coeff_host);
}


void init_search_vector(Sol_PCGPressure3DDeviceD &solver, int nx, int ny, int nz, bool init_random)
{
  int i,j,k;

  if (init_random) {
    Grid3DHostD pinit;
    pinit.init_congruent(solver.pressure());
    for (i=0; i < nx; i++)
      for (j=0; j < ny; j++)
        for (k=0; k < nz; k++)
          pinit.at(i,j,k) = .5 - (((double)rand()) / RAND_MAX);
    UNITTEST_ASSERT_TRUE(solver.pressure().copy_all_data(pinit));
  }
  else {
    UNITTEST_ASSERT_TRUE(solver.pressure().clear_zero());
  }
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

void run_neumann_test(int nx, int ny, int nz, float hx, float hy, float hz, double tol)
{
  BoundaryConditionSet bc;
  set_bc(bc, BC_NEUMANN, 0);
  
  Sol_PCGPressure3DDeviceD solver;
  Grid3DDeviceD rhs, coeff;

  init_rhs(rhs, nx, ny, nz, hx, hy, hz, -1, false); // init to sin waves, no axis
  init_coeff(coeff, nx, ny, nz, hx, hy, hz); 
  init_solver(solver, rhs, coeff, bc, nx, ny, nz, hx, hy, hz);
  init_search_vector(solver, nx, ny, nz, false); // init to zero
    
  double residual;
  UNITTEST_ASSERT_TRUE(solver.solve(residual,tol, 1000));

  UNITTEST_ASSERT_EQUAL_DOUBLE(residual, 0, tol);
}

void run_all_periodic_test(int nx, int ny, int nz, float hx, float hy, float hz, double tol)
{
  BoundaryConditionSet bc;
  set_bc(bc, BC_PERIODIC, 0);
  
  Sol_PCGPressure3DDeviceD solver;
  Grid3DDeviceD rhs, coeff;

  init_rhs(rhs, nx, ny, nz, hx, hy, hz, -1, false); // init to sin waves, no axis
  init_coeff(coeff, nx, ny, nz, hx, hy, hz); 
  init_solver(solver, rhs, coeff, bc, nx, ny, nz, hx, hy, hz);
  init_search_vector(solver, nx, ny, nz, false); // init to zero
    
  double residual;
  CPUTimer timer;
  timer.start();
  UNITTEST_ASSERT_TRUE(solver.solve(residual,tol,1000));
  timer.stop();
  printf("%f sec\n", timer.elapsed_sec());
  UNITTEST_ASSERT_EQUAL_DOUBLE(residual, 0, tol);
}


void run_all_bcs(int nx, int ny, int nz, float hx, float hy, float hz, float tol)
{
  run_all_periodic_test(nx, ny, nz, hx, hy, hz, tol);
  run_neumann_test(nx, ny, nz, hx, hy, hz, tol);
}

void run()
{
  run_all_bcs(256,256,256, 4.0 / 128, 4.0 / 128, 4.0 / 128, 1e-6);
  run_all_bcs(128, 128, 128, 4.0 / 128, 4.0 / 128, 4.0 / 128, 1e-6);
  run_all_bcs(128, 128, 128, 4.0 / 128, 5.0 / 128, 6.0 / 128, 1e-6);
  run_all_bcs(128, 128, 128, 5.0 / 128, 4.0 / 128, 6.0 / 128, 1e-6);
  run_all_bcs(128, 128, 128, 5.0 / 128, 6.0 / 128, 4.0 / 128, 1e-6);
  run_all_bcs(64, 128, 128, 4.0 / 64, 4.0 / 128, 4.0 / 128, 1e-6);
  run_all_bcs(128, 64, 128, 4.0 / 128, 4.0 / 64, 4.0 / 128, 1e-6);
  run_all_bcs(128, 128, 64, 4.0 / 128, 4.0 / 128, 4.0 / 64, 1e-6);
  run_all_bcs(96, 128, 80, 4.0 / 128, 4.0 / 128, 4.0 / 64, 1e-6);
}


DECLARE_UNITTEST_DOUBLE_END(PCGDoubleTest);
