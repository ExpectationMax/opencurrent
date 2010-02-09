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

#include "ocuutil/float_routines.h"
#include "ocuequation/sol_pcgpressure3d.h"
#include "ocuequation/sol_mgpressure3d.h"
#include "ocustorage/gridnetcdf.h"

using namespace ocu;

#ifndef OCU_DOUBLESUPPORT

int main(int argc, const char **argv)
{
  printf("[ERROR] pcg executable required double precision support (OCU_TARGET_SM >= SM_13)\n");
  return -1;
}



#else


void init_pcg_solver(Sol_PCGPressure3DDeviceD &solver, Grid3DDeviceD &rhs, Grid3DDeviceD &coeff,
                 BoundaryConditionSet bc, 
                 int nx, int ny, int nz, float hx, float hy, float hz)
{
  solver.bc = bc;
  solver.preconditioner = PRECOND_BFBT;
  solver.multigrid_use_fmg = true;
  solver.multigrid_cycles = 4;

  solver.initialize_storage(nx,ny,nz,hx,hy,hz,&rhs, &coeff);
}



void init_mg_solver(Sol_MultigridPressure3DDeviceD &solver, Grid3DDeviceD &rhs,
                 BoundaryConditionSet bc, 
                 int nx, int ny, int nz, double hx, double hy, double hz)
{
  solver.nu1 = 2;
  solver.nu2 = 2;
  solver.bc = bc;
  solver.make_symmetric_operator = false;

  solver.initialize_storage(nx,ny,nz,hx,hy,hz,&rhs);
}


void init_coeff(Grid3DDeviceD &coeff, int nx, int ny, int nz, double hx, double hy, double hz, double ratio)
{
  coeff.init(nx,ny,nz,1,1,1);
  Grid3DHostD coeff_host;
  coeff_host.init(nx,ny,nz,1,1,1);

#if 0
  int i,j,k;
  double pi = acos(-1.0);
  for (k=0; k < nz; k++) {
    for (j=0; j < ny; j++) {
      for (i=0; i < nx; i++) {
        double x = hx * i * 2*pi;
        double y = hy * j * 2*pi;
        double z = hz * k * 2*pi;
        double val = (1.0/ratio) + (1 + sin(x*10) * cos(y*4) * cos(z)) / 2;
        coeff_host.at(i,j,k) = val;
      }
    }
  }

  double vmax, vmin;
  coeff_host.reduce_max(vmax);
  coeff_host.reduce_min(vmin);

  printf("Coefficient ratio = %f\n", vmax / vmin);

#else

  double x_ratio = pow(ratio, 1.0/3.0);
  double y_ratio = pow(ratio, 1.0/3.0);
  double z_ratio = pow(ratio, 1.0/3.0);

  int i,j,k;
  for (k=0; k < nz; k++) {
    for (j=0; j < ny; j++) {
      for (i=0; i < nx; i++) {
        float i_var = (i < nx / 2) ? 1/x_ratio : 1.0;
        float j_var = (j < ny / 3) ? 1/y_ratio : 1.0;
        float k_var = (k < nz / 4) ? 1/z_ratio : 1.0;
        coeff_host.at(i,j,k) = k_var*j_var*k_var;
      }
    }
  }
#endif

  coeff.copy_all_data(coeff_host);
}

double init_rhs(Grid3DDeviceD &rhs, int nx, int ny, int nz, double hx, double hy, double hz, int axis, bool zero_rhs)
{
  rhs.init(nx,ny,nz,1,1,1);


  if (zero_rhs) {
    rhs.clear_zero();
    return 0;
  }
  else {
    Grid3DHostD rhs_host;
    rhs_host.init(nx,ny,nz,1,1,1);
    rhs_host.clear_zero();

    double pi = acos(-1.0);
    double integral = 0;

    int i,j,k;
    for (k=0; k < nz; k++) {
      double z = ((k) * 2 * pi * hz);
      if (axis == 0 || axis == 1) z = 0;
      for (j=0; j < ny; j++) {
        double y = ((j) * 2 * pi * hy);
        if (axis == 0 || axis == 2) y = pi/2;
        for (i=0; i < nx; i++) {
          double x = ((i) * 2 * pi * hx);
          if (axis == 1 || axis == 2) x = 0;
          //integral += (cos(x/2) * sin(y*5) * cos(z*7));
          if (i==8 && j == 8 && k == 8) 
            integral += 1;
        }
      }
    }

    double adjustment = integral / (nx * ny * nz);

    for (k=0; k < nz; k++) {
      double z = ((k) * 2 * pi * hz);
      if (axis == 0 || axis == 1) z = 0;
      for (j=0; j < ny; j++) {
        double y = ((j) * 2 * pi * hy);
        if (axis == 0 || axis == 2) y = pi/2;
        for (i=0; i < nx; i++) {
          double x = ((i) * 2 * pi * hx);
          if (axis == 1 || axis == 2) x = 0;
          //rhs_host.at(i,j,k) = (double) (cos(x/2) * sin(y*5) * cos(z*7) - adjustment);
          if (i==8 && j == 8 && k == 8) 
            rhs_host.at(i,j,k) = 1 - adjustment;
          else
            rhs_host.at(i,j,k) =  - adjustment;
        }
      }
    }

/*
    NetCDFGrid3DWriter  writer;
    writer.open("rhs.nc", nx, ny, nz);
    writer.define_variable("rhs", NC_DOUBLE, GS_CENTER_POINT);
    writer.add_data("rhs", rhs_host);
    writer.close();
*/
    double final_integral;
    rhs_host.reduce_sum(final_integral);

    rhs.copy_all_data(rhs_host);  
    return final_integral;
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

void compare_mg_pcg(int nx, int ny, int nz, double hx, double hy, double hz, double tol)
{
  BoundaryConditionSet bc;
  set_bc(bc, BC_PERIODIC, 0);
  
  Sol_PCGPressure3DDeviceD pcg_solver;
  Sol_MultigridPressure3DDeviceD mg_solver;
  Grid3DDeviceD rhs, coeff;

  init_rhs(rhs, nx, ny, nz, hx, hy, hz, -1, false); // init to sin waves, no axis
  init_coeff(coeff, nx, ny, nz, hx, hy, hz, 1000); 

  init_mg_solver (mg_solver,  rhs, bc, nx, ny, nz, hx, hy, hz);
  init_pcg_solver(pcg_solver, rhs, coeff, bc, nx, ny, nz, hx, hy, hz);
  pcg_solver.preconditioner = PRECOND_BFBT;

  mg_solver.pressure().clear_zero();
  pcg_solver.pressure().clear_zero();

  double residual;
  CPUTimer pcg_timer, mg_timer;
  pcg_timer.start();
  if (!pcg_solver.solve(residual,tol,1000))
    printf("FAILED\n");
  pcg_timer.stop();
  printf("PCG Residual = %g\n", residual);
  
  mg_timer.start();
  mg_solver.solve(residual, tol, 15);
  mg_timer.stop();
  printf("MG Residual = %g\n", residual);

  printf("MG %f sec, PCG: %f sec\n", mg_timer.elapsed_sec(), pcg_timer.elapsed_sec());
}


void run_pcg(int nx, int ny, int nz, double hx, double hy, double hz, double tol, double ratio)
{
  BoundaryConditionSet bc;
  set_bc(bc, BC_PERIODIC, 0);
  
  Sol_PCGPressure3DDeviceD pcg_solver;
  Grid3DDeviceD rhs, coeff;

  init_rhs(rhs, nx, ny, nz, hx, hy, hz, -1, false); // init to sin waves, no axis
  init_coeff(coeff, nx, ny, nz, hx, hy, hz, ratio); 

  init_pcg_solver(pcg_solver, rhs, coeff, bc, nx, ny, nz, hx, hy, hz);
  pcg_solver.pressure().clear_zero();

  double residual;
  CPUTimer pcg_timer;
  pcg_timer.start();
  if (!pcg_solver.solve(residual,tol,20))
    printf("FAILED\n");
  pcg_timer.stop();
  printf("PCG Residual = %g\n", residual);  
  printf("PCG: %f sec\n", pcg_timer.elapsed_sec());
}


void run_mg(int nx, int ny, int nz, double hx, double hy, double hz, double tol)
{
  BoundaryConditionSet bc;
  set_bc(bc, BC_NEUMANN, 0);
  
  Sol_MultigridPressure3DDeviceD pcg_solver;
  Grid3DDeviceD rhs;

  init_rhs(rhs, nx, ny, nz, hx, hy, hz, -1, false); // init to sin waves, no axis

  init_mg_solver(pcg_solver, rhs, bc, nx, ny, nz, hx, hy, hz);
  pcg_solver.pressure().clear_zero();

  double residual;
  CPUTimer pcg_timer;
  pcg_timer.start();
  if (!pcg_solver.solve(residual,tol))
    printf("FAILED\n");
  pcg_timer.stop();
  printf("PCG Residual = %g\n", residual);
  printf("PCG: %f sec\n", pcg_timer.elapsed_sec());
}




void compare_precond(int nx, int ny, int nz, double hx, double hy, double hz, double tol, double ratio)
{
  BoundaryConditionSet bc;
  set_bc(bc, BC_PERIODIC, 0);
  
  Sol_PCGPressure3DDeviceD pcgmg_solver;
  Sol_PCGPressure3DDeviceD pcgjac_solver;
  Grid3DDeviceD rhs, coeff;

  init_rhs(rhs, nx, ny, nz, hx, hy, hz, -1, false); // init to sin waves, no axis
  init_coeff(coeff, nx, ny, nz, hx, hy, hz, ratio); 

  init_pcg_solver(pcgmg_solver, rhs, coeff, bc, nx, ny, nz, hx, hy, hz);
  init_pcg_solver(pcgjac_solver, rhs, coeff, bc, nx, ny, nz, hx, hy, hz);
  pcgjac_solver.preconditioner = PRECOND_JACOBI;
  pcgmg_solver.preconditioner = PRECOND_BFBT;

  pcgmg_solver.pressure().clear_zero();
  pcgjac_solver.pressure().clear_zero();

  double residual;
  CPUTimer pcgjac_timer, pcgmg_timer;
  pcgjac_timer.start();
  if (!pcgjac_solver.solve(residual,tol,1000))
    printf("FAILED\n");
  pcgjac_timer.stop();
  printf("PCG Jacobi Residual = %g\n", residual);

  pcgmg_timer.start();
  if (!pcgmg_solver.solve(residual, tol, 1000))
    printf("FAILED\n");
  pcgmg_timer.stop();
  printf("PCG MG Residual = %g\n", residual);

  printf("PCG MG %f sec, PCG Jacobi: %f sec\n", pcgmg_timer.elapsed_sec(), pcgjac_timer.elapsed_sec());
}


int main(int argc, const char **argv)
{
  int dim = 64;

  run_mg(dim,dim,dim, 1.0/dim, 1.0/dim, 1.0/dim, 1e-8);
  /*run_pcg(dim,dim,dim, 1.0/dim, 1.0/dim, 1.0/dim, 1e-8, 100);
  run_pcg(dim,dim,dim, 1.0/dim, 1.0/dim, 1.0/dim, 1e-8, 1000);
  run_pcg(dim,dim,dim, 1.0/dim, 1.0/dim, 1.0/dim, 1e-8, 1e6);
  run_pcg(dim,dim,dim, 1.0/dim, 1.0/dim, 1.0/dim, 1e-8, 1e10);*/

  return 0;
}

#endif