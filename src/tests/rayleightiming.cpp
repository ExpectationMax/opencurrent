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

#include <cmath>

#include "tests/testframework.h"
#include "ocuequation/eqn_incompressns3d.h"
#include "ocustorage/grid3d.h"
#include "ocuutil/imagefile.h"
#include "ocuutil/color.h"
#include "ocuutil/timing_pool.h"



using namespace ocu;


DECLARE_UNITTEST_DOUBLE_BEGIN(RayleighTimingTest);


double rand_val(double min_val, double max_val) {
  return min_val + 2 * max_val * ((double)rand())/RAND_MAX;
}

void write_slice(const char *filename, const Grid3DDevice<double> &grid)
{
  Grid3DHost<double> h_grid;
  h_grid.init_congruent(grid);
  h_grid.copy_all_data(grid);


  int nx = grid.nx();
  int ny = grid.ny();
  int nz = grid.nz();

  ImageFile img;
  img.allocate(nx, ny);

  for (int i=0; i < nx; i++)
    for (int j=0; j < ny; j++) {
      double temperature = h_grid.at(i,j,nz/2);
      if (temperature < -2) temperature = -2;
      if (temperature > 2)  temperature = 2;
      //float3 color = make_float3(temperature, temperature, temperature);
      float3 color = hsv_to_rgb(make_float3((temperature)*360, 1, 1));
      //float3 color = pseudo_temperature((temperature+1)*.5);
      img.set_rgb(i,j,(unsigned char)(255*color.x),(unsigned char)(255*color.y),(unsigned char)(255*color.z));
    }

  img.write_ppm(filename);
}

void init_params(Eqn_IncompressibleNS3DParamsD &params, int res, double Ra, double Pr) {

  int nx = res;
  int ny = res/2;
  int nz = res;
  double domain_x = 2.0;
  double domain_y = 1.0;
  double domain_z = 2.0;

  double hx = domain_x/nx;
  double hy = domain_y/ny;
  double hz = domain_z/nz;

  params.init_grids(nx, ny, nz, false);
  params.hx = hx;
  params.hy = hy;
  params.hz = hz;
  params.max_divergence = 1e-6;
  
  // if everything is set to one, Ra = deltaT
  params.viscosity = Pr;
  params.thermal_diffusion = 1;
  params.gravity = -1;
  params.bouyancy = Ra * Pr;
  params.vertical_direction = DIR_YPOS;

  params.advection_scheme = IT_SECOND_ORDER_CENTERED;
  params.time_step = TS_ADAMS_BASHFORD2;
  params.cfl_factor = .7;

  BoundaryCondition dirichelet;
  dirichelet.type = BC_DIRICHELET;

  BoundaryCondition closed;
  closed.aux_value = 1; // no slip on bottom & top
  closed.type = BC_FORCED_INFLOW_VARIABLE_SLIP; // closed & no slip on all sides

  BoundaryCondition periodic;
  periodic.type = BC_PERIODIC;
  
  params.flow_bc = BoundaryConditionSet(periodic);
  params.temp_bc = BoundaryConditionSet(periodic);

  params.flow_bc.ypos = closed;
  params.flow_bc.yneg = closed;

  params.temp_bc.ypos = dirichelet;
  params.temp_bc.yneg = dirichelet;
  params.temp_bc.yneg.value = 1;

  int i,j,k;
  for (int i=0; i < nx; i++) {
    for (int j=0; j < ny; j++) {
      for (int k=0; k < nz; k++) {
        double y = 1 - (((j+.5) * hy) / domain_y);
        params.init_temp.at(i,j,k) = y + rand_val(-1e-2, 1e-2);

        params.init_u.at(i,j,k) = rand_val(-1e-2, 1e-2);
        params.init_v.at(i,j,k) = rand_val(-1e-2, 1e-2);
        params.init_w.at(i,j,k) = rand_val(-1e-2, 1e-2);
      }
    }
  }
}

void run_resolution(int res, double dt, double t1, double Ra, double Pr, bool do_diagnostic=true) {
  Eqn_IncompressibleNS3DParamsD params;
  Eqn_IncompressibleNS3DD eqn;

  init_params(params, res, Ra, Pr);
  UNITTEST_ASSERT_TRUE(eqn.set_parameters(params));

  int next_frame = 1;
  
  CPUTimer clock;
  CPUTimer step_clock;
  int step_count;
  int start_count=0;
  start_count = eqn.num_steps;
  clock.start();
  global_timer_clear_all();
  step_count = eqn.num_steps;
  step_clock.start();

  set_forge_ahead(true);

  for (double t = 0; t <= t1; t += dt) {
    UNITTEST_ASSERT_TRUE(eqn.advance_one_step(dt));

    if (do_diagnostic) {

      double max_u, max_v, max_w;
      eqn.get_u().reduce_maxabs(max_u);
      eqn.get_v().reduce_maxabs(max_v);
      eqn.get_w().reduce_maxabs(max_w); // not used in any calculations, but useful for troubleshooting

      printf("> Max u = %.12f, Max v = %.12f, Max w = %.12f\n", max_u, max_v, max_w);
      fflush(stdout);

      if (t > next_frame * t1/100) {
        char buff[1024];
        sprintf(buff, "output.%04d.ppm", next_frame);
        printf("%s\n", buff);
        write_slice(buff, eqn.get_temperature());
        next_frame++;
      }
    }
    else {

      if (t > next_frame * t1/100) {
        step_clock.stop();
        printf("ms/step = %f\n", step_clock.elapsed_ms() / (eqn.num_steps - step_count));
        char buff[1024];
        sprintf(buff, "output.%04d.ppm", next_frame);
        global_counter_print();
        global_counter_clear_all();
        printf("%s\n", buff);
        write_slice(buff, eqn.get_temperature());
        next_frame++;

        step_count = eqn.num_steps;
        step_clock.start();
      }

      printf("%.4f%% done\r", t/t1 * 100);
    }
  }
  clock.stop();
  printf("Elapsed sec: %.8f\n", clock.elapsed_sec());
  printf("ms/step = %f\n", clock.elapsed_ms() / (eqn.num_steps - start_count));

  printf("\n............ DONE ...............\n\n");
}


void run() {
//  run_resolution(256, 1.25e-6, .01, 1e7, 0.71, false);
//  run_resolution(128, 2 * 1.25e-6, .01, 1e7, 0.71, false);
//  run_resolution(64, 4 * 1.25e-6, .01, 1e7, 0.71, false);
//  run_resolution(384, 7.5e-7, .01, 1e7, 0.71, false);
//  run_resolution(256, 1.25e-6/4, .001, 1e8, 0.71, false);
  run_resolution(384, 7.5e-7/2, .005, 1e8, 0.71, false);
//run_resolution(24, 7.5e-7/2, .005, 1e8, 0.71, false);
  global_timer_print();
  global_counter_print();
}


DECLARE_UNITTEST_DOUBLE_END(RayleighTimingTest);

