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
#include <algorithm>
#include <list>
#include <vector>

#include "ocuequation/eqn_incompressns3d.h"
#include "ocustorage/grid3d.h"
#include "ocuutil/imagefile.h"
#include "ocuutil/color.h"
#include "ocuutil/timing_pool.h"

#ifdef OCU_DOUBLESUPPORT

#include "eqn_cubicrayleigh3d.h"

using namespace ocu;

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
      float3 color = hsv_to_rgb(make_float3((temperature + 2)*90, 1, 1));
      //float3 color = pseudo_temperature((temperature+1)*.5);
      img.set_rgb(i,j,(unsigned char)(255*color.x),(unsigned char)(255*color.y),(unsigned char)(255*color.z));
    }

  img.write_ppm(filename);
}

double linear_regression(const std::vector<double> &err_mult, const std::vector<double> &err_val) {

  // apply linear regression to estimate what value will be when error_mult is 0
  // aka y-intercept

  double x_m=0;
  double y_m=0;
  double num = 0;
  double denom = 0;

  int n = err_mult.size();
  int i;
  for (i=0; i < n; i++) {
    x_m += err_mult[i];
    y_m += err_val[i];
  }
  x_m /= n;
  y_m /= n;

  for (i=0; i < n; i++) {
    num += ((err_mult[i] - x_m) * (err_val[i] - y_m));
    denom += ((err_mult[i] - x_m) * (err_mult[i] - x_m));
  }
  
  double slope = num / denom;
  return y_m - (slope * x_m);
}



double stddev(const std::list<double> &data) {
  double mean = 0;
  int n = data.size();
  std::list<double>::const_iterator i;
  for(i= data.begin(); i != data.end(); i++) {
    mean += *i;
  }
  mean /= n;

  double var = 0;
  for(i= data.begin(); i != data.end(); i++) {
    var += (*i - mean) * (*i - mean);
  }
  var /= n;

  return sqrt(var);
}

void Rayleigh_init_params(Eqn_CubicRayleigh3DParamsD &params, int res, double Ra, DirectionType vertical) {

  int nx = res;
  int ny = res;
  int nz = res;
  float domain_x = 1.0;
  float domain_y = 1.0;
  float domain_z = 1.0;

  float hx = domain_x/nx;
  float hy = domain_y/ny;
  float hz = domain_z/nz;

  params.init_grids(nx, ny, nz);
  params.hx = hx;
  params.hy = hy;
  params.hz = hz;
  params.max_divergence = 1e-12;
  
  // if everything is set to one, Ra = deltaT
  params.viscosity = 1;
  params.thermal_diffusion = 1;
  params.gravity = -1;
  params.bouyancy = Ra;
  params.vertical_direction = vertical;

  //params.advection_scheme = IT_SECOND_ORDER_CENTERED;
  //params.time_step = TS_ADAMS_BASHFORD2;
  //params.cfl_factor = .7;
  params.advection_scheme = IT_FIRST_ORDER_UPWIND;
  params.time_step = TS_FORWARD_EULER;
  params.cfl_factor = .99;

  BoundaryCondition dirichelet;
  dirichelet.type = BC_DIRICHELET;
  BoundaryCondition closed;

  closed.aux_value = 1; // no slip on all sides
  closed.type = BC_FORCED_INFLOW_VARIABLE_SLIP; // closed & no slip on all sides
  
  params.flow_bc = BoundaryConditionSet(closed);
  params.temp_bc = BoundaryConditionSet(dirichelet);

  if (vertical & DIR_XAXIS_FLAG) {
    params.temp_bc.xpos = dirichelet;
    params.temp_bc.xneg = dirichelet;

    if (vertical == DIR_XNEG)
      params.temp_bc.xpos.value = 1;
    else
      params.temp_bc.xneg.value = 1;
  }
  else if (vertical & DIR_YAXIS_FLAG) {
    params.temp_bc.ypos = dirichelet;
    params.temp_bc.yneg = dirichelet;

    if (vertical == DIR_YNEG)
      params.temp_bc.ypos.value = 1;
    else
      params.temp_bc.yneg.value = 1;
  }
  else {
    params.temp_bc.zpos = dirichelet;
    params.temp_bc.zneg = dirichelet;

    if (vertical == DIR_ZNEG)
      params.temp_bc.zpos.value = 1;
    else
      params.temp_bc.zneg.value = 1;
  }

  double mint = 10000;
  double maxt = -10000;
  int i,j,k;
  for (int i=0; i < nx; i++) {
    for (int j=0; j < ny; j++) {
      for (int k=0; k < nz; k++) {
        float y;
        switch(vertical) {
          case DIR_XPOS: y = 1 - (((i+.5) * hx) / domain_x); break;
          case DIR_XNEG: y = (((i+.5) * hx) / domain_x); break;
          case DIR_YPOS: y = 1 - (((j+.5) * hy) / domain_y); break;
          case DIR_YNEG: y = (((j+.5) * hy) / domain_y); break;
          case DIR_ZPOS: y = 1 - (((k+.5) * hz) / domain_z); break;
          case DIR_ZNEG: y = (((k+.5) * hz) / domain_z); break;
        }

        double newtemp = y; // from 0 to 1
        mint = std::min(mint, newtemp);
        maxt = std::max(maxt, newtemp);
        params.init_temp.at(i,j,k) = newtemp;
        params.init_u.at(i,j,k) = rand_val(-1e-4, 1e-4);
        params.init_v.at(i,j,k) = rand_val(-1e-4, 1e-4);
        params.init_w.at(i,j,k) = rand_val(-1e-4, 1e-4);
      }
    }
  }

  printf("init min/max t = %f %f\n", mint, maxt);
}

bool Rayleigh_run_resolution_ra(int res, double deltaT, double &ratio) {
  printf("deltaT = %f\n", deltaT);

  DirectionType vertical = DIR_YPOS;
  Eqn_CubicRayleigh3DParamsD params;
  Eqn_CubicRayleigh3DD  eqn;

  Rayleigh_init_params(params, res, deltaT, vertical);
  eqn.set_parameters(params);

  std::list<double> data;

  int count = 1;
  int under_count = 0;
  bool done = false;
  double last_maxu=0, last_maxv = 0, last_maxw=0;
  bool ok = false;
  std::list<double> u_data, v_data;

  while (!done) {
    count++;        

    eqn.advance(.01);
    double max_u, max_v, max_w;
    eqn.get_u().reduce_maxabs(max_u);
    eqn.get_v().reduce_maxabs(max_v);
    eqn.get_w().reduce_maxabs(max_w); // not used in any calculations, but useful for troubleshooting

    double mag_lateral;
    double u_ratio, v_ratio;

    if (vertical & DIR_XAXIS_FLAG) {
      u_ratio = log(max_v / last_maxv);
      v_ratio = log(max_u / last_maxu);
      mag_lateral = max_v;
    }
    else if (vertical & DIR_YAXIS_FLAG) {
      u_ratio = log(max_u / last_maxu);
      v_ratio = log(max_v / last_maxv);
      mag_lateral = max_u;
    }
    else {
      u_ratio = log(max_u / last_maxu);
      v_ratio = log(max_w / last_maxw);
      mag_lateral = max_u;
    }

    u_data.push_back(u_ratio);
    v_data.push_back(v_ratio);

    if (count > 40) {
      u_data.pop_front();
      v_data.pop_front();

      double max_mag = fabs(u_ratio) > fabs(v_ratio) ? u_ratio : v_ratio;
      double max_stddev = fabs(u_ratio) > fabs(v_ratio) ? stddev(u_data) : stddev(v_data);
      // if both have stabilized, take the avg (?)
      if (max_stddev < .01 * fabs(max_mag)) {
        done = true;
        ok = true;
        ratio = std::max(u_ratio, v_ratio); // take max so if one is positive, we get a positive number
      }
    }

    if (fabs((u_ratio - v_ratio)/u_ratio) < .01) {
      under_count++;
      printf("under_count = %d\n", under_count);
      if (count > 100 && under_count > 10) {
        done = true;
        ratio = .5 * (u_ratio + v_ratio);
        ok = true;
      }
    }

    
    printf("> Log ratio: %.10f, %.10f, Max u = %.12f, Max v = %.12f, Max w = %.12f\n", u_ratio, v_ratio, max_u, max_v, max_w);
    fflush(stdout);

    last_maxu = max_u;
    last_maxv = max_v;
    last_maxw = max_w;

    if (mag_lateral < 1e-12) {
      printf("[Shrunk to zero]\n");
      done = true;
    }
    if (mag_lateral > .1) {
      printf("[Blew up]\n");
      done = true;
    }

    if (count > 5000) {
      printf("[Not converging]\n");
      done = true;
    }
  }

  printf("\n............ DONE ...............\n\n");

  return ok;
}

double Rayleigh_run_resolution(int res, double start, double end, double step=1.0) {
  // estimate the Ra_cr value
  printf("run resolution %d\n", res);

  double lb_ra=0, lb_decay;
  double ub_ra=0, ub_decay;

  for (double rayleigh_number = start; rayleigh_number <= end; rayleigh_number += step) {

    double decay_rate;

    Rayleigh_run_resolution_ra(res, rayleigh_number, decay_rate);
    if (decay_rate < 0) {
      lb_ra = rayleigh_number;
      lb_decay = decay_rate;
    }
    else {
      ub_ra = rayleigh_number;
      ub_decay = decay_rate;
      break;
    }
  }

  // interpolate to find zero-crossing.  We know that lb < 0 & ub > 0, hence the signs
  return lb_ra - lb_decay / (ub_decay - lb_decay);
}


void Rayleigh_run() {
  double ra_16 = 6642.534538;
//  ra_16 = Rayleigh_run_resolution(16, 6641, 6644,1);
  ra_16 = Rayleigh_run_resolution(16, 6641, 6643,.1);
  double ra_32 = 6756.473870; 
//  ra_32 = Rayleigh_run_resolution(32, 6756, 6760,1);
  ra_32 = Rayleigh_run_resolution(32, 6756, 6757,.1);
  double ra_64 = 6787.156363;
  //ra_64 = Rayleigh_run_resolution(64, 6787, 6800,1);
  double ra_128 = 6787.156363;
  //ra_128 = Rayleigh_run_resolution(128, 6790, 6800,1);
  
  double ra_exact = 6799.0; 

  double er_16 = ra_16 - ra_exact;
  double er_32 = ra_32 - ra_exact;
  double er_64 = ra_64 - ra_exact;
  double er_128 = ra_128 - ra_exact;

  double order_32 = log(er_16/er_32) / log(2.0);
  double order_64 = log(er_32/er_64) / log(2.0);
  double order_128 = log(er_64/er_32) / log(2.0);


  printf("16  - %f (error %f)\n", ra_16, er_16);
  printf("32  - %f (error %f) - order(%f)\n", ra_32, er_32, order_32);
  //printf("64  - %f (error %f) - order(%f)\n", ra_64, er_64, order_64);
  //printf("128  - %f (error %f) - order(%f)\n", ra_128, er_128, order_128);


  // the extrapolated value for Ra_cr based on our calculations are determined by the system of equations:
  // Ra_cr + 1 e = ra_32
  // Ra_cr + 4 e = ra_16
  // easiest way to solve this equation for t Ra_cr is by linear regression.  Ra_cr is simply
  // the y-intercept of the least-squares fit line.

  std::vector<double> err_mult, err_val;
  err_mult.push_back(4);
  err_val.push_back(ra_16);
  err_mult.push_back(1);
  err_val.push_back(ra_32);
  //err_mult.push_back(.25);
  //err_val.push_back(ra_64);
  //err_mult.push_back(1.0/16.0);
  //err_val.push_back(ra_128);

  double estimated_ra_exact = linear_regression(err_mult, err_val);

  // ideally error here will be very very small
  printf("\nestimated_ra_exact = %f, error = %.10f\n", estimated_ra_exact, fabs(estimated_ra_exact - ra_exact));
  
  fflush(stdout);
}






void Nusselt_init_params(Eqn_CubicRayleigh3DParamsD &params, int res, double Ra, double Pr, DirectionType vertical) {

  int nx = res;
  int ny = res;
  int nz = res;
  float domain_x = 1.0;
  float domain_y = 1.0;
  float domain_z = 1.0;

  float hx = domain_x/nx;
  float hy = domain_y/ny;
  float hz = domain_z/nz;

  params.init_grids(nx, ny, nz);
  params.hx = hx;
  params.hy = hy;
  params.hz = hz;
  params.max_divergence = 1e-8;
  
  // if everything is set to one, Ra = deltaT
  params.viscosity = Pr;
  params.thermal_diffusion = 1;
  params.gravity = -1;
  params.bouyancy = Ra * Pr;
  params.vertical_direction = vertical;

  params.advection_scheme = IT_SECOND_ORDER_CENTERED;
  params.time_step = TS_ADAMS_BASHFORD2;

  BoundaryCondition dirichelet;
  dirichelet.type = BC_DIRICHELET;
  BoundaryCondition closed;

  closed.aux_value = 1; // no slip on all sides
  closed.type = BC_FORCED_INFLOW_VARIABLE_SLIP; // closed & no slip on all sides
  
  params.flow_bc = BoundaryConditionSet(closed);
  params.temp_bc = BoundaryConditionSet(dirichelet);

  if (vertical & DIR_XAXIS_FLAG) {
    params.temp_bc.xpos = dirichelet;
    params.temp_bc.xneg = dirichelet;

    if (vertical == DIR_XNEG)
      params.temp_bc.xpos.value = 1;
    else
      params.temp_bc.xneg.value = 1;
  }
  else if (vertical & DIR_YAXIS_FLAG) {
    params.temp_bc.ypos = dirichelet;
    params.temp_bc.yneg = dirichelet;

    if (vertical == DIR_YNEG)
      params.temp_bc.ypos.value = 1;
    else
      params.temp_bc.yneg.value = 1;
  }
  else {
    params.temp_bc.zpos = dirichelet;
    params.temp_bc.zneg = dirichelet;

    if (vertical == DIR_ZNEG)
      params.temp_bc.zpos.value = 1;
    else
      params.temp_bc.zneg.value = 1;
  }

  int i,j,k;
  for (int i=0; i < nx; i++) {
    for (int j=0; j < ny; j++) {
      for (int k=0; k < nz; k++) {
        float y;
        switch(vertical) {
          case DIR_XPOS: y = 1 - (((i+.5) * hx) / domain_x); break;
          case DIR_XNEG: y = (((i+.5) * hx) / domain_x); break;
          case DIR_YPOS: y = 1 - (((j+.5) * hy) / domain_y); break;
          case DIR_YNEG: y = (((j+.5) * hy) / domain_y); break;
          case DIR_ZPOS: y = 1 - (((k+.5) * hz) / domain_z); break;
          case DIR_ZNEG: y = (((k+.5) * hz) / domain_z); break;
        }

        params.init_temp.at(i,j,k) = y + rand_val(-1e-4, 1e-4);
        params.init_u.at(i,j,k) = 0;
        params.init_v.at(i,j,k) = 0;
        params.init_w.at(i,j,k) = 0;
      }
    }
  }
}

double print_nusselt(const Grid3DDevice<double> &grid, const Grid3DDevice<double> &v, float hx, float hy, float hz)
{
  Grid3DHost<double> h_grid;
  h_grid.init_congruent(grid);
  h_grid.copy_all_data(grid);

  Grid3DHost<double> h_v;
  h_v.init_congruent(v);
  h_v.copy_all_data(v);


  int nx = grid.nx();
  int ny = grid.ny();
  int nz = grid.nz();
  double nu_b = 0;

  for (int j=0; j <= ny; j+=ny) {
    double sum = 0;

    for (int i=0; i < nx; i++) {
      for (int k=0; k < nz; k++) {
        double T_z = (h_grid.at(i,j,k) - h_grid.at(i,j-1,k)) / hy;
        double wT = h_v.at(i,j,k) * .5 * (h_grid.at(i,j,k) + h_grid.at(i,j-1,k));
        sum += wT - T_z;
      }
    }
    if (j==0) nu_b = sum / (nx * nz);
    printf("Nusselt[%d]: %f\n", j, sum / (nx * nz));
  }

  return nu_b;
}

double Nusselt_run_resolution(int res, double dt, double Ra, double Pr, double t1=1, bool do_diagnostic=true) {
  printf("\nRa =%f\n\n", Ra);
  DirectionType vertical = DIR_YPOS;
  Eqn_CubicRayleigh3DParamsD params;
  Eqn_CubicRayleigh3DD  eqn;

  Nusselt_init_params(params, res, Ra, Pr, vertical);
  eqn.set_parameters(params);

  double last_maxu=0, last_maxv = 0, last_maxw=0;
  int next_frame = 1;
  
  CPUTimer clock;
  clock.start();

  for (double t = 0; t <= t1; t += dt) {
    eqn.advance_one_step(dt);

    if (do_diagnostic) {
      double max_u, max_v, max_w;
      eqn.get_u().reduce_maxabs(max_u);
      eqn.get_v().reduce_maxabs(max_v);
      eqn.get_w().reduce_maxabs(max_w); // not used in any calculations, but useful for troubleshooting

      double mag_lateral;
      double u_ratio, v_ratio;

      if (vertical & DIR_XAXIS_FLAG) {
        u_ratio = log(max_v / last_maxv);
        v_ratio = log(max_u / last_maxu);
        mag_lateral = max_v;
      }
      else if (vertical & DIR_YAXIS_FLAG) {
        u_ratio = log(max_u / last_maxu);
        v_ratio = log(max_v / last_maxv);
        mag_lateral = max_u;
      }
      else {
        u_ratio = log(max_u / last_maxu);
        v_ratio = log(max_w / last_maxw);
        mag_lateral = max_u;
      }
      
      printf("> Log ratio: %.10f, %.10f, Max u = %.12f, Max v = %.12f, Max w = %.12f\n", u_ratio, v_ratio, max_u, max_v, max_w);
      fflush(stdout);

      last_maxu = max_u;
      last_maxv = max_v;
      last_maxw = max_w;

      if (t > next_frame * t1/100) {
        char buff[1024];
        sprintf(buff, "output.%04d.ppm", next_frame);
        printf("%s\n", buff);
        write_slice(buff, eqn.get_temperature());
        next_frame++;
      }
    }
    else {

      if (t > next_frame * t1/200) {
        next_frame++;
        print_nusselt(eqn.get_temperature(), eqn.get_v(), eqn.hx(), eqn.hy(), eqn.hz());
      }
      printf("%.4f%% done\r", t/t1 * 100);
    }
  }
  clock.stop();
  printf("Elapsed sec: %.8f\n", clock.elapsed_sec());

  printf("\n............ DONE ...............\n\n");

  // calculate Nusselt #
  double nu_b = print_nusselt(eqn.get_temperature(), eqn.get_v(), eqn.hx(), eqn.hy(), eqn.hz());

  return nu_b;
}


void Nusselt_run() {
  //UNITTEST_ASSERT_EQUAL_DOUBLE(run_resolution(32, 8e-5, 44e3, 0.71, 1, false), 2.055, .003);
//  Nusselt_run_resolution(32, 8e-5, 87e3, 0.71, 1, false);
//  Nusselt_run_resolution(32, 8e-5, 25e3, 0.71, 2, false);
//  Nusselt_run_resolution(32, 8e-5, 91e3, 0.71, 2, false);
  //Nusselt_run_resolution(32, 8e-5, 25e3, 0.71, 4, false);
  Nusselt_run_resolution(32, 2e-5, 44e3, 0.71, 4, false);
  //Nusselt_run_resolution(32, 8e-5, 45e3, 0.71, 4, false);
  //Nusselt_run_resolution(32, 8e-5, 87e3, 0.71, 4, false);
  //Nusselt_run_resolution(32, 8e-5, 91e3, 0.71, 8, false);

/*
  std::vector<double> err_mult, err_val;
  err_mult.push_back(4);
  err_val.push_back(run_resolution(16, 32e-5, 44e3, 0.71, 3, false));
  err_mult.push_back(1);
  err_val.push_back(run_resolution(32, 8e-5, 44e3, 0.71, 3, false));
//  err_mult.push_back(.25);
//  err_val.push_back(run_resolution(64, 2e-5, 44e3, 0.71, 1, false));

  double estimated_nu_exact = linear_regression(err_mult, err_val);
  printf("Extrapolated: %f\n", estimated_nu_exact);
*/
}



int main(int argc, const char **argv)
{
  Nusselt_run();
  return 0;
}

#else // ! OCU_DOUBLESUPPORT


int main(int argc, const char **argv)
{
  printf("[ERROR] cubicrayleigh only runs when compiled with double precision support\n");
  exit(-1);

  return 0;
}

#endif // OCU_DOUBLESUPPORT