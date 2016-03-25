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

#include "tests/testframework.h"
#include "ocuequation/eqn_incompressns3d.h"
#include "ocustorage/grid3d.h"
#include "ocuutil/imagefile.h"
#include "ocuutil/color.h"
#include "ocuutil/timing_pool.h"



using namespace ocu;


DECLARE_UNITTEST_DOUBLE_BEGIN(RayleighNoSlipTest);


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


double rand_val(double min_val, double max_val) {
  return min_val + 2 * max_val * ((double)rand())/RAND_MAX;
}

void init_params(Eqn_IncompressibleNS3DParams<double> &params, int res, double deltaT, DirectionType vertical, bool do_periodic) {

  int nx, ny, nz;
  float domain_x, domain_y, domain_z;
  double pi = acos(-1.0);

  if (vertical & DIR_XAXIS_FLAG) {
    nx = res;
    ny = res;
    nz = res/2;

    domain_x = 1.0;
    domain_y = pi/3.11;
    domain_z = domain_y * .5;
  }
  else if (vertical & DIR_YAXIS_FLAG) {
    nx = res;
    ny = res;
    nz = res/2;

    domain_x = pi/3.11;
    domain_y = 1.0;
    domain_z = domain_x * .5;
  }
  else {
    nx = res;
    ny = res/2;
    nz = res;

    domain_x = pi/3.11;
    domain_y = domain_x * .5;
    domain_z = 1.0;
  }


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
  params.bouyancy = 1;
  params.vertical_direction = vertical;

  params.advection_scheme = IT_FIRST_ORDER_UPWIND;
  params.time_step = TS_FORWARD_EULER;
  params.cfl_factor = .99;

  BoundaryCondition neumann;
  neumann.type = BC_NEUMANN;
  BoundaryCondition dirichelet;
  dirichelet.type = BC_DIRICHELET;
  BoundaryCondition periodic;
  periodic.type = BC_PERIODIC;
  BoundaryCondition closed;
  closed.type = BC_FORCED_INFLOW_VARIABLE_SLIP; // closed & no slip on all sides
  
  params.flow_bc = BoundaryConditionSet(closed);
  params.temp_bc = BoundaryConditionSet(neumann);

  if (vertical & DIR_XAXIS_FLAG) {
    params.temp_bc.xpos = dirichelet;
    params.temp_bc.xneg = dirichelet;
    params.flow_bc.xpos.aux_value = 1;
    params.flow_bc.xneg.aux_value = 1;
    
    if (do_periodic) {
      params.temp_bc.zpos = periodic;
      params.temp_bc.zneg = periodic;
      params.flow_bc.zpos = periodic;
      params.flow_bc.zneg = periodic;
    }

    if (vertical == DIR_XNEG)
      params.temp_bc.xpos.value = deltaT;
    else
      params.temp_bc.xneg.value = deltaT;
  }
  else if (vertical & DIR_YAXIS_FLAG) {
    params.temp_bc.ypos = dirichelet;
    params.temp_bc.yneg = dirichelet;
    params.flow_bc.ypos.aux_value = 1;
    params.flow_bc.yneg.aux_value = 1;

    if (do_periodic) {
      params.temp_bc.zpos = periodic;
      params.temp_bc.zneg = periodic;
      params.flow_bc.zpos = periodic;
      params.flow_bc.zneg = periodic;
    }

    if (vertical == DIR_YNEG)
      params.temp_bc.ypos.value = deltaT;
    else
      params.temp_bc.yneg.value = deltaT;
  }
  else {
    params.temp_bc.zpos = dirichelet;
    params.temp_bc.zneg = dirichelet;
    params.flow_bc.zpos.aux_value = 1;
    params.flow_bc.zneg.aux_value = 1;

    if (do_periodic) {
      params.temp_bc.ypos = periodic;
      params.temp_bc.yneg = periodic;
      params.flow_bc.ypos = periodic;
      params.flow_bc.yneg = periodic;
    }

    if (vertical == DIR_ZNEG)
      params.temp_bc.zpos.value = deltaT;
    else
      params.temp_bc.zneg.value = deltaT;
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

        double newtemp = y * deltaT;
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

bool run_resolution_ra(int res, double deltaT, double &ratio) {
  printf("deltaT = %f\n", deltaT);

  DirectionType vertical = DIR_YPOS;
  Eqn_IncompressibleNS3DParams<double> params;
  Eqn_IncompressibleNS3D<double> eqn;

  init_params(params, res, deltaT, vertical, false);
  UNITTEST_ASSERT_TRUE(eqn.set_parameters(params));

  std::list<double> data;

  int count = 1;
  int under_count = 0;
  bool done = false;
  double last_maxu=0, last_maxv = 0, last_maxw=0;
  bool ok = false;

  float start_time = elapsed_ms();

  while (!done) {
    count++;       

    UNITTEST_ASSERT_TRUE(eqn.advance(.01));
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
    
    if (fabs((u_ratio - v_ratio)/u_ratio) < .005) {
      under_count++;
      printf("under_count = %d\n", under_count);
      if (count > 50 && under_count > 5) {
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
    //  done = true;
    }

    if (count > 5000) {
      printf("[Not converging]\n");
      done = true;
    }
  }

  printf("\n............ DONE ...............\n\n");
  printf("ms/step = %f\n", (elapsed_ms() - start_time) / eqn.num_steps);

  return ok;
}

double run_resolution(int res, double start, double end, double step=1.0) {
  // estimate the Ra_cr value
  printf("run resolution %d\n", res);

  double lb_ra=0, lb_decay;
  double ub_ra=0, ub_decay;

  for (double rayleigh_number = start; rayleigh_number <= end; rayleigh_number += step) {

    double decay_rate;

    UNITTEST_ASSERT_TRUE(run_resolution_ra(res, rayleigh_number, decay_rate));
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

  UNITTEST_ASSERT_TRUE(lb_ra != 0);
  UNITTEST_ASSERT_TRUE(ub_ra != 0);

  // interpolate to find zero-crossing.  We know that lb < 0 & ub > 0, hence the signs
  return lb_ra - lb_decay / (ub_decay - lb_decay);
}

#define MAX_RES 32

void run() {

  double ra_16 = 16 <= MAX_RES ? run_resolution(16, 1674,1700,1) : 1674.296911;
  double ra_32 = 32 <= MAX_RES ? run_resolution(32, 1699,1700,.5) : 1699.253950;
  double ra_64 = 64 <= MAX_RES ? run_resolution(64, 1705,1706,1) : 1705.590663;

  double ra_exact = 1707.76; // from the literature

  double er_16 = ra_16 - ra_exact;
  double er_32 = ra_32 - ra_exact;
  double er_64 = ra_64 - ra_exact;

  double order_32 = log(er_16/er_32) / log(2.0);
  double order_64 = log(er_32/er_64) / log(2.0);


  printf("16  - %f (error %f)\n", ra_16, er_16);
  printf("32  - %f (error %f) - order(%f)\n", ra_32, er_32, order_32);
  printf("64  - %f (error %f) - order(%f)\n", ra_64, er_64, order_64);

  if (32 <= MAX_RES)
    UNITTEST_ASSERT_TRUE(order_32 > 1.9);
  if (64 <= MAX_RES)
    UNITTEST_ASSERT_TRUE(order_64 > 1.9);

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
  err_mult.push_back(.25);
  err_val.push_back(ra_64);

  double estimated_ra_exact = linear_regression(err_mult, err_val);

  // ideally error here will be very very small
  printf("\nestimated_ra_exact = %f, error = %.10f\n", estimated_ra_exact, fabs(estimated_ra_exact - ra_exact));
  
  global_timer_print();

  // should be pretty close to zero - probably even closer than this...
  UNITTEST_ASSERT_EQUAL_DOUBLE(estimated_ra_exact - ra_exact, 0, .5);
  fflush(stdout);
}


DECLARE_UNITTEST_DOUBLE_END(RayleighNoSlipTest);



