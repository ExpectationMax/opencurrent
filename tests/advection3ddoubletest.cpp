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

#include "tests/testframework.h"
#include "ocuutil/timer.h"
#include "ocuutil/imagefile.h"
#include "ocuutil/float_routines.h"
#include "ocustorage/grid3dboundary.h"
#include "ocuequation/eqn_scalaradvection3d.h"
#include "ocuequation/sol_project3d.h"



using namespace ocu;



enum Pattern {
  SQUARE,
  GRADIENT
};

void create_params(int dim, int axis, Pattern pat, Eqn_ScalarAdvection3DParamsD &params, InterpolationType interp_type)
{
  params.init_grids(dim, dim, dim);
  params.hx = 1;
  params.hy = 1;
  params.hz = 1;
  params.advection_scheme = interp_type;

  double vx = (axis == 0) ? 1.5 : (axis == 3) ? -1.5 : 0;
  double vy = (axis == 1) ? 1.5 : (axis == 4) ? -1.5 : 0;
  double vz = (axis == 2) ? 1.5 : (axis == 5) ? -1.5 : 0;
  params.u.clear(vx);
  params.v.clear(vy);
  params.w.clear(vz);

  params.bc.xpos.type = BC_PERIODIC;
  params.bc.xneg.type = BC_PERIODIC;
  params.bc.ypos.type = BC_PERIODIC;
  params.bc.yneg.type = BC_PERIODIC;
  params.bc.zpos.type = BC_PERIODIC;
  params.bc.zneg.type = BC_PERIODIC;

  int fourth = dim / 4;
  
  int i,j,k;

  if (pat == SQUARE) {
    for (i=0; i < dim; i++)
      for (j=0; j < dim; j++)
        for (k=0; k < dim; k++) {
          if (i > fourth && i < 3*fourth && j > fourth && j < 3*fourth && k > fourth && k < 3*fourth)
            params.initial_values.at(i,j,k) = 1;
          else
            params.initial_values.at(i,j,k) = 0;
        }
  }
  else {
    for (i=0; i < dim; i++)
      for (j=0; j < dim; j++)
        for (k=0; k < dim; k++) {
          float idx;
          if (axis == 0) { idx = i; }
          if (axis == 3) { idx = dim - i - 1; }
          if (axis == 1) { idx = j; }
          if (axis == 4) { idx = dim - j - 1; }
          if (axis == 2) { idx = k; }
          if (axis == 5) { idx = dim - k - 1; }

          params.initial_values.at(i,j,k) = (idx/dim) * (idx/dim);
        }

  }
}

DECLARE_UNITTEST_DOUBLE_BEGIN(Advection3DDoubleSymmetryTest);

void run_test(InterpolationType interp_type, double tol)
{
  int dim = 32;

  Eqn_ScalarAdvection3DParamsD params_x;
  create_params(dim, 0, GRADIENT, params_x, interp_type);
  params_x.hx = .6f;
  Eqn_ScalarAdvection3DD eqn_x;
  UNITTEST_ASSERT_TRUE(eqn_x.set_parameters(params_x));

  Eqn_ScalarAdvection3DParamsD params_y;
  create_params(dim, 1, GRADIENT, params_y, interp_type);
  params_y.hy = .6f;
  Eqn_ScalarAdvection3DD eqn_y;
  UNITTEST_ASSERT_TRUE(eqn_y.set_parameters(params_y));

  Eqn_ScalarAdvection3DParamsD params_z;
  create_params(dim, 2, GRADIENT, params_z, interp_type);
  params_z.hz = .6f;
  Eqn_ScalarAdvection3DD eqn_z;
  UNITTEST_ASSERT_TRUE(eqn_z.set_parameters(params_z));

  Eqn_ScalarAdvection3DParamsD params_xn;
  create_params(dim, 3, GRADIENT, params_xn, interp_type);
  params_xn.hx = .6f;
  Eqn_ScalarAdvection3DD eqn_xn;
  UNITTEST_ASSERT_TRUE(eqn_xn.set_parameters(params_xn));

  Eqn_ScalarAdvection3DParamsD params_yn;
  create_params(dim, 4, GRADIENT, params_yn, interp_type);
  params_yn.hy = .6f;
  Eqn_ScalarAdvection3DD eqn_yn;
  UNITTEST_ASSERT_TRUE(eqn_yn.set_parameters(params_yn));

  Eqn_ScalarAdvection3DParamsD params_zn;
  create_params(dim, 5, GRADIENT, params_zn, interp_type);
  params_zn.hz = .6f;
  Eqn_ScalarAdvection3DD eqn_zn;
  UNITTEST_ASSERT_TRUE(eqn_zn.set_parameters(params_zn));

  double eqn_zn_integral_before;
  params_zn.initial_values.reduce_sum(eqn_zn_integral_before);
  double eqn_yn_integral_before;
  params_yn.initial_values.reduce_sum(eqn_yn_integral_before);
  double eqn_xn_integral_before;
  params_xn.initial_values.reduce_sum(eqn_xn_integral_before);
  double eqn_zp_integral_before;
  params_z.initial_values.reduce_sum(eqn_zp_integral_before);
  double eqn_yp_integral_before;
  params_y.initial_values.reduce_sum(eqn_yp_integral_before);
  double eqn_xp_integral_before;
  params_x.initial_values.reduce_sum(eqn_xp_integral_before);


  int i,j,k;
  Grid3DHost<double> h_eqn_x, h_eqn_y, h_eqn_z, h_eqn_xn, h_eqn_yn, h_eqn_zn;
  h_eqn_x.init_congruent(eqn_x.phi);
  h_eqn_y.init_congruent(eqn_y.phi);
  h_eqn_z.init_congruent(eqn_z.phi);
  h_eqn_xn.init_congruent(eqn_xn.phi);
  h_eqn_yn.init_congruent(eqn_yn.phi);
  h_eqn_zn.init_congruent(eqn_zn.phi);

  for (int t=0; t < 10; t++) {
    eqn_x.advance(1.0);
    eqn_y.advance(1.0);
    eqn_z.advance(1.0);
    eqn_xn.advance(1.0);
    eqn_yn.advance(1.0);
    eqn_zn.advance(1.0);
    h_eqn_x.copy_all_data(eqn_x.phi);
    h_eqn_y.copy_all_data(eqn_y.phi);
    h_eqn_z.copy_all_data(eqn_z.phi);
    h_eqn_xn.copy_all_data(eqn_xn.phi);
    h_eqn_yn.copy_all_data(eqn_yn.phi);
    h_eqn_zn.copy_all_data(eqn_zn.phi);

    // test symmetry
    for (i=0; i < dim; i++)
      for (j=0; j < dim; j++)
        for (k=0; k < dim; k++) {
          double  val_x = h_eqn_x.at(i,j,k);
          double  val_y = h_eqn_y.at(j,i,k);
          double  val_z = h_eqn_z.at(k,j,i);
          double  val_xn = h_eqn_xn.at(dim-i-1,j,k);
          double  val_yn = h_eqn_yn.at(j,dim-i-1,k);
          double  val_zn = h_eqn_zn.at(k,j,dim-i-1);

          UNITTEST_ASSERT_EQUAL_DOUBLE(val_x, val_y, 1e-12);
          UNITTEST_ASSERT_EQUAL_DOUBLE(val_x, val_z, 1e-12);
          UNITTEST_ASSERT_EQUAL_DOUBLE(val_y, val_z, 1e-12);

          UNITTEST_ASSERT_EQUAL_DOUBLE(val_x, val_xn, 1e-12);
          UNITTEST_ASSERT_EQUAL_DOUBLE(val_y, val_yn, 1e-12);
          UNITTEST_ASSERT_EQUAL_DOUBLE(val_z, val_zn, 1e-12);
        }
  }

  double eqn_zn_integral_after;
  eqn_zn.phi.reduce_sum(eqn_zn_integral_after);
  double eqn_yn_integral_after;
  eqn_yn.phi.reduce_sum(eqn_yn_integral_after);
  double eqn_xn_integral_after;
  eqn_xn.phi.reduce_sum(eqn_xn_integral_after);
  double eqn_zp_integral_after;
  eqn_z.phi.reduce_sum(eqn_zp_integral_after);
  double eqn_yp_integral_after;
  eqn_y.phi.reduce_sum(eqn_yp_integral_after);
  double eqn_xp_integral_after;
  eqn_x.phi.reduce_sum(eqn_xp_integral_after);

  // these tolerances are very high, probably because of single precision only...
  UNITTEST_ASSERT_EQUAL_DOUBLE(eqn_xp_integral_after, eqn_xp_integral_before, tol * eqn_xp_integral_before);
  UNITTEST_ASSERT_EQUAL_DOUBLE(eqn_yp_integral_after, eqn_yp_integral_before, tol * eqn_yp_integral_before);
  UNITTEST_ASSERT_EQUAL_DOUBLE(eqn_zp_integral_after, eqn_zp_integral_before, tol * eqn_zp_integral_before);
  UNITTEST_ASSERT_EQUAL_DOUBLE(eqn_xn_integral_after, eqn_xn_integral_before, tol * eqn_xn_integral_before);
  UNITTEST_ASSERT_EQUAL_DOUBLE(eqn_yn_integral_after, eqn_yn_integral_before, tol * eqn_yn_integral_before);
  UNITTEST_ASSERT_EQUAL_DOUBLE(eqn_zn_integral_after, eqn_zn_integral_before, tol * eqn_zn_integral_before);
}

void run() {
  run_test(IT_FIRST_ORDER_UPWIND, 1e-12);
  run_test(IT_SECOND_ORDER_CENTERED, 1e-11); // even though is unstable, it should be symmetric for a while
}

DECLARE_UNITTEST_DOUBLE_END(Advection3DDoubleSymmetryTest);

DECLARE_UNITTEST_DOUBLE_BEGIN(Advection3DDoubleTest);
void run()
{
  int dim = 128;

  Eqn_ScalarAdvection3DParamsD params;
  create_params(dim, 0, SQUARE, params, IT_FIRST_ORDER_UPWIND);
  Eqn_ScalarAdvection3DD eqn;
  UNITTEST_ASSERT_TRUE(eqn.set_parameters(params));

  ImageFile img;
  img.allocate(dim, 100);

  int i,j,k;

  int fourth = dim/4;
  Grid3DHostD density;
  density.init_congruent(eqn.phi);

  for (int t=0; t < 100; t++) {
    printf("frame %d\n", t);
    eqn.advance(1.0);
    density.copy_all_data(eqn.phi);
    
    int i,j,k;
    for (i=0; i < dim; i++)
      for (j=0; j < dim; j++)
        for (k=0; k < dim; k++) {

          // everything should be finite - no nans
          UNITTEST_ASSERT_FINITE(density.at(i,j,k));

          // outside of the central half all values should be zero since there is no velocity in the y or z directions.
          if (!(j > fourth && j < 3*fourth && k > fourth && k < 3*fourth))
            UNITTEST_ASSERT_EQUAL_DOUBLE(density.at(i,j,k), 0, 0);

        }

    for (i=0; i < dim; i++) {
      double soln_val = density.at(i,dim/2, dim/2);
      if (soln_val > 1.0) soln_val = 1.0;
      unsigned char color = (unsigned char) (255 * (soln_val / 1.0));
      img.set_rgb(i,t, color, color, color);
    }
  }

  img.write_ppm("advection_d.ppm");
}
DECLARE_UNITTEST_DOUBLE_END(Advection3DDoubleTest);


DECLARE_UNITTEST_DOUBLE_BEGIN(Advection3DDoubleSwirlTest);
void run()
{
  Eqn_ScalarAdvection3DParamsD params;
  Grid3DHostD hu,hv,hw;  

  int dim = 64;
  float h = 1.0/dim;
  params.init_grids(dim, dim, dim);
  params.hx = h;
  params.hy = h;
  params.hz = h;

  int i,j,k;

  // set to an initial shape
  for (i=0; i < dim; i++)
    for (j=0; j < dim; j++)
      for (k=0; k < dim; k++) {
        double x,y,z;
        // position of neg u face
        x = i*h;
        y = (j+.5)*h;
        z = (k+.5)*h;

        params.u.at(i,j,k) = y - .5;

        // position of neg v face
        x = (i+.5)*h;
        y = j*h;
        z = (k+.5)*h;

        params.v.at(i,j,k) = .5 - x;

        params.w.at(i,j,k) = 0;
        
        // cell center position
        x = (i+.5)*h;
        y = (j+.5)*h;
        z = (k+.5)*h;
        
        // initial density is a sphere centered at (.5, .7, .5) with radius .2
        double rad = sqrt((x-.5)*(x-.5) + (y-.7)*(y-.7) + (z-.5)*(z-.5));
        params.initial_values.at(i,j,k) = rad < .2 ? 1 : 0;
      }



  params.nx = dim;
  params.ny = dim;
  params.nz = dim;
  params.hx = h;
  params.hy = h;
  params.hz = h;
  params.advection_scheme = IT_FIRST_ORDER_UPWIND;
  BoundaryCondition bc_example;
  bc_example.type = BC_PERIODIC;
  params.bc = BoundaryConditionSet(bc_example);


  Eqn_ScalarAdvection3DD eqn;
  UNITTEST_ASSERT_TRUE(eqn.set_parameters(params));



  Sol_ProjectDivergence3DDeviceD projector;
  BoundaryCondition proj_example;
  proj_example.type = BC_FORCED_INFLOW_VARIABLE_SLIP;

  projector.bc = BoundaryConditionSet(proj_example);
  UNITTEST_ASSERT_TRUE(apply_3d_mac_boundary_conditions_level1(eqn.u,eqn.v,eqn.w,projector.bc, h, h, h));
  UNITTEST_ASSERT_TRUE(projector.initialize_storage(eqn.nx(), eqn.ny(), eqn.nz(), eqn.hx(), eqn.hy(), eqn.hz(), &eqn.u, &eqn.v, &eqn.w));
  UNITTEST_ASSERT_TRUE(projector.solve(1e-10));

  double integral_before;
  eqn.phi.reduce_sum(integral_before);


  CPUTimer timer;
  double elapsed = 0;

  double pi = 2 * asin(1.0);
  // time it take for 1 revolution to complete, traveling at max a velocity of .5
  double rev_dt = 2 * pi;
  int n_steps = (int)ceil(1.5*pi*dim);
  for (int step=0; step < n_steps; step++) {
//    printf("step %d\n", step);

    timer.start();
    eqn.advance(rev_dt / n_steps);
    timer.stop();
    elapsed += timer.elapsed_ms();

/*
    ImageFile img;
    img.allocate(dim, dim);
    for (i=0; i < dim; i++)
      for (j=0; j < dim; j++) {
        double d = eqn.density().at(i,j,dim/2);
        unsigned char byte = (unsigned char) (255 * min(d, 1.0));
        img.set_rgb(i,j, byte, byte, byte);
      }
      char buff[1024];
      sprintf(buff, "swirl.%04d.ppm", step);
      img.write_ppm(buff);
*/
  }

  double integral_after;
  eqn.phi.reduce_sum(integral_after);

  //printf("Avg ms = %f\n", elapsed / n_steps);

  UNITTEST_ASSERT_EQUAL_DOUBLE(integral_before, integral_after, 1e-10);

  // integral should be the same - done
  // output slices & verify visually - done
  // test convergence to original shape (A = Orig - Advected, error = L2(A))
  // test symmetry
}

DECLARE_UNITTEST_DOUBLE_END(Advection3DDoubleSwirlTest);
