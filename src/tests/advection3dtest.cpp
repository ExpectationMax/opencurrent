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
#include "ocuequation/eqn_scalaradvection3d.h"
#include "ocuutil/imagefile.h"
#include "ocuutil/float_routines.h"
#include "ocuutil/interpolation.h"


using namespace ocu;

enum Pattern {
  SQUARE,
  GRADIENT
};

void create_params(int dim, int axis, Pattern pat, Eqn_ScalarAdvection3DParamsF &params)
{
  params.init_grids(dim, dim, dim);
  params.hx = 1;
  params.hy = 1;
  params.hz = 1;
  params.advection_scheme = IT_FIRST_ORDER_UPWIND;

  float vx = (axis == 0) ? 1.5 : (axis == 3) ? -1.5 : 0;
  float vy = (axis == 1) ? 1.5 : (axis == 4) ? -1.5 : 0;
  float vz = (axis == 2) ? 1.5 : (axis == 5) ? -1.5 : 0;
  
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


DECLARE_UNITTEST_BEGIN(Advection3DSymmetryTest);

void run()
{
  int dim = 32;

  Eqn_ScalarAdvection3DParamsF params_x;
  Grid3DDeviceF ux, vx, wx;
  create_params(dim, 0, GRADIENT, params_x);
  params_x.hx = .6f;
  Eqn_ScalarAdvection3DF eqn_x;
  UNITTEST_ASSERT_TRUE(eqn_x.set_parameters(params_x));

  Eqn_ScalarAdvection3DParamsF params_y;
  Grid3DDeviceF uy, vy, wy;
  create_params(dim, 1, GRADIENT, params_y);
  params_y.hy = .6f;
  Eqn_ScalarAdvection3DF eqn_y;
  UNITTEST_ASSERT_TRUE(eqn_y.set_parameters(params_y));

  Eqn_ScalarAdvection3DParamsF params_z;
  Grid3DDeviceF uz, vz, wz;
  create_params(dim, 2, GRADIENT, params_z);
  params_z.hz = .6f;
  Eqn_ScalarAdvection3DF eqn_z;
  UNITTEST_ASSERT_TRUE(eqn_z.set_parameters(params_z));

  Eqn_ScalarAdvection3DParamsF params_xn;
  Grid3DDeviceF uxn, vxn, wxn;
  create_params(dim, 3, GRADIENT, params_xn);
  params_xn.hx = .6f;
  Eqn_ScalarAdvection3DF eqn_xn;
  UNITTEST_ASSERT_TRUE(eqn_xn.set_parameters(params_xn));

  Eqn_ScalarAdvection3DParamsF params_yn;
  Grid3DDeviceF uyn, vyn, wyn;
  create_params(dim, 4, GRADIENT, params_yn);
  params_yn.hy = .6f;
  Eqn_ScalarAdvection3DF eqn_yn;
  UNITTEST_ASSERT_TRUE(eqn_yn.set_parameters(params_yn));

  Eqn_ScalarAdvection3DParamsF params_zn;
  Grid3DDeviceF uzn, vzn, wzn;
  create_params(dim, 5, GRADIENT, params_zn);
  params_zn.hz = .6f;
  Eqn_ScalarAdvection3DF eqn_zn;
  UNITTEST_ASSERT_TRUE(eqn_zn.set_parameters(params_zn));

  float eqn_zn_integral_before;
  params_zn.initial_values.reduce_sum(eqn_zn_integral_before);
  float eqn_yn_integral_before;
  params_yn.initial_values.reduce_sum(eqn_yn_integral_before);
  float eqn_xn_integral_before;
  params_xn.initial_values.reduce_sum(eqn_xn_integral_before);
  float eqn_zp_integral_before;
  params_z.initial_values.reduce_sum(eqn_zp_integral_before);
  float eqn_yp_integral_before;
  params_y.initial_values.reduce_sum(eqn_yp_integral_before);
  float eqn_xp_integral_before;
  params_x.initial_values.reduce_sum(eqn_xp_integral_before);


  int i,j,k;
  Grid3DHost<float> h_eqn_x, h_eqn_y, h_eqn_z, h_eqn_xn, h_eqn_yn, h_eqn_zn;
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
          float val_x = h_eqn_x.at(i,j,k);
          float val_y = h_eqn_y.at(j,i,k);
          float val_z = h_eqn_z.at(k,j,i);
          float val_xn = h_eqn_xn.at(dim-i-1,j,k);
          float val_yn = h_eqn_yn.at(j,dim-i-1,k);
          float val_zn = h_eqn_zn.at(k,j,dim-i-1);

          UNITTEST_ASSERT_EQUAL_FLOAT(val_x, val_y, 1e-5);
          UNITTEST_ASSERT_EQUAL_FLOAT(val_x, val_z, 1e-5);
          UNITTEST_ASSERT_EQUAL_FLOAT(val_y, val_z, 1e-5);

          UNITTEST_ASSERT_EQUAL_FLOAT(val_x, val_xn, 1e-5);
          UNITTEST_ASSERT_EQUAL_FLOAT(val_y, val_yn, 1e-5);
          UNITTEST_ASSERT_EQUAL_FLOAT(val_z, val_zn, 1e-5);
        }
  }

  float eqn_zn_integral_after;
  eqn_zn.phi.reduce_sum(eqn_zn_integral_after);
  float eqn_yn_integral_after;
  eqn_yn.phi.reduce_sum(eqn_yn_integral_after);
  float eqn_xn_integral_after;
  eqn_xn.phi.reduce_sum(eqn_xn_integral_after);
  float eqn_zp_integral_after;
  eqn_z.phi.reduce_sum(eqn_zp_integral_after);
  float eqn_yp_integral_after;
  eqn_y.phi.reduce_sum(eqn_yp_integral_after);
  float eqn_xp_integral_after;
  eqn_x.phi.reduce_sum(eqn_xp_integral_after);

  // these tolerances are very high, probably because of single precision only...
  UNITTEST_ASSERT_EQUAL_FLOAT(eqn_xp_integral_after, eqn_xp_integral_before, 1e-3 * eqn_xp_integral_before);
  UNITTEST_ASSERT_EQUAL_FLOAT(eqn_yp_integral_after, eqn_yp_integral_before, 1e-3 * eqn_yp_integral_before);
  UNITTEST_ASSERT_EQUAL_FLOAT(eqn_zp_integral_after, eqn_zp_integral_before, 1e-3 * eqn_zp_integral_before);
  UNITTEST_ASSERT_EQUAL_FLOAT(eqn_xn_integral_after, eqn_xn_integral_before, 1e-3 * eqn_xn_integral_before);
  UNITTEST_ASSERT_EQUAL_FLOAT(eqn_yn_integral_after, eqn_yn_integral_before, 1e-3 * eqn_yn_integral_before);
  UNITTEST_ASSERT_EQUAL_FLOAT(eqn_zn_integral_after, eqn_zn_integral_before, 1e-3 * eqn_zn_integral_before);
}

DECLARE_UNITTEST_END(Advection3DSymmetryTest);



DECLARE_UNITTEST_BEGIN(Advection3DTest);

void run()
{
  int dim = 128;

  Eqn_ScalarAdvection3DParamsF params;
  create_params(dim, 0, SQUARE, params);
  Eqn_ScalarAdvection3DF eqn;
  UNITTEST_ASSERT_TRUE(eqn.set_parameters(params));

  ImageFile img;
  img.allocate(dim, 100);

  int i,j,k;

  int fourth = dim/4;
  Grid3DHostF density;
  density.init_congruent(eqn.phi);

  for (int t=0; t < 100; t++) {
    //printf("frame %d\n", t);
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
            UNITTEST_ASSERT_EQUAL_FLOAT(density.at(i,j,k), 0, 0);

        }

    for (i=0; i < dim; i++) {
      float soln_val = density.at(i,dim/2, dim/2);
      if (soln_val > 1.0f) soln_val = 1.0f;
      unsigned char color = (unsigned char) (255 * (soln_val / 1.0f));
      img.set_rgb(i,t, color, color, color);
    }
  }

  //img.write_ppm("advection.ppm");
}

DECLARE_UNITTEST_END(Advection3DTest);
