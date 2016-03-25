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
#include "ocustorage/grid3d.h"
#include "ocuutil/timer.h"
#include "ocuutil/float_routines.h"

using namespace ocu;

DECLARE_UNITTEST_BEGIN(Grid3DTest);


void run()
{
  Grid3DHostF host, host2;
  Grid3DDeviceF device;

  UNITTEST_ASSERT_TRUE(host.init(12,13,14,1,1,1));
  UNITTEST_ASSERT_TRUE(host2.init(12,13,14,1,1,1));
  UNITTEST_ASSERT_TRUE(device.init(12,13,14,1,1,1));

  int i,j,k;

  for(i=0; i < 12; i++)
    for (j=0; j < 13; j++)
      for (k=0; k < 14; k++)
        host.at(i,j,k) = (i+.41) * -j + k * .11;

  UNITTEST_ASSERT_TRUE(device.copy_all_data(host));
  UNITTEST_ASSERT_TRUE(device.linear_combination(.5, device));
  UNITTEST_ASSERT_TRUE(host2.copy_all_data(device));

  for(i=0; i < 12; i++)
    for (j=0; j < 13; j++)
      for (k=0; k < 14; k++)
        UNITTEST_ASSERT_EQUAL_FLOAT(host2.at(i,j,k), host.at(i,j,k) * .5, 0);
}

DECLARE_UNITTEST_END(Grid3DTest);


DECLARE_UNITTEST_BEGIN(Grid3DReduceTest);


#define NUM_TESTS 5
void run()
{
  int nx[NUM_TESTS] = {12,32,128,651,1};
  int ny[NUM_TESTS] = {13,32,128,62,1};
  int nz[NUM_TESTS] = {14,32,128,121,351};
  int gx[NUM_TESTS] = {1,0,2,2,0};
  int gy[NUM_TESTS] = {1,0,2,2,0};
  int gz[NUM_TESTS] = {1,0,3,1,3};
  float tol[NUM_TESTS] = {1e-4, 1e-3,1e-3,2e-3,1e-4};

  for (int test=0; test < NUM_TESTS; test++) {

    Grid3DHostF host;
    Grid3DDeviceF device;

    UNITTEST_ASSERT_TRUE(host.init(nx[test], ny[test], nz[test], gx[test], gy[test], gz[test]));
    UNITTEST_ASSERT_TRUE(device.init(nx[test], ny[test], nz[test], gx[test], gy[test], gz[test]));

    int i,j,k;

    for(i=0; i < nx[test]; i++)
      for (j=0; j < ny[test]; j++)
        for (k=0; k < nz[test]; k++)
          host.at(i,j,k) = .001 * ((i+.41) * -j + k * .11);

    UNITTEST_ASSERT_TRUE(device.copy_all_data(host));

    float host_red, dev_red;
    UNITTEST_ASSERT_TRUE(host.reduce_maxabs(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_maxabs(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, 0);

    UNITTEST_ASSERT_TRUE(host.reduce_max(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_max(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, 0);

    UNITTEST_ASSERT_TRUE(host.reduce_min(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_min(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, 0);

    UNITTEST_ASSERT_TRUE(host.reduce_sum(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_sum(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, fabs(host_red) * tol[test]);

    UNITTEST_ASSERT_TRUE(host.reduce_sqrsum(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_sqrsum(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, fabs(host_red) * tol[test]);

    UNITTEST_ASSERT_TRUE(host.reduce_checknan(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_checknan(dev_red));
    UNITTEST_ASSERT_FINITE(host_red);
    UNITTEST_ASSERT_FINITE(dev_red);

    // set one value to nan, check that checknan catches it
    float inf = HUGE_VAL * HUGE_VAL;
    UNITTEST_ASSERT_TRUE(!check_float(inf));
    host.at(nx[test]/2, ny[test]/2, nz[test]/2) = inf;
    UNITTEST_ASSERT_TRUE(device.copy_all_data(host));
    UNITTEST_ASSERT_TRUE(host.reduce_checknan(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_checknan(dev_red));
    UNITTEST_ASSERT_TRUE(!check_float(host_red));
    UNITTEST_ASSERT_TRUE(!check_float(dev_red));

  }

}

DECLARE_UNITTEST_END(Grid3DReduceTest);


DECLARE_UNITTEST_BEGIN(Reduce3DTimingTest);

void run()
{
  CPUTimer timer;
  Grid3DHostF host;
  Grid3DDeviceF device;

  UNITTEST_ASSERT_TRUE(host.init(256,256,256,1,1,1));
  UNITTEST_ASSERT_TRUE(device.init(256,256,256,1,1,1));

  int i,j,k;

  for(i=0; i < 256; i++)
    for (j=0; j < 256; j++)
      for (k=0; k < 256; k++)
        host.at(i,j,k) = .001 * ((i+.41) * -j + k * .11);

  UNITTEST_ASSERT_TRUE(device.copy_all_data(host));

  float host_red, dev_red;
  float host_time, dev_time;

  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_maxabs(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_maxabs(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);


  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_max(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_max(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);


  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_min(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_min(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);



  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_sum(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_sum(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);

  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_sqrsum(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_sqrsum(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);

}

DECLARE_UNITTEST_END(Reduce3DTimingTest);


DECLARE_UNITTEST_DOUBLE_BEGIN(Grid3DReduceDoubleTest);

#define D_NUM_TESTS 5
void run()
{
  int nx[D_NUM_TESTS] = {12,32,128,651,1};
  int ny[D_NUM_TESTS] = {13,32,128,62,1};
  int nz[D_NUM_TESTS] = {14,32,128,121,351};
  int gx[D_NUM_TESTS] = {1,0,2,2,0};
  int gy[D_NUM_TESTS] = {1,0,2,2,0};
  int gz[D_NUM_TESTS] = {1,0,3,1,3};
  float tol[D_NUM_TESTS] = {0,0,0,0,0};

  //set_forge_ahead(true);

  for (int test=0; test < D_NUM_TESTS; test++) {
    Grid3DHostD host;
    Grid3DDeviceD device;

    UNITTEST_ASSERT_TRUE(host.init(nx[test], ny[test], nz[test], gx[test], gy[test], gz[test]));
    UNITTEST_ASSERT_TRUE(device.init(nx[test], ny[test], nz[test], gx[test], gy[test], gz[test]));

    int i,j,k;

    for(i=0; i < nx[test]; i++)
      for (j=0; j < ny[test]; j++)
        for (k=0; k < nz[test]; k++)
          host.at(i,j,k) = .001 * ((i+.41) * -j + k * .1134523543511233412312341234);

    UNITTEST_ASSERT_TRUE(device.copy_all_data(host));

    double host_red, dev_red;
    UNITTEST_ASSERT_TRUE(host.reduce_maxabs(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_maxabs(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, 0);

    UNITTEST_ASSERT_TRUE(host.reduce_max(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_max(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, 0);

    UNITTEST_ASSERT_TRUE(host.reduce_min(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_min(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, 0);

    UNITTEST_ASSERT_TRUE(host.reduce_sum(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_sum(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, fabs(host_red) * tol[test]);

    UNITTEST_ASSERT_TRUE(host.reduce_sqrsum(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_sqrsum(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, fabs(host_red) * tol[test]);
  }
}

DECLARE_UNITTEST_DOUBLE_END(Grid3DReduceDoubleTest);



DECLARE_UNITTEST_DOUBLE_BEGIN(Reduce3DDoubleTimingTest);

void run()
{
  CPUTimer timer;
  Grid3DHostD host;
  Grid3DDeviceD device;

  UNITTEST_ASSERT_TRUE(host.init(256,256,256,1,1,1));
  UNITTEST_ASSERT_TRUE(device.init(256,256,256,1,1,1));

  int i,j,k;

  for(i=0; i < 256; i++)
    for (j=0; j < 256; j++)
      for (k=0; k < 256; k++)
        host.at(i,j,k) = .001 * ((i+.41) * -j + k * .11);

  UNITTEST_ASSERT_TRUE(device.copy_all_data(host));

  double host_red, dev_red;
  double host_time, dev_time;

  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_maxabs(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_maxabs(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);


  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_max(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_max(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);


  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_min(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_min(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);



  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_sum(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_sum(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);

  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_sqrsum(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_sqrsum(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);
}

DECLARE_UNITTEST_DOUBLE_END(Reduce3DDoubleTimingTest);
