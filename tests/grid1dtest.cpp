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
#include "ocustorage/grid1d.h"
#include "ocuutil/timer.h"


using namespace ocu;


DECLARE_UNITTEST_BEGIN(Grid1DReduceTest);


#define NUM_TESTS 5
void run()
{
  int nx[NUM_TESTS] = {1024*1024,1024*1024,145,325681,325681};
  int gx[NUM_TESTS] = {1,0,2,2,0};
  float tol[NUM_TESTS] = {1e-3,1e-3,1e-3,1e-3,1e-3};


  for (int test=0; test < NUM_TESTS; test++) {

    Grid1DHostF host;
    Grid1DDeviceF device;

    UNITTEST_ASSERT_TRUE(host.init(nx[test], gx[test]));
    UNITTEST_ASSERT_TRUE(device.init(nx[test], gx[test]));

    int i;

    for(i=0; i < nx[test]; i++)
      host.at(i) = .001 * (i+.41);

    UNITTEST_ASSERT_TRUE(device.copy_all_data(host));

    float host_red, dev_red;
    UNITTEST_ASSERT_TRUE(host.reduce_maxabs(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_maxabs(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, 0);


    UNITTEST_ASSERT_TRUE(host.reduce_sum(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_sum(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, fabs(host_red) * tol[test]);

    UNITTEST_ASSERT_TRUE(host.reduce_sqrsum(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_sqrsum(dev_red));
    UNITTEST_ASSERT_EQUAL_FLOAT(host_red, dev_red, fabs(host_red) * tol[test]);
  }

}

DECLARE_UNITTEST_END(Grid1DReduceTest);


DECLARE_UNITTEST_DOUBLE_BEGIN(Grid1DReduceDoubleTest);

#define D_NUM_TESTS 5

void run()
{
  int nx[D_NUM_TESTS] = {1024*1024,1024*1024,145,325681,325681};
  int gx[D_NUM_TESTS] = {1,0,2,2,0};
  float tol[D_NUM_TESTS] = {1e-13,1e-13,1e-13,1e-13,1e-13};

  //set_forge_ahead(true);

  for (int test=0; test < D_NUM_TESTS; test++) {

    Grid1DHostD host;
    Grid1DDeviceD device;

    UNITTEST_ASSERT_TRUE(host.init(nx[test], gx[test]));
    UNITTEST_ASSERT_TRUE(device.init(nx[test], gx[test]));

    int i;

    for(i=0; i < nx[test]; i++)
      host.at(i) = .001 * (i+.41);

    UNITTEST_ASSERT_TRUE(device.copy_all_data(host));

    double host_red, dev_red;
    UNITTEST_ASSERT_TRUE(host.reduce_maxabs(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_maxabs(dev_red));
    UNITTEST_ASSERT_EQUAL_DOUBLE(host_red, dev_red, 0);

    UNITTEST_ASSERT_TRUE(host.reduce_max(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_max(dev_red));
    UNITTEST_ASSERT_EQUAL_DOUBLE(host_red, dev_red, 0);

    UNITTEST_ASSERT_TRUE(host.reduce_min(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_min(dev_red));
    UNITTEST_ASSERT_EQUAL_DOUBLE(host_red, dev_red, 0);

    UNITTEST_ASSERT_TRUE(host.reduce_sum(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_sum(dev_red));
    UNITTEST_ASSERT_EQUAL_DOUBLE(host_red, dev_red, fabs(host_red) * tol[test]);

    UNITTEST_ASSERT_TRUE(host.reduce_sqrsum(host_red));
    UNITTEST_ASSERT_TRUE(device.reduce_sqrsum(dev_red));
    UNITTEST_ASSERT_EQUAL_DOUBLE(host_red, dev_red, fabs(host_red) * tol[test]);
  }
}

DECLARE_UNITTEST_DOUBLE_END(Grid1DReduceDoubleTest);



DECLARE_UNITTEST_BEGIN(Reduce1DTimingTest);

void run()
{
  CPUTimer timer;
  Grid1DHostF host;
  Grid1DDeviceF device;

  UNITTEST_ASSERT_TRUE(host.init(1024*1024,1));
  UNITTEST_ASSERT_TRUE(device.init(1024*1024,1));

  int i,j,k;

  for(i=0; i < 1024*1024; i++)
        host.at(i) = .001 * (i+.41);

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
  printf("1D device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);


  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_sum(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_sum(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("1D device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);

  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_sqrsum(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_sqrsum(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("1D device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);

}

DECLARE_UNITTEST_END(Reduce1DTimingTest);


DECLARE_UNITTEST_DOUBLE_BEGIN(Reduce1DDoubleTimingTest);

void run()
{
  CPUTimer timer;
  Grid1DHostD host;
  Grid1DDeviceD device;

  UNITTEST_ASSERT_TRUE(host.init(1024*1024*10,1));
  UNITTEST_ASSERT_TRUE(device.init(1024*1024*10,1));

  int i,j,k;

  for(i=0; i < 1024*1024; i++)
        host.at(i) = .001 * (i+.41);

  UNITTEST_ASSERT_TRUE(device.copy_all_data(host));

  double host_red, dev_red;
  float host_time, dev_time;

  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_maxabs(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_maxabs(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("Double 1D: device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);

  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_max(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_max(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("Double 1D: device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);

  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_min(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_min(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("Double 1D: device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);



  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_sum(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_sum(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("Double 1D: device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);

  timer.start();
  UNITTEST_ASSERT_TRUE(host.reduce_sqrsum(host_red));
  timer.stop();
  host_time = timer.elapsed_ms();

  timer.start();
  UNITTEST_ASSERT_TRUE(device.reduce_sqrsum(dev_red));
  timer.stop();
  dev_time = timer.elapsed_ms();
  printf("Double 1D: device reduce = %fms, host reduce = %fms, speedup = %fx\n", dev_time, host_time, host_time/dev_time);
}

DECLARE_UNITTEST_DOUBLE_END(Reduce1DDoubleTimingTest);

