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
#include "ocuutil/reduction_op.h"
#include "ocustorage/coarray.h"

using namespace ocu;


DECLARE_UNITTEST_MULTIGPU_BEGIN(MultiReduceTest);

void run()
{
  UNITTEST_ASSERT_EQUAL_INT(ThreadManager::barrier_reduce(1, HostReduceSum<int>()), ThreadManager::num_images());
  UNITTEST_ASSERT_EQUAL_FLOAT(ThreadManager::barrier_reduce(1.0f, HostReduceSum<float>()), (float)ThreadManager::num_images(), 0);
  UNITTEST_ASSERT_EQUAL_DOUBLE(ThreadManager::barrier_reduce(1.0, HostReduceSum<double>()), (double)ThreadManager::num_images(), 0);

  int sqr_sum = 0;
  for (int i=0; i < ThreadManager::num_images(); i++)
    sqr_sum += i*i;
  UNITTEST_ASSERT_EQUAL_INT(ThreadManager::barrier_reduce(ThreadManager::this_image(), HostReduceSqrSum<int>()), sqr_sum);
  UNITTEST_ASSERT_EQUAL_FLOAT(ThreadManager::barrier_reduce((float)ThreadManager::this_image(), HostReduceSqrSum<float>()), (float)sqr_sum, 0);
  UNITTEST_ASSERT_EQUAL_DOUBLE(ThreadManager::barrier_reduce((double)ThreadManager::this_image(), HostReduceSqrSum<double>()), (double)sqr_sum, 0);

  UNITTEST_ASSERT_EQUAL_INT(ThreadManager::barrier_reduce(ThreadManager::this_image(), HostReduceMax<int>()), ThreadManager::num_images()-1);
  UNITTEST_ASSERT_EQUAL_FLOAT(ThreadManager::barrier_reduce((float)ThreadManager::this_image(), HostReduceMax<float>()), (float)(ThreadManager::num_images()-1), 0);
  UNITTEST_ASSERT_EQUAL_DOUBLE(ThreadManager::barrier_reduce((double)ThreadManager::this_image(), HostReduceMax<double>()), (double)(ThreadManager::num_images()-1), 0);

  UNITTEST_ASSERT_EQUAL_INT(ThreadManager::barrier_reduce(ThreadManager::this_image(), HostReduceMin<int>()), 0);
  UNITTEST_ASSERT_EQUAL_FLOAT(ThreadManager::barrier_reduce((float)ThreadManager::this_image(), HostReduceMin<float>()), 0, 0);
  UNITTEST_ASSERT_EQUAL_DOUBLE(ThreadManager::barrier_reduce((double)ThreadManager::this_image(), HostReduceMin<double>()), 0, 0);

  UNITTEST_ASSERT_EQUAL_INT(ThreadManager::barrier_reduce(-ThreadManager::this_image(), HostReduceMaxAbsI()), ThreadManager::num_images()-1);
  UNITTEST_ASSERT_EQUAL_FLOAT(ThreadManager::barrier_reduce(-(float)ThreadManager::this_image(), HostReduceMaxAbsF()), (float)(ThreadManager::num_images()-1), 0);
  UNITTEST_ASSERT_EQUAL_DOUBLE(ThreadManager::barrier_reduce(-(double)ThreadManager::this_image(), HostReduceMaxAbsD()), (double)(ThreadManager::num_images()-1), 0);

}

DECLARE_UNITTEST_END(MultiReduceTest);


DECLARE_UNITTEST_MULTIGPU_BEGIN(CoArray1DTest);


void fill_grid(Grid1DHostI &h_a, int id)
{
  int i;
  for (i=0; i < 66; i++)
    h_a.at(i-1) = id * 66 + i;
}

void recover(int &id, int &x, int val)
{
  id = val / 66;
  val -= id * 66;

  x = val;
  x--;
}

void check_grid(Grid1DHostI &h_a, int id, int from_id, const Region1D &rgn, int x_offset)
{
  // all ids in h_a should be id, except those in rgn, which should be from_id.
  // in rgn, recovered x,y,z should be 
  int i;
  for (i=0; i < 64; i++) {
    int recovered_id, recovered_i;
    recover(recovered_id, recovered_i, h_a.at(i));

    if (rgn.is_inside(i)) {
      recovered_i -= x_offset;
      UNITTEST_ASSERT_EQUAL_INT(recovered_id, from_id);
    }
    else 
      UNITTEST_ASSERT_EQUAL_INT(recovered_id, id);

    UNITTEST_ASSERT_EQUAL_INT(recovered_i, i);
  }
}

void test_xfer(int from_tid, int dst_tid, bool dst_on_host, int src_tid, bool src_on_host, int x0, int x1, int x_offset)
{
  int my_tid = ThreadManager::this_image();

  Region1D *dst_rgn, *src_rgn;

  Grid1DHostCoI h_a("h_a");
  Grid1DDeviceCoI d_a("d_a");
  Grid1DHostI h_temp;

  Region1D h_src_rgn = h_a.co(src_tid)->region(x0 + x_offset, x1 + x_offset);
  Region1D h_dst_rgn = h_a.co(dst_tid)->region(x0,x1);
  Region1D d_src_rgn = d_a.co(src_tid)->region(x0 + x_offset, x1 + x_offset);
  Region1D d_dst_rgn = d_a.co(dst_tid)->region(x0,x1);

  h_a.init(64, 1, true);
  h_temp.init(64, 1, true);
  d_a.init(64, 1);

  fill_grid(h_a, my_tid);
  d_a.copy_all_data(h_a);

  if (dst_on_host) 
    dst_rgn = &h_dst_rgn;
  else
    dst_rgn = &d_dst_rgn;

  if (src_on_host)
    src_rgn = &h_src_rgn;
  else
    src_rgn = &d_src_rgn;

  int hdl = CoArrayManager::barrier_allocate(*dst_rgn, *src_rgn);
  CoArrayManager::barrier_exchange(hdl);

  // wait for all exchanges to finish
  CoArrayManager::barrier_exchange_fence();

  if (dst_on_host) {
    check_grid(h_a, my_tid, from_tid, *dst_rgn, x_offset);
  }
  else {
    h_temp.copy_all_data(d_a);
    check_grid(h_temp, my_tid, from_tid, *dst_rgn, x_offset);
  }

  CoArrayManager::barrier_deallocate(hdl);
}

void run_directional_test(bool dst_on_host, bool src_on_host)
{
  int tid = ThreadManager::this_image();
  int nbr_tid = (tid + 1) % ThreadManager::num_images();
  int from_tid = (tid - 1 + ThreadManager::num_images()) % ThreadManager::num_images();

  // typical ghost cell regions
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, -1, 0, 64);
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, -1, 20, 30);
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, 50, 64, -30);

  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, -1, 0, 64);
  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, -1, 20, 30);
  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, 50, 64, -30);

}

void run()
{
  run_directional_test(true, true);
  run_directional_test(true, false);
  run_directional_test(false, true);
  run_directional_test(false, false);

}




DECLARE_UNITTEST_END(CoArray1DTest);



DECLARE_UNITTEST_MULTIGPU_BEGIN(CoArray3DTest);

void fill_grid(Grid3DHostI &h_a, int id)
{
  int i,j,k;
  for (i=0; i < 66; i++)
    for (j=0; j < 66; j++)
      for (k=0; k < 66; k++) {
        h_a.at(i-1,j-1,k-1) = id * 66*66*66 + i * 66*66 + j * 66 + k;
      }
}

void recover(int &id, int &x, int &y, int &z, int val)
{
  id = val / (66 * 66 * 66);
  val -= (id * 66 * 66 * 66);

  x = val / (66 * 66);
  val -= (x * 66 * 66);
  y = val / 66;
  val -= (y * 66);
  z = val;

  x--;
  y--;
  z--;
}

void check_grid(Grid3DHostI &h_a, int id, int from_id, const Region3D &rgn, int x_offset, int y_offset, int z_offset)
{
  // all ids in h_a should be id, except those in rgn, which should be from_id.
  // in rgn, recovered x,y,z should be 
  int i,j,k;
  for (i=0; i < 64; i++)
    for (j=0; j < 64; j++)
      for (k=0; k < 64; k++) {
        int recovered_id, recovered_i, recovered_j, recovered_k;
        recover(recovered_id, recovered_i, recovered_j, recovered_k, h_a.at(i,j,k));

        if (rgn.is_inside(i,j,k)) {
          recovered_i -= x_offset;
          recovered_j -= y_offset;
          recovered_k -= z_offset;
          UNITTEST_ASSERT_EQUAL_INT(recovered_id, from_id);
          if (recovered_id != from_id) {
            printf("%d %d %d\n", i,j,k);
          }
        }
        else 
          UNITTEST_ASSERT_EQUAL_INT(recovered_id, id);

        UNITTEST_ASSERT_EQUAL_INT(recovered_i, i);
        UNITTEST_ASSERT_EQUAL_INT(recovered_j, j);
        UNITTEST_ASSERT_EQUAL_INT(recovered_k, k);
      }
}

void test_xfer(int from_tid, int dst_tid, bool dst_on_host, int src_tid, bool src_on_host, int x0, int x1, int y0, int y1, int z0, int z1, int x_offset, int y_offset, int z_offset)
{
  int my_tid = ThreadManager::this_image();

  Region3D *dst_rgn, *src_rgn;

  Grid3DHostCoI h_a("h_a");
  Grid3DDeviceCoI d_a("d_a");
  Grid3DHostI h_temp;

  Region3D h_src_rgn = h_a.co(src_tid)->region(x0 + x_offset, x1 + x_offset)(y0 + y_offset, y1 + y_offset)(z0 + z_offset, z1 + z_offset);
  Region3D h_dst_rgn = h_a.co(dst_tid)->region(x0,x1)(y0,y1)(z0,z1);
  Region3D d_src_rgn = d_a.co(src_tid)->region(x0 + x_offset, x1 + x_offset)(y0 + y_offset, y1 + y_offset)(z0 + z_offset, z1 + z_offset);
  Region3D d_dst_rgn = d_a.co(dst_tid)->region(x0,x1)(y0,y1)(z0,z1);

  h_a.init(64, 64, 64, 1, 1, 1, true);
  h_temp.init_congruent(h_a, true);
  d_a.init_congruent(h_a);

  fill_grid(h_a, my_tid);
  d_a.copy_all_data(h_a);

  if (dst_on_host) 
    dst_rgn = &h_dst_rgn;
  else
    dst_rgn = &d_dst_rgn;

  if (src_on_host)
    src_rgn = &h_src_rgn;
  else
    src_rgn = &d_src_rgn;

  int hdl = CoArrayManager::barrier_allocate(*dst_rgn, *src_rgn);
  CoArrayManager::barrier_exchange(hdl);

  // wait for all exchanges to finish
  CoArrayManager::barrier_exchange_fence();

  if (dst_on_host) {
    check_grid(h_a, my_tid, from_tid, *dst_rgn, x_offset, y_offset, z_offset);
  }
  else {
    h_temp.copy_all_data(d_a);
    check_grid(h_temp, my_tid, from_tid, *dst_rgn, x_offset, y_offset, z_offset);
  }

  CoArrayManager::barrier_deallocate(hdl);
}

void run_directional_test(bool dst_on_host, bool src_on_host)
{
  int tid = ThreadManager::this_image();
  int nbr_tid = (tid + 1) % ThreadManager::num_images();
  int from_tid = (tid - 1 + ThreadManager::num_images()) % ThreadManager::num_images();

  // typical ghost cell regions
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, 0, 63, 0, 63, -1, 0, 0, 0, 64);
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, 0, 63, -1, 0, 0, 63, 0, 64, 0);
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, -1, 0, 0, 63, 0, 63, 64, 0, 0);

  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, 0, 63, 0, 63, -1, 0, 0, 0, 64);
  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, 0, 63, -1, 0, 0, 63, 0, 64, 0);
  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, -1, 0, 0, 63, 0, 63, 64, 0, 0);

  // entire contiguous chunks
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, -1, 64, -1, 64, -1, 7, 0, 0, 32);
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, -1, 64, -1, 7, -1, 64, 0, 32, 0);
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, -1, 7, -1, 64, -1, 64, 32, 0, 0);

  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, -1, 64, -1, 64, -1, 7, 0, 0, 32);
  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, -1, 64, -1, 7, -1, 64, 0, 32, 0);
  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, -1, 7, -1, 64, -1, 64, 32, 0, 0);

  // random boxes
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, 10, 20, 5, 15, 35, 45, 10, 15, -20);
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, 10, 15, 5, 12, 35, 45, 10, 15, -20);
  test_xfer(nbr_tid, tid, dst_on_host, nbr_tid, src_on_host, 10, 10, 5, 5, 35, 45, 10, 15, -20);

  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, 10, 20, 5, 15, 35, 45, 10, 15, -20);
  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, 10, 15, 5, 12, 35, 45, 10, 15, -20);
  test_xfer(from_tid, nbr_tid, dst_on_host, tid, src_on_host, 10, 10, 5, 5, 35, 45, 10, 15, -20);

}


void run()
{
  run_directional_test(true, true);
  run_directional_test(true, false);
  run_directional_test(false, true);
  run_directional_test(false, false);
}



DECLARE_UNITTEST_END(CoArray3DTest);



DECLARE_UNITTEST_MULTIGPU_DOUBLE_BEGIN(DoubleTransferTest);


union doubleff
{
  int f[2];
  double d;
};

void run()
{
  int tid = ThreadManager::this_image();
  int nbr_tid = (tid + 1) % ThreadManager::num_images();

  Grid3DDeviceCoD a("a"), b("b");
  a.init(64, 64, 64, 0, 0, 0);
  b.init(64, 64, 64, 0, 0, 0);

  Grid3DHostD ha, hb;
  ha.init_congruent(a);
  hb.init_congruent(b);

  int i,j,k;
  for (i=0; i < 64; i++)
    for (j=0; j < 64; j++)
      for (k=0; k < 64; k++) {
        ha.at(i,j,k) = sin((double)i)*cos((double)j)+sin(k/100.0);
      }

  a.copy_all_data(ha);


  int hdl = CoArrayManager::barrier_allocate(b.co(nbr_tid)->region()()(), a.co(tid)->region()()());
  CoArrayManager::barrier_exchange(hdl);
  CoArrayManager::barrier_exchange_fence();

  hb.copy_all_data(b);

  // compare hb to ha
  for (i=0; i < 64; i++)
    for (j=0; j < 64; j++)
      for (k=0; k < 64; k++) {
        //UNITTEST_ASSERT_EQUAL_DOUBLE(hb.at(i,j,k), ha.at(i,j,k), 0);

        doubleff dff1, dff2;
        dff1.d = hb.at(i,j,k);
        dff2.d = ha.at(i,j,k);

        if (dff1.d != dff2.d) {
          printf("%X %X\n%X %X\n\n", dff1.f[0], dff1.f[1], dff2.f[0], dff2.f[1]);
        }

      }


}

DECLARE_UNITTEST_DOUBLE_END(DoubleTransferTest);

