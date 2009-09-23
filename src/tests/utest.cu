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

void do_error() {

  printf("utext [option] [test1] [test2] ...\n");
  printf("Options are:\n");
  printf(" -gpu N    Run on the numbered GPU.  Can also set via env var FLUID_UTEST_GPU\n");
  printf(" -help     Print this message\n");
  printf("\n");
  printf("Current tests are:\n");
  UnitTestDriver::s_driver().print_tests();

  exit(-1);
}


int main(int argc, char **argv)
{
  int gpu = getenv("FLUIDSIM_UTEST_GPU") ? atoi(getenv("FLUIDSIM_UTEST_GPU")) : 0;

  int unprocessed_args = argc-1;
  int cur_arg = 1;

  while(cur_arg < argc && argv[cur_arg][0] == '-') {

    if (strcmp(argv[cur_arg], "-gpu")==0) {
      cur_arg++;
      unprocessed_args--;

      if (cur_arg < argc) {
        gpu = atoi(argv[cur_arg]);
      }
      else do_error();
    }

    if (strcmp(argv[cur_arg], "-help")==0) {
      do_error();
    }

    cur_arg++;
    unprocessed_args--;
  }


  cudaError_t er = cudaSetDevice(gpu);
  if (er != (unsigned int)CUDA_SUCCESS) {
    printf("[ERROR] cudaSetDevice failed with \"%s\"\n", cudaGetErrorString(er));
    exit(-1);
  }
  printf("[INFO] Running on GPU %d\n", gpu);

  // call this once to force everything to initialize so any timing results are not skewed
  cudaFree(0);

  if (unprocessed_args == 0) {
    if (!UnitTestDriver::s_driver().run_all_tests())
      exit(-1);
  }
  else {
    std::vector<std::string> tests;
    for (int i=argc-unprocessed_args; i < argc; i++)
      tests.push_back(argv[i]);

    if (!UnitTestDriver::s_driver().run_tests(tests))
      exit(-1);
  }

  return 0;
}

