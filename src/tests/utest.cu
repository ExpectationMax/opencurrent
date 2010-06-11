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

#include "ocustorage/coarray.h"
#include "tests/testframework.h"

void do_error() {

#ifdef OCU_OMP
  const char *OMP = "";
#else
  const char *OMP = "[DISABLED] ";
#endif

  printf("utest [option] [test1] [test2] ...\n");
  printf("Options are:\n");
  printf(" -gpu N     Run on the numbered GPU.  Can also set via env var OCU_UTEST_GPU.  Default value is 0.\n");
  printf(" -help      Print this message\n");
  printf(" -multi     %sRun in multigpu mode.  Only multi-gpu-enabled tests will run.\n", OMP );
  printf(" -numgpus N %sSet GPU count for multi gpu mode.  Can also set via env var OCU_UTEST_MULTI.  Default value is 2.\n", OMP);
  printf(" -repeat N  Repeat all tests N times\n");
  printf("\n");
  printf("Current tests are:\n");
  UnitTestDriver::s_driver().print_tests();

  exit(-1);
}


int main(int argc, char **argv)
{
  int dev_cnt;
  cudaGetDeviceCount(&dev_cnt);

  int gpu = getenv("OCU_UTEST_GPU") ? atoi(getenv("OCU_UTEST_GPU")) : 0;
  int num_gpus = getenv("OCU_UTEST_MULTI") ? atoi(getenv("OCU_UTEST_MULTI")) : dev_cnt;
  bool do_multi = false;

  int unprocessed_args = argc-1;
  int cur_arg = 1;
  int repeat = 1;

  while(cur_arg < argc && argv[cur_arg][0] == '-') {

    if (strcmp(argv[cur_arg], "-gpu")==0) {
      cur_arg++;
      unprocessed_args--;

      if (cur_arg < argc) {
        gpu = atoi(argv[cur_arg]);
      }
      else do_error();
    }

    if (strcmp(argv[cur_arg], "-numgpus")==0) {
#ifndef OCU_OMP
      printf("[ERROR] -numgpus option invalid when compiled with OCU_OMP_ENABLED FALSE");
      do_error();
#else
      cur_arg++;
      unprocessed_args--;

      if (cur_arg < argc) {
        num_gpus = atoi(argv[cur_arg]);
      }
      else do_error();
#endif
    }

    if (strcmp(argv[cur_arg], "-repeat")==0) {
      cur_arg++;
      unprocessed_args--;

      if (cur_arg < argc) {
        repeat = atoi(argv[cur_arg]);
      }
      else do_error();
    }

    if (strcmp(argv[cur_arg], "-multi")==0) {
#ifndef OCU_OMP
      printf("[ERROR] -multi option invalid when compiled with OCU_OMP_ENABLED FALSE");
      do_error();
#else
      do_multi = true;
#endif
    }

    if (strcmp(argv[cur_arg], "-help")==0) {
      do_error();
    }

    cur_arg++;
    unprocessed_args--;
  }

  UnitTestDriver::s_driver().set_multi(do_multi);

  if (do_multi) {

#ifndef OCU_OMP
    printf("[ERROR] Cannot run in multi mode when compiled with OCU_OMP_ENABLED FALSE\n");
#else

    // start n threads, init all multithreading stuff, etc.
    printf("[INFO] Running in multi-GPU mode with %d devices\n", num_gpus);

    if (!ocu::CoArrayManager::initialize(num_gpus)) {
      printf("[ERROR] Could not initialize CoArrayManager\n");
      exit(-1);
    }

    if (!ocu::ThreadManager::initialize(num_gpus)) {
      printf("[ERROR] Could not initialize ThreadManager\n");
      exit(-1);
    }

#pragma omp parallel
    {
      if (!ocu::ThreadManager::initialize_image(ocu::ThreadManager::this_image())) {
        printf("[ERROR] Could not initialize ThreadManager image %d\n", ocu::ThreadManager::this_image());
        exit(-1);
      }

      if (!ocu::CoArrayManager::initialize_image(ocu::ThreadManager::this_image())) {
        printf("[ERROR] Could not initialize CoArrayManager image %d\n", ocu::ThreadManager::this_image());
        exit(-1);
      }


      if (unprocessed_args == 0) {
        for (int r=0; r < repeat; r++) {
          if (!UnitTestDriver::s_driver().run_all_tests())
            exit(-1);
        }
      }
      else {
        std::vector<std::string> tests;
        for (int i=argc-unprocessed_args; i < argc; i++)
          tests.push_back(argv[i]);

        for (int r=0; r < repeat; r++) {
          if (!UnitTestDriver::s_driver().run_tests(tests))
            exit(-1);
        }
      }
    }
#endif
  }
  else {

    if (!ocu::ThreadManager::initialize(1)) {
      printf("[ERROR] Could not initialize ThreadManager\n");
      exit(-1);
    }

    if (!ocu::ThreadManager::initialize_image(gpu)) {
      printf("[ERROR] Could not initialize ThreadManager on gpu %d\n", gpu);
      exit(-1);
    }

    printf("[INFO] Running on GPU %d\n", gpu);

    // call this once to force everything to initialize so any timing results are not skewed
    cudaFree(0);

    if (unprocessed_args == 0) {
      for (int r=0; r < repeat; r++) {
        if (!UnitTestDriver::s_driver().run_single_gpu_tests())
          exit(-1);
      }
    }
    else {
      std::vector<std::string> tests;
      for (int i=argc-unprocessed_args; i < argc; i++)
        tests.push_back(argv[i]);

      for (int r=0; r < repeat; r++) {
        if (!UnitTestDriver::s_driver().run_tests(tests))
          exit(-1);
      }
    }
  }

  return 0;
}

