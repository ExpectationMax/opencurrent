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
#include "ocuutil/float_routines.h"
#include "ocuutil/thread.h"

UnitTest::UnitTest(const char *name, bool multi) :
  _name(name), _forge_ahead(false), _failed(false), _multi(multi)
{
  UnitTestDriver::s_driver().register_test(this);
}

void 
UnitTest::assert_equal_double(double a, double b, double tol, const char *filename, int lineno)
{
  if (fabs(a-b) <= tol) return;

  printf("[ASSERT] %s::assert_equal_double(%.10f, %.10f, %.10f) at %s line %d\n", _name.c_str(), a, b, tol, filename, lineno);
  set_failed();

  if (!_forge_ahead)
    throw UnitTestFailedException();
}

void 
UnitTest::assert_equal_float(float a, float b, float tol, const char *filename, int lineno)
{
  if (fabsf(a-b) <= tol) return;

  printf("[ASSERT] %s::assert_equal_float(%.10f, %.10f, %.10f) at %s line %d\n", _name.c_str(), a, b, tol, filename, lineno);
  set_failed();

  if (!_forge_ahead)
    throw UnitTestFailedException();
}

void 
UnitTest::assert_equal_int(int a, int b, const char *filename, int lineno)
{
  if (a == b) return;

  printf("[ASSERT] %s::assert_equal_int(%d, %d) at %s line %d\n", _name.c_str(), a, b, filename, lineno);
  set_failed();

  if (!_forge_ahead)
    throw UnitTestFailedException();
}

void 
UnitTest::assert_finite(double a, const char *filename, int lineno)
{
  if (ocu::check_float(a)) return;

  printf("[ASSERT] %s::assert_finite at %s line %d\n", _name.c_str(), filename, lineno);
  set_failed();

  if (!_forge_ahead)
    throw UnitTestFailedException();
}

void 
UnitTest::assert_true(bool a, const char *filename, int lineno)
{
  if (a) return;

  printf("[ASSERT] %s::assert_true at %s line %d\n", _name.c_str(), filename, lineno);
  set_failed();

  if (!_forge_ahead)
    throw UnitTestFailedException();
}


void UnitTestDriver::register_test(UnitTest *test)
{
  _test_list.push_back(test);
}

bool UnitTestDriver::run_tests(const std::vector<UnitTest *> &tests_to_run)
{
  bool any_failed = false;
  int i;
  printf("Running tests: ");
  for (i=0; i < tests_to_run.size(); i++) {
    if (i != 0)
      printf(", ");
    printf("%s", tests_to_run[i]->name());
  }
  printf("\n");

  for (i=0; i < tests_to_run.size(); i++) {
    if (_multi_mode) {
      ocu::ThreadManager::barrier();
    }

    bool ok = true;

    if (tests_to_run[i]->is_multi() && !_multi_mode) {
      printf("[ERROR] %s is a multi-gpu test, running in single-gpu mode\n", tests_to_run[i]->name());
      ok = false;
    }
    else {
      try {


        if (!tests_to_run[i]->is_multi()) {
          if (ocu::ThreadManager::this_image() == 0) {
            printf("running %s on thread 0\n", tests_to_run[i]->name());
            tests_to_run[i]->start_test();
          }
        }
        else {
          printf("running %s\n", tests_to_run[i]->name());
          tests_to_run[i]->start_test();
        }
      }
      catch(...) {
        ok = false;
      }
    }

    if (tests_to_run[i]->failed())
      ok = false;

    if (!ok) {
      any_failed = true;
      printf("[FAILED] %s\n", tests_to_run[i]->name());
    }
  }
  printf("\n");

  if (_multi_mode) {
    ocu::ThreadManager::barrier();
  }

  if (!any_failed) {
    printf("[PASSED]\n");
    return true;
  }
  else {
    printf("There were failures.\n");
    return false;
  }
}

void UnitTestDriver::print_tests()
{
  for (int i=0; i < _test_list.size(); i++) {
    if (i != 0)
      printf(", ");
    printf("%s (%s)", _test_list[i]->name(), _test_list[i]->is_multi() ? "multi" : "single");
  }
  printf("\n");
}



bool UnitTestDriver::run_single_gpu_tests()
{
  // build the list
  std::vector<UnitTest *> single_list;
  for (int i=0; i < _test_list.size(); i++)
    if (!_test_list[i]->is_multi())
      single_list.push_back(_test_list[i]);
  return run_tests(single_list);
}


bool UnitTestDriver::run_all_tests()
{
  return run_tests(_test_list);
}

bool 
UnitTestDriver::run_tests(const std::vector<std::string> &tests)
{
  int i, j;
  std::vector<UnitTest *> tests_to_run;

  for (j=0; j < tests.size(); j++) {

    bool found = false;
    for (i = 0; !found && i < _test_list.size(); i++)
      if (tests[j] == _test_list[i]->name()) {

        tests_to_run.push_back(_test_list[i]);
        found = true;
      }
   
    if (!found) {
      printf("[WARNING] UnitTestDriver::run_tests - test %s not found\n", tests[j].c_str());
    }
  }

  return run_tests(tests_to_run);
}

UnitTestDriver &
UnitTestDriver::s_driver()
{
  static UnitTestDriver s_instance;
  return s_instance;
}

