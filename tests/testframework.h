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

#ifndef __OCU_TESTS_TEST_FRAMEWORK_H__
#define __OCU_TESTS_TEST_FRAMEWORK_H__

#include <cstdio>
#include <string>
#include <vector>
#include "ocuutil/defines.h"
#include "ocuutil/timer.h"

class UnitTestFailedException {
};

class UnitTest {
protected:

  std::string _name;
  bool        _forge_ahead;
  bool        _failed;
  ocu::CPUTimer _timer;
  bool        _multi;

  UnitTest(const char *name, bool multigpu);

  void assert_equal_double(double, double, double tol, const char *filename, int lineno);
  void assert_equal_float(float, float, float tol, const char *filename, int lineno);
  void assert_equal_int(int, int, const char *filename, int lineno);
  void assert_finite(double, const char *filename, int lineno);
  void assert_true(bool, const char *filename, int lineno);

  void set_forge_ahead(bool onoff) { _forge_ahead = onoff; }
  void set_failed() { _failed = true; }

  float elapsed_ms() { ocu::CPUTimer temp = _timer; temp.stop(); return temp.elapsed_ms(); }

  virtual void run() = 0;

public:

  bool is_multi() const { return _multi; }
  bool failed() const { return _failed; }
  const char *name() const { return _name.c_str(); }

  void start_test() {
    _timer.start();
    // need to forge ahead so that all threads will participate in all barriers.
    if (is_multi()) set_forge_ahead(true);
    run();
  }
};


#define UNITTEST_ASSERT_EQUAL_DOUBLE(a,b,t) this->assert_equal_double((a),(b),(t),__FILE__, __LINE__)
#define UNITTEST_ASSERT_EQUAL_FLOAT(a,b,t)  this->assert_equal_float((a),(b),(t),__FILE__, __LINE__)
#define UNITTEST_ASSERT_EQUAL_INT(a,b)      this->assert_equal_int((a),(b),__FILE__, __LINE__)
#define UNITTEST_ASSERT_FINITE(a)           this->assert_finite((a),__FILE__, __LINE__)
#define UNITTEST_ASSERT_TRUE(a)             this->assert_true((a),__FILE__, __LINE__)

#define DECLARE_UNITTEST_BEGIN(TEST) \
  template<typename DUMMY_TYPE> \
  class TEST : public UnitTest { \
  public: \
   TEST() : UnitTest(#TEST, false) { } \
   int dummy_so_semi_colon_will_be_parsed


#define DECLARE_UNITTEST_DOUBLE_BEGIN(TEST) \
  template<typename DUMMY_TYPE> \
  class TEST : public UnitTest { \
  public: \
   TEST() : UnitTest(#TEST, false) { } \
   int dummy_so_semi_colon_will_be_parsed


#define DECLARE_UNITTEST_END(TEST) \
  }; \
  TEST<int> TEST##_instance

#ifdef OCU_DOUBLESUPPORT
#define DECLARE_UNITTEST_DOUBLE_END(TEST) \
  }; \
  TEST<int> TEST##_instance
#else
#define DECLARE_UNITTEST_DOUBLE_END(TEST) \
  }; \
  int TEST##_instance
#endif

// Multi GPU tests

#define DECLARE_UNITTEST_MULTIGPU_BEGIN(TEST) \
  template<typename DUMMY_TYPE> \
  class TEST : public UnitTest { \
  public: \
   TEST() : UnitTest(#TEST, true) { } \
   int dummy_so_semi_colon_will_be_parsed


#define DECLARE_UNITTEST_MULTIGPU_DOUBLE_BEGIN(TEST) \
  template<typename DUMMY_TYPE> \
  class TEST : public UnitTest { \
  public: \
   TEST() : UnitTest(#TEST, true) { } \
   int dummy_so_semi_colon_will_be_parsed




class UnitTestDriver {
  std::vector<UnitTest *> _test_list;
  bool _multi_mode;

  bool run_tests(const std::vector<UnitTest *> &tests);

public:

  UnitTestDriver() : _multi_mode(false) {}

  void set_multi(bool m) { _multi_mode = m; }

  void register_test(UnitTest *);

  bool run_all_tests();
  bool run_single_gpu_tests();
  bool run_tests(const std::vector<std::string> &tests);
  void print_tests();

  static UnitTestDriver &s_driver();
};

#endif

