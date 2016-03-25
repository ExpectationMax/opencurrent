#  Copyright 2008-2009 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License


SET(OCU_REQUIRED_LIBS "")

IF (OCU_NETCDF_ENABLED)
  FIND_PACKAGE(NetCDF)
  IF (OCU_REQUIRED_LIBS STREQUAL "")
    SET(OCU_REQUIRED_LIBS "${NetCDF_LIBRARY}")
  ELSE ()
    SET(OCU_REQUIRED_LIBS "${OCU_REQUIRED_LIBS} ${NetCDF_LIBRARY}")
  ENDIF ()
ENDIF (OCU_NETCDF_ENABLED)


LINK_DIRECTORIES(${CUDAPARSE_BINARY_DIR}/ocuequation 
	${CUDAPARSE_BINARY_DIR}/ocustorage
	${CUDAPARSE_BINARY_DIR}/ocuutil)

CUDA_ADD_EXECUTABLE(utest 
	utest.cu
	testframework.cpp 
	testframework.h
	diffusion1dtest.cpp
	diffusion3dtest.cpp
	netcdftest.cpp
	grid3dtest.cpp
	grid1dtest.cpp
	samplingtest.cpp
	advection3dtest.cpp
	advection3ddoubletest.cpp
	mgtest.cpp
	projecttest.cpp
	projectdoubletest.cpp
	mgdoubletest.cpp
	mgmixedtest.cpp
	nstest.cpp
	lockextest.cpp
	pcgtest.cpp
	rayleigh.cpp
	rayleighnoslip.cpp
	rayleightiming.cpp
	diffusion1dmultitest.cu
	diffusion3dmultitest.cu
	coarraytest.cu
	mgmultitest.cu)

ADD_TEST(Diffusion1DTest ${EXECUTABLE_OUTPUT_PATH}/utest Diffusion1DTest)
ADD_TEST(Diffusion3DTest ${EXECUTABLE_OUTPUT_PATH}/utest Diffusion3DTest)
ADD_TEST(Advection3DTest ${EXECUTABLE_OUTPUT_PATH}/utest Advection3DTest Advection3DSymmetryTest)
ADD_TEST(Grid1DTest ${EXECUTABLE_OUTPUT_PATH}/utest Grid1DReduceTest Reduce1DTimingTest)
ADD_TEST(Grid3DTest ${EXECUTABLE_OUTPUT_PATH}/utest Grid3DTest Grid3DReduceTest Reduce3DTimingTest)
ADD_TEST(MultigridTest ${EXECUTABLE_OUTPUT_PATH}/utest MultigridTest)
ADD_TEST(ProjectTest ${EXECUTABLE_OUTPUT_PATH}/utest ProjectTest)
ADD_TEST(Advection3DDoubleTest ${EXECUTABLE_OUTPUT_PATH}/utest Advection3DDoubleTest Advection3DDoubleSymmetryTest Advection3DDoubleSwirlTest)
ADD_TEST(Grid1DDoubleTest ${EXECUTABLE_OUTPUT_PATH}/utest Grid1DReduceDoubleTest Reduce1DDoubleTimingTest)
ADD_TEST(Grid3DDoubleTest ${EXECUTABLE_OUTPUT_PATH}/utest Grid3DTest Grid3DReduceDoubleTest Reduce3DDoubleTimingTest)
ADD_TEST(MultigridDoubleTest ${EXECUTABLE_OUTPUT_PATH}/utest MultigridDoubleTest)
ADD_TEST(MultigridMixedTest ${EXECUTABLE_OUTPUT_PATH}/utest MultigridMixedTest)
ADD_TEST(ProjectDoubleTest ${EXECUTABLE_OUTPUT_PATH}/utest ProjectDoubleTest)
ADD_TEST(SamplingTest ${EXECUTABLE_OUTPUT_PATH}/utest SamplingTest)
ADD_TEST(NSTest ${EXECUTABLE_OUTPUT_PATH}/utest NSTest)
ADD_TEST(LockExTest ${EXECUTABLE_OUTPUT_PATH}/utest LockExTest LockExDoubleTest)
ADD_TEST(PCGTest ${EXECUTABLE_OUTPUT_PATH}/utest PCGDoubleTest)

IF (OCU_NETCDF_ENABLED)
  ADD_TEST(NetCDFTest ${EXECUTABLE_OUTPUT_PATH}/utest NetCDFTest GridNetCDFTest)
ENDIF (OCU_NETCDF_ENABLED)

IF (OCU_OMP_ENABLED)
  ADD_TEST(Diffusion1DMultiTest ${EXECUTABLE_OUTPUT_PATH}/utest -multi Diffusion1DMultiTest)
  ADD_TEST(Diffusion3DMultiTest ${EXECUTABLE_OUTPUT_PATH}/utest -multi Diffusion3DMultiTest)
  ADD_TEST(CoArray1DTest ${EXECUTABLE_OUTPUT_PATH}/utest -repeat 10 -multi CoArray1DTest)
  ADD_TEST(CoArray3DTest ${EXECUTABLE_OUTPUT_PATH}/utest -repeat 10 -multi CoArray3DTest DoubleTransferTest)
  ADD_TEST(MultiReduceTest ${EXECUTABLE_OUTPUT_PATH}/utest -multi MultiReduceTest)
  ADD_TEST(MultigridMultiTest ${EXECUTABLE_OUTPUT_PATH}/utest -multi MultigridMultiTest)
ENDIF (OCU_OMP_ENABLED)

TARGET_LINK_LIBRARIES(utest ocuequation ocustorage ocuutil ${OCU_REQUIRED_LIBS})
