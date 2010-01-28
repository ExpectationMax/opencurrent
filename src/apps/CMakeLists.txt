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


FIND_PACKAGE(NetCDF)

LINK_DIRECTORIES(${CUDAPARSE_BINARY_DIR}/ocuequation 
  ${CUDAPARSE_BINARY_DIR}/ocustorage
  ${CUDAPARSE_BINARY_DIR}/ocuutil)

ADD_EXECUTABLE(testapp testapp.cpp)
ADD_EXECUTABLE(lockex lockex.cpp)
ADD_EXECUTABLE(incompress incompress.cpp)
CUDA_ADD_EXECUTABLE(cubicrayleigh cubicrayleigh.cpp eqn_cubicrayleigh3d.cu eqn_cubicrayleigh3d.h)
ADD_EXECUTABLE(pcg pcg.cpp)

TARGET_LINK_LIBRARIES(testapp ocuequation ocustorage ocuutil ${NetCDF_LIBRARY})
TARGET_LINK_LIBRARIES(lockex ocuequation ocustorage ocuutil ${NetCDF_LIBRARY})
TARGET_LINK_LIBRARIES(incompress ocuequation ocustorage ocuutil ${NetCDF_LIBRARY})
TARGET_LINK_LIBRARIES(cubicrayleigh ocuequation ocustorage ocuutil ${NetCDF_LIBRARY})
TARGET_LINK_LIBRARIES(pcg ocuequation ocustorage ocuutil ${NetCDF_LIBRARY})