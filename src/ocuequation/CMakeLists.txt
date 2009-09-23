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


CUDA_ADD_LIBRARY(ocuequation 
  solver.h
  solver.cu
  equation.h
  equation.cpp
  error_handler.h
  parameters.h
  eqn_diffusion1d.h
  eqn_diffusion1d.cpp
  eqn_incompressns3d.h
  eqn_incompressns3d.cpp
  eqn_incompressns3d.cu
  eqn_scalaradvection3d.h
  eqn_scalaradvection3d.cpp
  eqn_diffusion3d.h
  eqn_diffusion3d.cpp
  sol_laplaciancent1d.h
  sol_laplaciancent1ddev.cu
  sol_laplaciancent1dhost.cpp
  sol_passiveadvection3d.h
  sol_passiveadvection3ddev.cu
  sol_selfadvection3d.h
  sol_selfadvection3ddev.cu
  sol_mgpressure3d.h
  sol_mgpressuremixed3d.h
  sol_mgpressure3d.cpp
  sol_mgpressuremixed3ddevd.cpp
  sol_mgpressure3ddev.cu
  sol_mgpressure3ddevd.cu
  sol_mgpressure3ddevf.cu
  sol_project3d.h
  sol_projectmixed3d.h
  sol_project3ddev.cu
  sol_projectmixed3d.cu
  sol_project3d.cpp
  sol_laplaciancent3d.h
  sol_laplaciancent3ddev.cu
  sol_divergence3d.h
  sol_divergence3ddev.cu
  sol_gradient3d.h
  sol_gradient3ddev.cu
  ) 


