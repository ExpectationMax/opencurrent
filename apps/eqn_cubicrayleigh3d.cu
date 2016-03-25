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

#ifdef OCU_DOUBLESUPPORT

#include "eqn_cubicrayleigh3d.h"
#include "ocuutil/kernel_wrapper.h"
#include "ocuutil/thread.h"

__global__ void kernel_apply_thermal_boundary_conditions(
    double *phi, 
    ocu::DirectionType vertical,
    double hx, double hy, double hz, int xstride, int ystride, int nx, int ny, int nz)
{
  // boundaries on faces that are not vertical.
  int X = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  int Y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
  double enforce_val=0;

  // boundaries in +-z direction on the xy face
  if ((!(vertical & ocu::DIR_ZAXIS_FLAG)) && (X < nx && Y < ny)) {
      int i=X;
      int j=Y;

      if (vertical == ocu::DIR_XPOS)
        enforce_val = 1.0f - ((i+.5f)*hx / (nx * hx));
      else if (vertical == ocu::DIR_XNEG)
        enforce_val = (i+.5f)*hx / (nx * hx);
      else if (vertical == ocu::DIR_YPOS)
        enforce_val = 1.0f - ((j+.5f)*hy / (ny * hy));
      else // if (vertical == DIR_YNEG)
        enforce_val = (j+.5f)*hy / (ny * hy);

      phi[i * xstride + j * ystride + -1] =  -phi[i * xstride + j * ystride + 0     ] + 2 * enforce_val;
      phi[i * xstride + j * ystride + nz] =  -phi[i * xstride + j * ystride + (nz-1)] + 2 * enforce_val;
  }

  // boundaries in +-y direction in the xz face 
  if ((!(vertical & ocu::DIR_YAXIS_FLAG)) && (X < nx && Y < nz)) {
      int i=X;
      int k=Y;

      if (vertical == ocu::DIR_XPOS)
        enforce_val = 1.0f - ((i+.5f)*hx / (nx * hx));
      else if (vertical == ocu::DIR_XNEG)
        enforce_val = (i+.5f)*hx / (nx * hx);
      else if (vertical == ocu::DIR_ZPOS)
        enforce_val = 1.0f - ((k+.5f)*hz / (nz * hz));
      else // if (vertical == DIR_ZNEG)
        enforce_val = (k+.5f)*hz / (nz * hz);

      phi[i * xstride + -1 * ystride + k] =  -phi[i * xstride + 0      * ystride + k] + 2 * enforce_val;
      phi[i * xstride + ny * ystride + k] =  -phi[i * xstride + (ny-1) * ystride + k] + 2 * enforce_val;
  }

  // boundaries on the +-x direction in the yz face
  if ((!(vertical & ocu::DIR_XAXIS_FLAG)) && (X < ny && Y < nz)) {
      int j=X;
      int k=Y;

      if (vertical == ocu::DIR_YPOS)
        enforce_val = 1.0f - ((j+.5f)*hy / (ny * hy));
      else if (vertical == ocu::DIR_YNEG)
        enforce_val = (j+.5f)*hy / (ny * hy);
      else if (vertical == ocu::DIR_ZPOS)
        enforce_val = 1.0f - ((k+.5f)*hz / (nz * hz));
      else // if (vertical == DIR_ZNEG)
        enforce_val = (k+.5f)*hz / (nz * hz);

      phi[-1 * xstride + j * ystride + k] =  -phi[0      * xstride + j * ystride + k] + 2 * enforce_val;
      phi[nx * xstride + j * ystride + k] =  -phi[(nx-1) * xstride + j * ystride + k] + 2 * enforce_val;
  }

}


namespace ocu {

bool Eqn_CubicRayleigh3DD::enforce_thermal_boundary_conditions()
{
  int max_nxny = nx() > ny() ? nx() : ny();
  int max_nynz = ny() > nz() ? ny() : nz();

  dim3 Dg((max_nxny+15) / 16, (max_nynz+15) / 16);
  dim3 Db(16, 16);
  
  KernelWrapper wrapper;
  wrapper.PreKernel();

  kernel_apply_thermal_boundary_conditions<<<Dg, Db, 0, ThreadManager::get_compute_stream()>>>(&_temp.at(0,0,0), 
    _vertical_direction,
    hx(), hy(), hz(), _temp.xstride(), _temp.ystride(), nx(), ny(), nz());

  return wrapper.PostKernel("kernel_apply_thermal_boundary_conditions", nz());
}

bool Eqn_CubicRayleigh3DD::set_parameters(const Eqn_CubicRayleigh3DParamsD &params)
{
  check_ok(Eqn_IncompressibleNS3D<double>::set_parameters(params));
  check_ok(enforce_thermal_boundary_conditions());

  return !any_error();
}

bool Eqn_CubicRayleigh3DD::advance_one_step(double dt)
{
  clear_error();

  check_ok(Eqn_IncompressibleNS3D<double>::advance_one_step(dt));
  check_ok(enforce_thermal_boundary_conditions());

  // enforce thermal bc

  return !any_error();
}


}

#else // ! OCU_DOUBLESUPPORT

#endif //OCU_DOUBLESUPPORT
