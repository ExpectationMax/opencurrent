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

#include <cuda.h>
#include <cstdio>
#include "ocuutil/kernel_wrapper.h"
#include "ocustorage/grid3dsample.h"



template<typename T>
__device__ float trilinear_lookup(const T *phi, int xstride, int ystride, float x, float y, float z)
{
  // assumes x,y,z are in bounds.  
  int gi = (int)floorf(x);
  int gj = (int)floorf(y);
  int gk = (int)floorf(z);

  T sx = x - gi;
  T sy = y - gj;
  T sz = z - gk;

  // is this needed?
  //sx = saturate(sx);
  //sy = saturate(sy);
  //sz = saturate(sz);

  float val = (((phi[gi*xstride +   gj  *ystride +   gk  ]) * (1-sx) + (phi[(gi+1)*xstride +   gj  *ystride +   gk  ]) * (sx)) * (1-sy) +
               ((phi[gi*xstride + (gj+1)*ystride +   gk  ]) * (1-sx) + (phi[(gi+1)*xstride + (gj+1)*ystride +   gk  ]) * (sx)) * (  sy)) * (1-sz) +
              (((phi[gi*xstride +   gj  *ystride + (gk+1)]) * (1-sx) + (phi[(gi+1)*xstride +   gj  *ystride + (gk+1)]) * (sx)) * (1-sy) +
               ((phi[gi*xstride + (gj+1)*ystride + (gk+1)]) * (1-sx) + (phi[(gi+1)*xstride + (gj+1)*ystride + (gk+1)]) * (sx)) * (  sy)) * (  sz);

  return val;
}


__device__ void apply_boundary_conditions(float &x, float &value_offset, float valid_nx, float period_nx, const ocu::BoundaryCondition &xpos, const ocu::BoundaryCondition &xneg, float inv_hx)
{
  if (x < 0) {
    // apply xneg boundary condition

    if (xneg.type == ocu::BC_PERIODIC) {
      // periodic = fmod on the position
      x = fmodf(x, period_nx);
      if (x < 0) x += period_nx;
        
    }
    else if (x < -.5f) {// BC_SCALAR_SLIP or BC_DIRICHELET or BC_NEUMANN
      if (xneg.type == ocu::BC_NEUMANN) {
        // correction for extrapolation (if non-zero Neumann)
        value_offset += (-.5f - x) * xneg.value / inv_hx; 
      }
      // interpolated value at positional_offset will be correct
      x = -.5f;
    }
  }
  else if (x > period_nx) {
    // apply xpos boundary condition

    if (xpos.type == ocu::BC_PERIODIC) {
      // periodic = fmod on the position
      x = fmodf(x, period_nx);
    }
    else if (x > valid_nx-.5f) { // BC_SCALAR_SLIP or BC_DIRICHELET or BC_NEUMANN
      if (xpos.type == ocu::BC_NEUMANN) {
        // correction for extrapolation (if non-zero Neumann)
        value_offset += (x - (valid_nx-.5f)) * xpos.value / inv_hx; 
      }
      x = valid_nx-.5f;
    }
  }
}

template<typename T>
__global__ void kernel_sample_points_3d(T *phi_sampled, const float *position_x, const float *position_y, const float *position_z, int nphi,
                                        ocu::BoundaryConditionSet bc,
                                        const T *phi, int xstride, int ystride,
                                        float valid_nx, float valid_ny, float valid_nz,
                                        float period_nx, float period_ny, float period_nz,
                                        float inv_hx, float inv_hy, float inv_hz,
                                        float shiftx, float shifty, float shiftz)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= nphi)
    return;

  float x = position_x[idx];
  float y = position_y[idx];
  float z = position_z[idx];

  x *= inv_hx;
  y *= inv_hy;
  z *= inv_hz;

  x -= shiftx;
  y -= shifty;
  z -= shiftz;

  float value_offset = 0;

  apply_boundary_conditions(x, value_offset, valid_nx, period_nx, bc.xpos, bc.xneg, inv_hx);
  apply_boundary_conditions(y, value_offset, valid_ny, period_ny, bc.ypos, bc.yneg, inv_hy);
  apply_boundary_conditions(z, value_offset, valid_nz, period_nz, bc.zpos, bc.zneg, inv_hz);

  // point sample
  phi_sampled[idx] = trilinear_lookup(phi, xstride, ystride, x, y, z) + value_offset;
}


namespace ocu {

template<typename T>
bool
sample_points_3d(
  Grid1DDevice<T> &phi_sampled,
  const Grid1DDevice<float> &position_x, const Grid1DDevice<float> &position_y, const Grid1DDevice<float> &position_z,
  const Grid3DDevice<T> &phi,
  const BoundaryConditionSet &bc,
  float period_nx, float period_ny, float period_nz,
  float hx, float hy, float hz, 
  float shiftx, float shifty, float shiftz)
{
  if (!bc.check_type(BC_PERIODIC, BC_NEUMANN, BC_DIRICHELET, BC_SCALAR_SLIP)) {
    printf("[ERROR] sample_points_3d - invalid boundary condition types %d, %d, %d, %d, %d,%d\n", bc.xpos.type, bc.xneg.type, bc.ypos.type, bc.yneg.type, bc.zpos.type, bc.zneg.type);
    return false;
  }

  int nphi = phi_sampled.nx();
  if (position_x.nx() != nphi || position_y.nx() != nphi || position_y.nx() != nphi) {
    printf("[ERROR] sample_points_3d - position arrays do not have matching sizes\n");
    return false;
  }

  if (period_nx > phi.nx() || period_ny > phi.ny() || period_nz > phi.nz()) {
    printf("[ERROR] sample_points_3d - periods cannot be greater than the array sizes (%f %f %f > %d %d %d)\n", period_nx, period_ny, period_nz, phi.nx(), phi.ny(), phi.nz());
    return false;
  }

  dim3 Dg((nphi+127)/128);
  dim3 Db(128);

  KernelWrapper wrapper;
  wrapper.PreKernel();
  kernel_sample_points_3d<<<Dg, Db>>>(&phi_sampled.at(0), &position_x.at(0), &position_y.at(0), &position_z.at(0), nphi, bc,
                                      &phi.at(0,0,0), phi.xstride(), phi.ystride(), 
                                      (float)phi.nx(), (float)phi.ny(), (float)phi.nz(),
                                      period_nx, period_ny, period_nz,
                                      1.0f/hx, 1.0f/hy, 1.0f/hz, 
                                      shiftx, shifty, shiftz);
  return wrapper.PostKernel("kernel_sample_points_3d");  
}

template<typename T>
bool
sample_points_mac_grid_3d(
  Grid1DDevice<T> &vx_sampled, Grid1DDevice<T> &vy_sampled, Grid1DDevice<T> &vz_sampled,
  const Grid1DDevice<float> &position_x, const Grid1DDevice<float> &position_y, const Grid1DDevice<float> &position_z,
  const Grid3DDevice<T> &u, const Grid3DDevice<T> &v, const Grid3DDevice<T> &w,
  const BoundaryConditionSet &bc,
  float hx, float hy, float hz)
{
  if (!bc.check_type(BC_PERIODIC, BC_FORCED_INFLOW_VARIABLE_SLIP)) {
    printf("[ERROR] sample_points_mac_grid_3d - invalid boundary condition types %d, %d, %d, %d, %d,%d\n", bc.xpos.type, bc.xneg.type, bc.ypos.type, bc.yneg.type, bc.zpos.type, bc.zneg.type);
    return false;
  }

  int nx = u.nx()-1;
  int ny = v.ny()-1;
  int nz = w.nz()-1;
  if (v.nx() != nx || w.nx() != nx || u.ny() != ny || w.ny() != ny || u.nz() != nz || v.nz() != nz) {
    printf("[ERROR] sample_points_mac_grid_3d - dimension mismatch between u,v,w\n");
    return false;
  }

  BoundaryConditionSet u_bc, v_bc, w_bc;
  // translate from mac boundary conditions to scalar field boundary conditions...
  
  if (bc.xpos.type == BC_PERIODIC) {
    u_bc.xpos.type = BC_PERIODIC;
    v_bc.xpos.type = BC_PERIODIC;
    w_bc.xpos.type = BC_PERIODIC;
  }
  else if (bc.xpos.type == BC_FORCED_INFLOW_VARIABLE_SLIP) {
    u_bc.xpos.type = BC_DIRICHELET; // not really dirichelet, though, right?  except that dirichelet rules will work...
    v_bc.xpos.type = BC_SCALAR_SLIP;
    w_bc.xpos.type = BC_SCALAR_SLIP;
  }
  
  if (bc.xneg.type == BC_PERIODIC) {
    u_bc.xneg.type = BC_PERIODIC;
    v_bc.xneg.type = BC_PERIODIC;
    w_bc.xneg.type = BC_PERIODIC;
  }
  else if (bc.xpos.type == BC_FORCED_INFLOW_VARIABLE_SLIP) {
    u_bc.xneg.type = BC_DIRICHELET; // not really dirichelet, though, right?  except that dirichelet rules will work...
    v_bc.xneg.type = BC_SCALAR_SLIP;
    w_bc.xneg.type = BC_SCALAR_SLIP;
  }

  if (bc.ypos.type == BC_PERIODIC) {
    u_bc.ypos.type = BC_PERIODIC;
    v_bc.ypos.type = BC_PERIODIC;
    w_bc.ypos.type = BC_PERIODIC;
  }
  else if (bc.ypos.type == BC_FORCED_INFLOW_VARIABLE_SLIP) {
    u_bc.ypos.type = BC_DIRICHELET; // not really dirichelet, though, right?  except that dirichelet rules will work...
    v_bc.ypos.type = BC_SCALAR_SLIP;
    w_bc.ypos.type = BC_SCALAR_SLIP;
  }
  
  if (bc.yneg.type == BC_PERIODIC) {
    u_bc.yneg.type = BC_PERIODIC;
    v_bc.yneg.type = BC_PERIODIC;
    w_bc.yneg.type = BC_PERIODIC;
  }
  else if (bc.ypos.type == BC_FORCED_INFLOW_VARIABLE_SLIP) {
    u_bc.yneg.type = BC_DIRICHELET; // not really dirichelet, though, right?  except that dirichelet rules will work...
    v_bc.yneg.type = BC_SCALAR_SLIP;
    w_bc.yneg.type = BC_SCALAR_SLIP;
  }

  if (bc.zpos.type == BC_PERIODIC) {
    u_bc.zpos.type = BC_PERIODIC;
    v_bc.zpos.type = BC_PERIODIC;
    w_bc.zpos.type = BC_PERIODIC;
  }
  else if (bc.zpos.type == BC_FORCED_INFLOW_VARIABLE_SLIP) {
    u_bc.zpos.type = BC_DIRICHELET; // not really dirichelet, though, right?  except that dirichelet rules will work...
    v_bc.zpos.type = BC_SCALAR_SLIP;
    w_bc.zpos.type = BC_SCALAR_SLIP;
  }
  
  if (bc.zneg.type == BC_PERIODIC) {
    u_bc.zneg.type = BC_PERIODIC;
    v_bc.zneg.type = BC_PERIODIC;
    w_bc.zneg.type = BC_PERIODIC;
  }
  else if (bc.zpos.type == BC_FORCED_INFLOW_VARIABLE_SLIP) {
    u_bc.zneg.type = BC_DIRICHELET; // not really dirichelet, though, right?  except that dirichelet rules will work...
    v_bc.zneg.type = BC_SCALAR_SLIP;
    w_bc.zneg.type = BC_SCALAR_SLIP;
  }

  if (!sample_points_3d(vx_sampled, position_x, position_y, position_z, u, u_bc, (float)nx, (float)ny, (float)nz, hx, hy, hz, 0,.5f,.5f)) {
    printf("[ERROR] sample_points_mac_grid_3d - failed on u sampling\n");
    return false;
  }

  if (!sample_points_3d(vy_sampled, position_x, position_y, position_z, v, v_bc, (float)nx, (float)ny, (float)nz, hx, hy, hz, .5f,0,.5f)) {
    printf("[ERROR] sample_points_mac_grid_3d - failed on v sampling\n");
    return false;
  }

  if (!sample_points_3d(vz_sampled, position_x, position_y, position_z, w, w_bc, (float)nx, (float)ny, (float)nz, hx, hy, hz, .5f,.5f,0)) {
    printf("[ERROR] sample_points_mac_grid_3d - failed on w sampling\n");
    return false;
  }

  return true;
}




template bool sample_points_3d(Grid1DDevice<float> &phi_sampled,
  const Grid1DDevice<float> &position_x, const Grid1DDevice<float> &position_y, const Grid1DDevice<float> &position_z,
  const Grid3DDevice<float> &phi,
  const BoundaryConditionSet &bc,
  float period_nx, float period_ny, float period_nz, 
  float hx, float hy, float hz, 
  float shiftx, float shifty, float shiftz);

template bool sample_points_mac_grid_3d(Grid1DDevice<float> &vx_sampled, Grid1DDevice<float> &vy_sampled, Grid1DDevice<float> &vz_sampled,
  const Grid1DDevice<float> &position_x, const Grid1DDevice<float> &position_y, const Grid1DDevice<float> &position_z,
  const Grid3DDevice<float> &u, const Grid3DDevice<float> &v, const Grid3DDevice<float> &w,
  const BoundaryConditionSet &bc,
  float hx, float hy, float hz);

#ifdef OCU_DOUBLESUPPORT

template bool sample_points_3d(Grid1DDevice<double> &phi_sampled,
  const Grid1DDevice<float> &position_x, const Grid1DDevice<float> &position_y, const Grid1DDevice<float> &position_z,
  const Grid3DDevice<double> &phi,
  const BoundaryConditionSet &bc,
  float period_nx, float period_ny, float period_nz, 
  float hx, float hy, float hz, 
  float shiftx, float shifty, float shiftz);

template bool sample_points_mac_grid_3d(Grid1DDevice<double> &vx_sampled, Grid1DDevice<double> &vy_sampled, Grid1DDevice<double> &vz_sampled,
  const Grid1DDevice<float> &position_x, const Grid1DDevice<float> &position_y, const Grid1DDevice<float> &position_z,
  const Grid3DDevice<double> &u, const Grid3DDevice<double> &v, const Grid3DDevice<double> &w,
  const BoundaryConditionSet &bc,
  float hx, float hy, float hz);

#endif // OCU_DOUBLESUPPORT

} // end namespace 

