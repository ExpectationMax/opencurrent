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

#include <cstdio>
#include <string>

#include "ocustorage/coarray.h"
#include "ocustorage/grid3dboundary.h"
#include "ocuequation/sol_project3d.h"


namespace ocu {


template<typename T>
bool Sol_ProjectDivergence3DDeviceStorage<T>::initialize_device_storage(
  int nx, int ny, int nz, double hx, double hy, double hz, Grid3DDevice<T> *u_val, Grid3DDevice<T> *v_val, Grid3DDevice<T> *w_val)
{
  if (!initialize_base_storage(nx,ny,nz,hx,hy,hz,u_val, v_val, w_val)) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_device_storage - failed to initialize base storage\n");
    return false;
  }

  u = u_val;
  v = v_val;
  w = w_val;

  if (!divergence.init(_nx, _ny, _nz, 1, 1, 1, u->pnx() - (_nx + 2), u->pny() - (_ny + 2), u->pnz() - (_nz + 2))) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_device_storage - failed to initialize divergence\n");
    return false;
  }

  if (!u_val->check_layout_match(divergence)) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_device_storage - divergence layout mismatch\n");
    return false;
  }

  return true;
}





template<typename T>
bool Sol_ProjectDivergence3DDevice<T>::solve(double tolerance)
{
  this->clear_error();
  double residual = 0;

  check_ok(apply_3d_mac_boundary_conditions_level1( *this->u, *this->v, *this->w,  this->bc, this->_hx, this->_hy, this->_hz), "Sol_ProjectDivergence3DDevice::solve - could not enforce boundary conditions");
  check_ok(this->divergence_solver.solve(), "Sol_ProjectDivergence3DDevice::solve - could not calculate divergence");
  check_ok(this->pressure_solver.solve(residual, tolerance, 15), "Sol_ProjectDivergence3DDevice::solve - could not solve for pressure\n");
  check_ok(this->gradient_solver.solve(), "Sol_ProjectDivergence3DDevice::solve - could not subtract gradient of pressure\n");
  check_ok(apply_3d_mac_boundary_conditions_level1( *this->u, *this->v, *this->w,  this->bc, this->_hx, this->_hy, this->_hz), "Sol_ProjectDivergence3DDevice::solve - could not enforce boundary conditions\n");

  return !this->any_error();
}

template<typename T>
bool Sol_ProjectDivergence3DDevice<T>::solve_divergence_only()
{
  this->clear_error();

  check_ok(apply_3d_mac_boundary_conditions_level1( *this->u, *this->v, *this->w,  this->bc, this->_hx, this->_hy, this->_hz), "Sol_ProjectDivergence3DDevice::solve - could not enforce boundary conditions");
  check_ok(this->divergence_solver.solve(), "Sol_ProjectDivergence3DDevice::solve - could not calculate divergence");

  return !this->any_error();
}






template<typename T>
bool Sol_ProjectDivergence3DDevice<T>::initialize_storage(
  int nx, int ny, int nz, double hx, double hy, double hz, Grid3DDevice<T> *u_val, Grid3DDevice<T> *v_val, Grid3DDevice<T> *w_val)
{
  if (!initialize_base_storage(nx,ny,nz,hx,hy,hz,u_val, v_val, w_val)) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - failed to initialize base storage\n");
    return false;
  }

  if (!initialize_device_storage(nx,ny,nz,hx,hy,hz,u_val, v_val, w_val)) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - failed to initialize device storage\n");
    return false;
  }

  this->pressure_solver.bc.xneg = convert_bc_to_poisson_eqn(this->bc.xneg);
  this->pressure_solver.bc.xpos = convert_bc_to_poisson_eqn(this->bc.xpos);
  this->pressure_solver.bc.yneg = convert_bc_to_poisson_eqn(this->bc.yneg);
  this->pressure_solver.bc.ypos = convert_bc_to_poisson_eqn(this->bc.ypos);
  this->pressure_solver.bc.zneg = convert_bc_to_poisson_eqn(this->bc.zneg);
  this->pressure_solver.bc.zpos = convert_bc_to_poisson_eqn(this->bc.zpos);

  if (!divergence_solver.initialize_storage(nx, ny, nz, hx, hy, hz, this->u, this->v, this->w, &this->divergence)) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - failed to initialize divergence_solver\n");
    return false;
  }

  if (!pressure_solver.initialize_storage(nx, ny, nz, hx, hy, hz, &this->divergence)) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - failed to initialize pressure_solver\n");
    return false;
  }

  if (!gradient_solver.initialize_storage(nx, ny, nz, hx, hy, hz, this->u, this->v, this->w, &this->pressure_solver.pressure())) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - failed to initialize gradient_solver\n");
    return false;
  }
  this->gradient_solver.coefficient = -1;

  if (!u_val->check_layout_match(this->pressure_solver.pressure())) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - pressure layout mismatch\n");
    return false;
  }

  this->pressure_solver.pressure().clear_zero();
  this->pressure_solver.convergence = CONVERGENCE_LINF;

  return true;
}



template<typename T>
Sol_ProjectDivergence3DDeviceCo<T>::Sol_ProjectDivergence3DDeviceCo(const char *name) :
  pressure_solver((std::string(name) + std::string(".pressure_solver")).c_str())
{
  _u_negx_hdl = -1;
  _v_negx_hdl = -1;
  _w_negx_hdl = -1;
  _u_posx_hdl = -1;
  _v_posx_hdl = -1;
  _w_posx_hdl = -1;
}

template<typename T>
Sol_ProjectDivergence3DDeviceCo<T>::~Sol_ProjectDivergence3DDeviceCo() 
{
  CoArrayManager::barrier_deallocate(_u_negx_hdl);
  CoArrayManager::barrier_deallocate(_v_negx_hdl);
  CoArrayManager::barrier_deallocate(_w_negx_hdl);
  CoArrayManager::barrier_deallocate(_u_posx_hdl);
  CoArrayManager::barrier_deallocate(_v_posx_hdl);
  CoArrayManager::barrier_deallocate(_w_posx_hdl);
}

template<typename T>
bool Sol_ProjectDivergence3DDeviceCo<T>::initialize_storage(
  int nx, int ny, int nz, double hx, double hy, double hz, Grid3DDeviceCo<T> *u_val, Grid3DDeviceCo<T> *v_val, Grid3DDeviceCo<T> *w_val)
{
  int tid = ThreadManager::this_image();
  int num_images = ThreadManager::num_images();

  if (!initialize_base_storage(nx,ny,nz,hx,hy,hz,u_val, v_val, w_val)) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - failed to initialize base storage\n");
    return false;
  }

  if (!initialize_device_storage(nx,ny,nz,hx,hy,hz,u_val, v_val, w_val)) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - failed to initialize device storage\n");
    return false;
  }

  co_u = u_val;
  co_v = v_val;
  co_w = w_val;

  // figure out who our neighbors are
  int negx_image = (tid - 1 + num_images) % num_images;
  int posx_image = (tid + 1) % num_images;

  if (tid == 0 && this->bc.xneg.type != BC_PERIODIC)
    negx_image = -1;

  if (tid == num_images - 1 && this->bc.xpos.type != BC_PERIODIC) 
    posx_image = -1;

  // set up bc's & transfers here
  this->local_bc = this->bc;
  if (negx_image != -1)
    local_bc.xneg.type = BC_NONE;

  if (posx_image != -1) 
    local_bc.xpos.type = BC_NONE;

  // pressure solver gets the non-local bc's, since it will figure out its own local bc's
  this->pressure_solver.bc.xneg = convert_bc_to_poisson_eqn(this->bc.xneg);
  this->pressure_solver.bc.xpos = convert_bc_to_poisson_eqn(this->bc.xpos);
  this->pressure_solver.bc.yneg = convert_bc_to_poisson_eqn(this->bc.yneg);
  this->pressure_solver.bc.ypos = convert_bc_to_poisson_eqn(this->bc.ypos);
  this->pressure_solver.bc.zneg = convert_bc_to_poisson_eqn(this->bc.zneg);
  this->pressure_solver.bc.zpos = convert_bc_to_poisson_eqn(this->bc.zpos);

  if (posx_image != -1) {
    Region3D uto   = co_u                ->region(this->_nx, this->_nx+1)()();
    Region3D ufrom = co_u->co(posx_image)->region(0,1)()();

    _u_posx_hdl = CoArrayManager::barrier_allocate(uto, ufrom);
    if (_u_posx_hdl == -1)
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate _u_posx_hdl\n");

    Region3D vto   = co_v                ->region(this->_nx)()();
    Region3D vfrom = co_v->co(posx_image)->region(0)()();

    _v_posx_hdl = CoArrayManager::barrier_allocate(vto, vfrom);
    if (_v_posx_hdl == -1)
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate _v_posx_hdl\n");

    Region3D wto   = co_w                ->region(this->_nx)()();
    Region3D wfrom = co_w->co(posx_image)->region(0)()();

    _w_posx_hdl = CoArrayManager::barrier_allocate(wto, wfrom);
    if (_w_posx_hdl == -1)
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate _w_posx_hdl\n");
  }
  else {
    _u_posx_hdl = CoArrayManager::barrier_allocate();
    _v_posx_hdl = CoArrayManager::barrier_allocate();
    _w_posx_hdl = CoArrayManager::barrier_allocate();
  }  

  if (negx_image != -1) {
    Region3D uto   = co_u                ->region(-1)()();
    Region3D ufrom = co_u->co(negx_image)->region(this->_nx-1)()();

    _u_negx_hdl = CoArrayManager::barrier_allocate(uto, ufrom);
    if (_u_negx_hdl  == -1)
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate _u_negx_hdl\n");

    Region3D vto   = co_v                ->region(-1)()();
    Region3D vfrom = co_v->co(negx_image)->region(this->_nx-1)()();

    _v_negx_hdl = CoArrayManager::barrier_allocate(vto, vfrom);
    if (_v_negx_hdl == -1)
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate _v_negx_hdl\n");

    Region3D wto   = co_w                ->region(-1)()();
    Region3D wfrom = co_w->co(negx_image)->region(this->_nx-1)()();

    _w_negx_hdl = CoArrayManager::barrier_allocate(wto, wfrom);
    if (_w_negx_hdl == -1)
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate _w_negx_hdl\n");
  }
  else {
    _u_negx_hdl = CoArrayManager::barrier_allocate();
    _v_negx_hdl = CoArrayManager::barrier_allocate();
    _w_negx_hdl = CoArrayManager::barrier_allocate();
  }  

  if (!divergence_solver.initialize_storage(nx, ny, nz, hx, hy, hz, this->u, this->v, this->w, &this->divergence)) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - failed to initialize divergence_solver\n");
    return false;
  }

  if (!pressure_solver.initialize_storage(nx, ny, nz, hx, hy, hz, &this->divergence)) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - failed to initialize pressure_solver\n");
    return false;
  }

  if (!gradient_solver.initialize_storage(nx, ny, nz, hx, hy, hz, this->u, this->v, this->w, &this->pressure_solver.co_pressure())) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - failed to initialize gradient_solver\n");
    return false;
  }
  this->gradient_solver.coefficient = -1;

  if (!u_val->check_layout_match(this->pressure_solver.co_pressure())) {
    printf("[ERROR] Sol_ProjectDivergence3DDevice::initialize_storage - pressure layout mismatch\n");
    return false;
  }

  this->pressure_solver.co_pressure().clear_zero();
  this->pressure_solver.convergence = CONVERGENCE_LINF;

  return true;
}



template<typename T>
bool Sol_ProjectDivergence3DDeviceCo<T>::solve(double tolerance)
{
  this->clear_error();
  double residual = 0;

  CoArrayManager::barrier_exchange(_u_negx_hdl);
  CoArrayManager::barrier_exchange(_v_negx_hdl);
  CoArrayManager::barrier_exchange(_w_negx_hdl);
  CoArrayManager::barrier_exchange(_u_posx_hdl);
  CoArrayManager::barrier_exchange(_v_posx_hdl);
  CoArrayManager::barrier_exchange(_w_posx_hdl);
  check_ok(apply_3d_mac_boundary_conditions_level1( *this->u, *this->v, *this->w,  this->local_bc, this->_hx, this->_hy, this->_hz), "Sol_ProjectDivergence3DDevice::solve - could not enforce boundary conditions");

  ThreadManager::io_fence();
  ThreadManager::barrier();

  check_ok(this->divergence_solver.solve(), "Sol_ProjectDivergence3DDeviceCo::solve - could not calculate divergence");
  check_ok(this->pressure_solver.solve(residual, tolerance, 15), "Sol_ProjectDivergence3DDeviceCo::solve - could not solve for pressure\n");

  // NB: we do not need to exchange pressure ghost cells, since we are guaranteed that the pressure 
  // solver finishes with ghost cells valid and up-to-date.
  check_ok(this->gradient_solver.solve(), "Sol_ProjectDivergence3DDeviceCo::solve - could not subtract gradient of pressure\n");

  CoArrayManager::barrier_exchange(_u_negx_hdl);
  CoArrayManager::barrier_exchange(_v_negx_hdl);
  CoArrayManager::barrier_exchange(_w_negx_hdl);
  CoArrayManager::barrier_exchange(_u_posx_hdl);
  CoArrayManager::barrier_exchange(_v_posx_hdl);
  CoArrayManager::barrier_exchange(_w_posx_hdl);
  check_ok(apply_3d_mac_boundary_conditions_level1( *this->u, *this->v, *this->w,  this->local_bc, this->_hx, this->_hy, this->_hz), "Sol_ProjectDivergence3DDeviceCo::solve - could not enforce boundary conditions\n");

  ThreadManager::io_fence();
  ThreadManager::barrier();

  return !this->any_error();
}

template<typename T>
bool Sol_ProjectDivergence3DDeviceCo<T>::solve_divergence_only()
{
  this->clear_error();

  check_ok(apply_3d_mac_boundary_conditions_level1( *this->u, *this->v, *this->w,  this->local_bc, this->_hx, this->_hy, this->_hz), "Sol_ProjectDivergence3DDeviceCo::solve - could not enforce boundary conditions");
  check_ok(this->divergence_solver.solve(), "Sol_ProjectDivergence3DDeviceCo::solve - could not calculate divergence");

  return !this->any_error();
}

template class Sol_ProjectDivergence3DDeviceStorage<float>;
template class Sol_ProjectDivergence3DDevice<float>;
template class Sol_ProjectDivergence3DDeviceCo<float>;

#ifdef OCU_DOUBLESUPPORT
template class Sol_ProjectDivergence3DDeviceStorage<double>;
template class Sol_ProjectDivergence3DDevice<double>;
template class Sol_ProjectDivergence3DDeviceCo<double>;
#endif // OCU_DOUBLESUPPORT



}

