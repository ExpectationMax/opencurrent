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
#include "ocuequation/sol_mgpressure3d.h"
#include "ocustorage/grid3dboundary.h"

namespace ocu {



template<typename T>
BoundaryConditionSet
Sol_MultigridPressure3DDeviceCo<T>::get_bc_at_level(int level) const
{
  BoundaryConditionSet bc_result = _local_bc[level];
  if (level != 0) 
    bc_result.make_homogeneous();
  return bc_result;
}


template<typename T>
void Sol_MultigridPressure3DDeviceCo<T>::apply_boundary_conditions(int level)
{
  if (level < _multi_thread_cutoff_level) {

    // todo: can these proceed in parallel? does the order matter?

    if (!invoke_kernel_enforce_bc(this->get_u(level), this->get_bc_at_level(level), this->hx(level), this->hy(level), this->hz(level))) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::apply_boundary_conditions - failed at level %d \n", level);
      this->add_error();
    }

    ThreadManager::compute_fence();

    CoArrayManager::barrier_exchange(_u_posx_hdl[level]);
    CoArrayManager::barrier_exchange(_u_negx_hdl[level]);

    CoArrayManager::barrier_exchange_fence();

  }
  else if (ThreadManager::this_image() == 0) {
    // only do work on thread 0.  This is entirely local, so we can just call the base class.
    if (!invoke_kernel_enforce_bc(this->get_u(level), this->get_bc_at_level(level), this->hx(level), this->hy(level), this->hz(level))) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::apply_boundary_conditions - failed at level %d \n", level);
      this->add_error();
    }
  }
}

template<typename T>
void 
Sol_MultigridPressure3DDeviceCo<T>::apply_host_boundary_conditions(Grid3DHost<T> &h_u, int level)
{
  if (level < this->_multi_thread_cutoff_level) {

//    printf("right one\n");

    // todo: can these proceed in parallel? does the order matter?
    if (!apply_3d_boundary_conditions_level1_nocorners(h_u, get_bc_at_level(level), this->hx(level), this->hy(level), this->hz(level))) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::apply_host_boundary_conditions - failed at level %d \n", level);
      this->add_error();
    }

    ThreadManager::compute_fence();

    CoArrayManager::barrier_exchange(_hu_posx_hdl[level]);
    CoArrayManager::barrier_exchange(_hu_negx_hdl[level]);

    CoArrayManager::barrier_exchange_fence();

  }
  else if (ThreadManager::this_image() == 0) {
    // only do work on thread 0.  This is entirely local, so we can just call the base class.
    if (!apply_3d_boundary_conditions_level1_nocorners(h_u, this->get_bc_at_level(level), this->hx(level), this->hy(level), this->hz(level))) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::apply_host_boundary_conditions - failed at level %d \n", level);
      this->add_error();
    }
  }

}


template<typename T>
void Sol_MultigridPressure3DDeviceCo<T>::relax(int level, int iterations, Sol_MultigridPressure3DBase::RelaxOrder order)
{
  if (level < _multi_thread_cutoff_level || ThreadManager::this_image() == 0) {

    apply_boundary_conditions(level);
    Sol_MultigridPressure3DDevice<T>::relax(level, iterations, order);
  }
}

template<typename T>
void Sol_MultigridPressure3DDeviceCo<T>::restrict_residuals(int fine_level, int coarse_level, double *l2, double *linf)
{
  bool do_restrict = fine_level != coarse_level;

  // special case: at cutoff level, and not doing restrict.  then only thread 1 does anything.
  // if restricting to level <= cutoff, do the restrict
  // if restrict level is the cutoff, after restrict do gather
  // calculate norms using co_ reductions.
  if (!do_restrict && coarse_level == _multi_thread_cutoff_level) {
    if (ThreadManager::this_image() == 0)
      Sol_MultigridPressure3DDevice<T>::restrict_residuals(fine_level, coarse_level, l2, linf);
  }
  else if (coarse_level <= _multi_thread_cutoff_level) {

      check_ok(bind_tex_calculate_residual(this->get_u(fine_level), this->get_b(fine_level)));
      check_ok(invoke_kernel_calculate_residual(this->get_u(fine_level), this->get_b(fine_level), this->get_r(fine_level), this->get_h(fine_level)));
      check_ok(this->unbind_tex_calculate_residual());

      if (do_restrict ) {

        if (coarse_level == _multi_thread_cutoff_level) {
          // thread 0 restricts into _r_restrict, which was allocated specifically as restrict target at cutoff level.
          check_ok(invoke_kernel_restrict(this->get_r(fine_level), _b_restrict));

        }
        else {
          // proceed as normal locally
          check_ok(invoke_kernel_restrict(this->get_r(fine_level), this->get_b(coarse_level)));
        }

      }

      if (l2) {
        T residual_norm = 0;
        check_ok(get_co_r(fine_level).co_reduce_sqrsum(residual_norm));
        residual_norm /= (this->nx(fine_level) * this->ny(fine_level) * this->nz(fine_level) * num_active_images(fine_level));
        *l2 = sqrt(residual_norm);
      }

      if (linf) {
        T linf_norm = 0;
        check_ok(get_co_r(fine_level).co_reduce_maxabs(linf_norm));
        *linf = (double)linf_norm;
      }
  }

  // at cutoff level, gather into thread 0 after calculation
  if (coarse_level == _multi_thread_cutoff_level && fine_level != coarse_level) {
    CoArrayManager::barrier_exchange(_gather_b_hdl);
    CoArrayManager::barrier_exchange_fence();
  }

  // beyond cutoff, thread 0 proceeds, all others no-op.
  if ((coarse_level > _multi_thread_cutoff_level) && (ThreadManager::this_image() == 0)) {
    Sol_MultigridPressure3DDevice<T>::restrict_residuals(fine_level, coarse_level, l2, linf);
  }

  if (this->any_error()) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceCo::restrict_residuals - failed at level %d -> %d\n", fine_level, coarse_level);
  }
}

template<typename T>
void Sol_MultigridPressure3DDeviceCo<T>::prolong(int coarse_level, int fine_level)
{
  if (coarse_level < _multi_thread_cutoff_level) {
    // all threads proceeeds as normal
    Sol_MultigridPressure3DDevice<T>::prolong(coarse_level, fine_level);
  }
  else if (coarse_level == _multi_thread_cutoff_level) {

    // scatter u, then prolong in parallel
    CoArrayManager::barrier_exchange(_scatter_u_hdl);
    CoArrayManager::barrier_exchange_fence();

    check_ok(bind_tex_prolong(_u_prolong, this->get_u(fine_level)));
    check_ok(invoke_kernel_prolong(_u_prolong, this->get_u(fine_level)));

    check_ok(this->unbind_tex_prolong());

    // this happens after the scatter
    apply_boundary_conditions(fine_level);

  }
  else if (coarse_level > _multi_thread_cutoff_level && ThreadManager::this_image() == 0){
    // thread 0 proceeeds as normal
    Sol_MultigridPressure3DDevice<T>::prolong(coarse_level, fine_level);
  }

  if (this->any_error()) {
    printf("[ERROR] Sol_MultigridPressure3DDeviceCo::prolong - failed at level %d -> %d\n", coarse_level, fine_level);
  }
}

template<typename T>
void Sol_MultigridPressure3DDeviceCo<T>::clear_zero(int level)
{
  // TODO: i think this is correct... need to double check when this gets called
  if (level < _multi_thread_cutoff_level || ThreadManager::this_image() == 0) {
    Sol_MultigridPressure3DDevice<T>::clear_zero(level);
  }
}


template<typename T>
Sol_MultigridPressure3DDeviceCo<T>::Sol_MultigridPressure3DDeviceCo(const char *id) :
  _co_pressure((std::string(id) + std::string(".u.0")).c_str()),
  _id(id)
{
  _gather_b_hdl = -1;
  _scatter_u_hdl = -1;

  _multi_thread_cutoff_level = -1; 

  grid_size_for_cutoff = 64 * 64 * 64;
}

template<typename T>
Sol_MultigridPressure3DDeviceCo<T>::~Sol_MultigridPressure3DDeviceCo()
{
  for (int level = 0; level < _u_posx_hdl.size(); level++) {
    CoArrayManager::barrier_deallocate(_u_posx_hdl[level]);
    CoArrayManager::barrier_deallocate(_u_negx_hdl[level]);
  }

  CoArrayManager::barrier_deallocate(_gather_b_hdl);
  CoArrayManager::barrier_deallocate(_scatter_u_hdl);
}


template<typename T>
bool Sol_MultigridPressure3DDeviceCo<T>::initialize_storage(int nx_val, int ny_val, int nz_val, double hx_val, double hy_val, double hz_val, Grid3DDevice<T> *rhs)
{
  int tid = ThreadManager::this_image();
  int num_images = ThreadManager::num_images();

  //**** verify input & init base storage ****

  if (!this->initialize_base_storage(nx_val * num_images, ny_val, nz_val, hx_val, hy_val, hz_val)) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - error in grid calculation\n");
    return false;
  }

  int b_gx = rhs->gx();
  int b_gy = rhs->gy();
  int b_gz = rhs->gz();
  int b_padx = rhs->paddingx();
  int b_pady = rhs->paddingy();
  int b_padz = rhs->paddingz();

  if (b_gx < 1 || b_gy < 1 || b_gz < 1) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - rhs has invalid ghost cells (%d,%d,%d), must be >= 1\n", b_gx, b_gy, b_gz);
    return false;
  }

  if (rhs->nx() != nx_val, rhs->ny() != ny_val, rhs->nz() != nz_val) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - rhs dimension mismatch (%d,%d,%d) != (%d,%d,%d)\n", rhs->nx(), rhs->ny(), rhs->nz(), nx_val, ny_val, nz_val);
    return false;
  }

  //**** cutoff level ****

  // select _multi_thread_cutoff_level/  For simplicity, the finest grid cannot be below the cutoff by construction.
  int level;
  
  this->_multi_thread_cutoff_level = this->_num_levels;

  // force cutoff >= 1
  for (level = 1; level < this->_num_levels; level++) {
    int volume = this->nx(level) * this->ny(level) * this->nz(level);
    // we we get to cutoff size, all data gets copied to image 0 to do all the work.
    if (volume <= grid_size_for_cutoff) {
      _multi_thread_cutoff_level = level;
      break;
    }
  }


  // this is size at which relax() will run
  for (level = 0; level < _multi_thread_cutoff_level && level < this->_num_levels; level++) {
    this->_dim[level].x /= num_images;
  }

  //**** boundary conditions ****

  this->_update_bc_between_colors = true;

  // figure out who our neighbors are
  int negx_image = (tid - 1 + num_images) % num_images;
  int posx_image = (tid + 1) % num_images;

  if (tid == 0 && this->bc.xneg.type != BC_PERIODIC)
    negx_image = -1;

  if (tid == num_images - 1 && this->bc.xpos.type != BC_PERIODIC) 
    posx_image = -1;



  //**** grid hierarchy ****

  // do allocation and initialization
  this->_r_grid = new Grid3DDevice<T> *[this->_num_levels];
  this->_u_grid = new Grid3DDevice<T> *[this->_num_levels];
  this->_b_grid = new Grid3DDevice<T> *[this->_num_levels];
  this->_hu_grid = new Grid3DHost<T> *[this->_num_levels];
  this->_hb_grid = new Grid3DHost<T> *[this->_num_levels];
  
  // init all ptrs to null
  memset(this->_hu_grid, 0, sizeof(Grid3DHost<T> *) * this->_num_levels);
  memset(this->_hb_grid, 0, sizeof(Grid3DHost<T> *) * this->_num_levels);
  memset(this->_u_grid, 0, sizeof(Grid3DDevice<T> *) * this->_num_levels);
  memset(this->_r_grid, 0, sizeof(Grid3DDevice<T> *) * this->_num_levels);
  memset(this->_b_grid, 0, sizeof(Grid3DDevice<T> *) * this->_num_levels);

  _hu_posx_hdl.resize(this->_num_levels, -1);
  _hu_negx_hdl.resize(this->_num_levels, -1);
  _u_posx_hdl.resize(this->_num_levels, -1);
  _u_negx_hdl.resize(this->_num_levels, -1);

  this->_b_grid[0] = rhs;

  char r_id[1024];
  char u_id[1024];
  char hu_id[1024];
  char b_id[1024];
  sprintf(r_id, "%s.r.0", _id.c_str());

  this->_r_grid[0] = new Grid3DDeviceCo<T>(r_id);
  if (!get_co_r(0).init(nx_val, ny_val, nz_val,b_gx,b_gy,b_gz, b_padx, b_pady, b_padz)) { 
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _r_grid[0]\n");
    return false;
  }

  this->_u_grid[0] = &_co_pressure;
  if (!get_co_u(0).init(nx_val, ny_val, nz_val,b_gx,b_gy,b_gz, b_padx, b_pady, b_padz)) {
    printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _u_grid[0]\n");
    return false;
  }

  this->_u_grid[0]->clear_zero();

  if (_multi_thread_cutoff_level < this->_num_levels) {
    if (!_u_prolong.init(this->nx(_multi_thread_cutoff_level) / num_images, this->ny(_multi_thread_cutoff_level ), this->nz(_multi_thread_cutoff_level ),1,1,1)) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _u_prolong\n");
      return false;
    }

    if (!_b_restrict.init(this->nx(_multi_thread_cutoff_level) / num_images, this->ny(_multi_thread_cutoff_level ), this->nz(_multi_thread_cutoff_level ),1,1,1)) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _b_restrict\n");
      return false;
    }
  }


  for (level=1; level < this->_num_levels; level++) {

    sprintf(u_id, "%s.u.%d", _id.c_str(), level);
    sprintf(r_id, "%s.r.%d", _id.c_str(), level);
    sprintf(b_id, "%s.b.%d", _id.c_str(), level);

    this->_u_grid[level] = new Grid3DDeviceCo<T>(u_id);
    this->_r_grid[level] = new Grid3DDeviceCo<T>(r_id);
    this->_b_grid[level] = new Grid3DDeviceCo<T>(b_id);


    // todo: really, we don't need to allocate all of these on the non-zero thread.  but then bookkeeping gets a bit tricker
    if (!get_co_u(level).init(this->nx(level), this->ny(level), this->nz(level),1,1,1)) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _u_grid[%d]\n", level);
      return false;
    }
    if (!get_co_b(level).init(this->nx(level), this->ny(level), this->nz(level),1,1,1)) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _b_grid[%d]\n", level);
      return false;
    }
    if (!get_co_r(level).init(this->nx(level), this->ny(level), this->nz(level),1,1,1)) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - could not initialize _r_grid[%d]\n", level);
      return false;
    }

    this->_u_grid[level]->clear_zero();
    this->_r_grid[level]->clear_zero();
    this->_b_grid[level]->clear_zero();

    if (level == this->_num_levels-1) {
      sprintf(hu_id, "%s.hu.0", _id.c_str());

      this->_hu_grid[level] = new Grid3DHostCo<T>(hu_id);
      this->_hb_grid[level] = new Grid3DHost<T>();
      this->_hu_grid[level]->init_congruent(*this->_u_grid[level]);
      this->_hb_grid[level]->init_congruent(*this->_b_grid[level]);
      this->_hu_grid[level]->clear_zero();
      this->_hb_grid[level]->clear_zero();
    }
  }


  for (level=0; level < this->_num_levels; level++) {

    BoundaryConditionSet local = this->bc;
    if (tid == 0 && level >= _multi_thread_cutoff_level) {
      _local_bc.push_back(this->bc);
    }
    else {
      // by this point, negx_image & posx_image take periodic bc's into account
      if (negx_image != -1)
        local.xneg.type = BC_NONE;
      
      if (posx_image != -1) 
        local.xpos.type = BC_NONE;
    
      _local_bc.push_back(local);
    }

    if (level < _multi_thread_cutoff_level) {

      // set up ghost node exchange
      if (posx_image != -1) {
        Region3D from_posx = get_co_u(level).region(this->nx(level)-1)()();
        Region3D to_posx   = get_co_u(level).co(posx_image)->region(-1)()();

        int hdl = CoArrayManager::barrier_allocate(to_posx, from_posx);
        if (hdl == -1) {
          printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate u_posx_hdl[%d]\n", level);
        }
        _u_posx_hdl[level] = hdl;
      }
      else {
        _u_posx_hdl[level] = CoArrayManager::barrier_allocate();
      }

      if (negx_image != -1) {
        Region3D from_negx = get_co_u(level).region(0)()();
        Region3D to_negx   = get_co_u(level).co(negx_image)->region(this->nx(level))()();

        int hdl = CoArrayManager::barrier_allocate(to_negx, from_negx);
        if (hdl == -1) {
          printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate u_negx_hdl[%d]\n", level);
        }
        _u_negx_hdl[level] = hdl;
      }
      else {
        _u_negx_hdl[level] = CoArrayManager::barrier_allocate();
      }

      // do host u exchange as well
      if (level == this->_num_levels-1) {

        if (posx_image != -1) {
          Region3D from_posx = get_host_co_u(level)->region(this->nx(level)-1)()();
          Region3D to_posx   = get_host_co_u(level)->co(posx_image)->region(-1)()();

          int hdl = CoArrayManager::barrier_allocate(to_posx, from_posx);
          if (hdl == -1) {
            printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate hu_posx_hdl[%d]\n", level);
          }
          _hu_posx_hdl[level] = hdl;
        }
        else {
          _hu_posx_hdl[level] = CoArrayManager::barrier_allocate();
        }

        if (negx_image != -1) {
          Region3D from_negx = get_host_co_u(level)->region(0)()();
          Region3D to_negx   = get_host_co_u(level)->co(negx_image)->region(this->nx(level))()();

          int hdl = CoArrayManager::barrier_allocate(to_negx, from_negx);
          if (hdl == -1) {
            printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate hu_negx_hdl[%d]\n", level);
          }
          _hu_negx_hdl[level] = hdl;
        }
        else {
          _hu_negx_hdl[level] = CoArrayManager::barrier_allocate();
        }
      }

    }
    else if (level == _multi_thread_cutoff_level) {
      
      int slice_nx = this->nx(level) / num_images;

      Region3D from_b =            _b_restrict.region(0           ,slice_nx-1)()();
      Region3D to_b   = get_co_b(level).co(0)->region(slice_nx*tid,slice_nx*(tid+1)-1)()();
      _gather_b_hdl = CoArrayManager::barrier_allocate(to_b, from_b);
      if (_gather_b_hdl == -1) {
        printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate gather_b\n");
      }

      // need to transfer 1 extra cell in all dimensions to account for boundary conditions.
      Region3D from_u = get_co_u(level).co(0)->region(slice_nx*tid-1,slice_nx*(tid+1))()();
      Region3D to_u   =             _u_prolong.region(            -1,slice_nx)()();
      _scatter_u_hdl = CoArrayManager::barrier_allocate(to_u, from_u);
      if (_scatter_u_hdl  == -1) {
        printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - failed to allocate scatter_u\n");
      }
    }
  }

  // post-validation
  for (level=1; level < this->_num_levels; level++) {
    if (!this->get_u(level).check_interior_dimension_match(this->get_r(level)) || !this->get_u(level).check_layout_match(this->get_r(level)) ||
        !this->get_u(level).check_interior_dimension_match(this->get_b(level)) || !this->get_u(level).check_layout_match(this->get_b(level))) {
      printf("[ERROR] Sol_MultigridPressure3DDevice::initialize_storage - grid dimension mismatch\n");
      return false;
    }
  }

  this->initialize_diag_grids();

  return true;

}

template class Sol_MultigridPressure3DDeviceCo<float>;

#ifdef OCU_DOUBLESUPPORT
template class Sol_MultigridPressure3DDeviceCo<double>;
#endif

} // end namespace
