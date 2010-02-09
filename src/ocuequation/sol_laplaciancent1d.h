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

#ifndef __OCU_EQUATION_LAPLACIAN_CENT_1D_H__
#define __OCU_EQUATION_LAPLACIAN_CENT_1D_H__

#include <string>
#include "ocuutil/boundary_condition.h"
#include "ocuequation/solver.h"
#include "ocustorage/grid1d.h"

namespace ocu {

//! \class Diffusion1DCentered
//! \brief
//! This solver computes the time derivative of a 1D density field given a diffusion coefficient and boundary conditions.
//! The discretization is assumed to be finite volume-type, where the grid point i is located position at (i*h)/2.
//! Therefore "left" boundary conditions are applied half-way between cell 0 and cell -1, and half-way between cells.
//! The solver uses a 2nd-order central differencing discretization of the laplacian.
class Sol_LaplacianCentered1D : public Solver
{
protected:
  //**** CACHED INTERNAL STATE ****
  int _nx;
  double _coefficient;
  double _hx;

public:
  //**** MANAGERS ****
  Sol_LaplacianCentered1D() { 
    _coefficient = 1.0f;
    _hx = 1;
    _nx = 0;
  }

  int          nx()                    const  { return _nx; }
  double      &h()                            { return _hx; }
  const double&h()                     const  { return _hx; }
  double      &coefficient()                  { return _coefficient; }
  const double&coefficient()           const  { return _coefficient; }
};

class Sol_LaplacianCentered1DHost : public Sol_LaplacianCentered1D
{
  //**** INTERNAL METHODS ****
  void apply_boundary_conditions();

public:

  //**** PUBLIC STATE ****
  Grid1DHostF density;
  Grid1DHostF deriv_densitydt;
  BoundaryCondition left;
  BoundaryCondition right;

  //**** MANAGERS ****
  Sol_LaplacianCentered1DHost() { 
  }

  //**** PUBLIC INTERFACE ****
  bool initialize_storage(int nx);

  bool solve();
};



//! \class Diffusion1DCenteredDevice
//! \brief
//! This solver computes the time derivative of a 1D density field given a diffusion coefficient and boundary conditions.
//! The discretization is assumed to be finite volume-type, where the grid point i is located position at (i*h)/2.
//! Therefore "left" boundary conditions are applied half-way between cell 0 and cell -1, and half-way between cells.
//! The solver uses a 2nd-order central differencing discretization of the laplacian.

class Sol_LaplacianCentered1DDevice : public Sol_LaplacianCentered1D
{
  //**** INTERNAL METHODS ****
  void apply_boundary_conditions();

public:

  //**** PUBLIC STATE ****
  Grid1DDeviceF density;
  Grid1DDeviceF deriv_densitydt;
  BoundaryCondition left;
  BoundaryCondition right;

  //**** MANAGERS ****
  Sol_LaplacianCentered1DDevice() { 
  }

  //**** PUBLIC INTERFACE ****
  bool initialize_storage(int nx);

  bool solve();

};


class Sol_LaplacianCentered1DDeviceNew : public Sol_LaplacianCentered1D
{
public:

  //**** PUBLIC STATE ****
  Grid1DDeviceF *density;
  Grid1DDeviceF deriv_densitydt;

  //**** MANAGERS ****
  Sol_LaplacianCentered1DDeviceNew()
  { 
    coefficient() = 1.0f;
    density = 0;
  }

  //**** PUBLIC INTERFACE ****
  bool initialize_storage(int nx, Grid1DDeviceF *density_val);

  bool solve();
};





} // end namespace

#endif

