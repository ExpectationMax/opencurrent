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

#ifndef __APP_EQN_CUBIC_RAYLEIGH_3D_H__
#define __APP_EQN_CUBIC_RAYLEIGH_3D_H__

#include "ocuequation/eqn_incompressns3d.h"


namespace ocu {

class Eqn_CubicRayleigh3DParamsD : public Eqn_IncompressibleNS3DParams<double>
{
};

class Eqn_CubicRayleigh3DD : public Eqn_IncompressibleNS3D<double> {

  bool enforce_thermal_boundary_conditions();
public:

  bool set_parameters(const Eqn_CubicRayleigh3DParamsD &params);
  bool advance_one_step(double dt);

};







} // end namespace


#endif

