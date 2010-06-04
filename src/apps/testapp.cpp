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

#include <assert.h>
#include "ocuutil/imagefile.h"
#include "ocuequation/eqn_diffusion1d.h"
#include "ocuutil/timing_pool.h"


using namespace ocu;

int main(int argc, char **argv)
{
  Eqn_Diffusion1DParams params;
  params.h = 1;
  params.nx = 100;
  params.left.type = BC_NEUMANN;
  params.left.value = 0;
  params.right.type = BC_NEUMANN;
  params.right.value = 0;
  params.diffusion_coefficient = 1;

  params.initial_values.init(100, 0);

  int i,t;

  // initialize to an impulse
  for (i=0; i < 100; i++) {
    if (i < 40) params.initial_values.at(i) = 0;
    else if (i < 60) params.initial_values.at(i) = 1;
    else params.initial_values.at(i) = 0;
  }

  Eqn_Diffusion1D eqn;
  if (!eqn.set_parameters(params))
    assert(false);

  ImageFile img;
  img.allocate(100, 800);

  for (t=0; t < 800; t++) {
    eqn.advance(.1);
    eqn.copy_density_to_host();
    
    for (int i=0; i < 100; i++) {
      float soln_val = eqn.density().at(i);
      if (soln_val > 1.0f) soln_val = 1.0f;
      unsigned char color = (unsigned char) (255 * (soln_val / 1.0f));
      img.set_rgb(i,t, color, color, color);
    }
  }


  img.write_ppm("solution.ppm");

  global_timer_print();

  return 0;
}
