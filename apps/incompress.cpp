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

#include "ocustorage/gridnetcdf.h"
#include "ocuequation/sol_projectmixed3d.h"
#include "ocuutil/timer.h"

using namespace ocu;

#if defined(OCU_DOUBLESUPPORT) && defined(OCU_NETCDF)

int main(int argc, const char **argv) 
{

  if (argc < 4) {
    printf("usage: incompress <in.nc> <out.nc> [<tolerance>]\n");
    exit(-1);
  }

  const char *input = argv[1];
  const char *output = argv[2];
  double tolerance = 1e-6;
  if (argc == 4) 
    tolerance = atof(argv[3]);


  NetCDFGrid3DReader reader;
  NetCDFGrid3DWriter writer;
  if (!reader.open(input)) {
    printf("[ERROR] Could not open output file %s\n", output);
    exit(-1);
  }

  int nx = reader.nx();
  int ny = reader.ny();
  int nz = reader.nz();
  float hx = reader.hx();
  float hy = reader.hy();
  float hz = reader.hz();

  std::vector<Grid3DDimension> dimensions(3);
  dimensions[0].init(nx+1,ny,nz,1,1,1);
  dimensions[1].init(nx,ny+1,nz,1,1,1);
  dimensions[2].init(nx,ny,nz+1,1,1,1);  
  Grid3DDimension::pad_for_congruence(dimensions);

  Grid3DHostD h_u, h_v, h_w;
  Grid3DDeviceD d_u, d_v, d_w;

  d_u.init_congruent(dimensions[0]);
  d_v.init_congruent(dimensions[1]);
  d_w.init_congruent(dimensions[2]);
  h_u.init_congruent(dimensions[0],true);
  h_v.init_congruent(dimensions[1],true);
  h_w.init_congruent(dimensions[2],true);

  Sol_ProjectDivergenceMixed3DDeviceD projector;
  BoundaryCondition bc;
  bc.type = BC_FORCED_INFLOW_VARIABLE_SLIP;
  projector.bc = BoundaryConditionSet(bc);
  projector.initialize_storage(nx, ny, nz, hx, hy, hz, &d_u, &d_v, &d_w);

  if (!writer.open(output, nx, ny, nz, hx, hy, hz)) {
    printf("[ERROR] Could not open output file %s\n", output);
    exit(-1);
  }

  writer.define_variable("u", NC_DOUBLE, GS_U_FACE);
  writer.define_variable("v", NC_DOUBLE, GS_V_FACE);
  writer.define_variable("w", NC_DOUBLE, GS_W_FACE);

  int nsteps = reader.num_time_levels();
  for (int step = 0; step < nsteps; step++) {
    CPUTimer all_timer;
    all_timer.start();

    reader.read_variable("u", h_u, step);
    reader.read_variable("v", h_v, step);
    reader.read_variable("w", h_w, step);

    d_u.copy_all_data(h_u);
    d_v.copy_all_data(h_v);
    d_w.copy_all_data(h_w);

    double linf_before, linf_after;
    projector.solve_divergence_only();
    projector.divergence.reduce_maxabs(linf_before);

    GPUTimer solve_timer;
    solve_timer.start();
    if (!projector.solve(tolerance)) {
      printf("[ERROR] Solver failed\n");
      exit(-1);
    }
    solve_timer.stop();

    projector.solve_divergence_only();
    projector.divergence.reduce_maxabs(linf_after);

    h_u.copy_all_data(d_u);
    h_v.copy_all_data(d_v);
    h_w.copy_all_data(d_w);

    size_t out_level;
    writer.add_time_level(reader.get_time(step), out_level);
    writer.add_data("u", h_u, out_level);
    writer.add_data("v", h_v, out_level);
    writer.add_data("w", h_w, out_level);

    all_timer.stop();
    printf("[Step %d] Error: %g / %g, Time: %f sec / %f sec (%.1f%%)\n", step, linf_before, linf_after, all_timer.elapsed_sec(), solve_timer.elapsed_sec(), 100 * solve_timer.elapsed_sec() /all_timer.elapsed_sec());
  }

  return 0;
}

#else

int main(int argc, const char **argv) 
{
  printf("This program can only be compiled with double precision and NetCDF support\n");
  return -1;
}

#endif


