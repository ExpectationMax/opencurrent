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

#include "tests/testframework.h"

#ifdef OCU_NETCDF

#include <netcdf.h>

#include "ocustorage/gridnetcdf.h"

using namespace ocu;

DECLARE_UNITTEST_BEGIN(NetCDFTest);

float func(int x, int y) { return (x-.1f) * (y+1.233f); }

#define NX 10
#define NY 10

//! adapted from http://www.unidata.ucar.edu/software/netcdf/docs/netcdf-tutorial.html
void run()
{
  int ncid;
  int x_dimid, y_dimid;
  int dimids[2];
  int varid;

  float data[NX][NY];

  int i,j;

  for (i=0; i < NX; i++)
    for (j=0; j < NY; j++)
      data[i][j] = func(i,j);

  // write the file
  UNITTEST_ASSERT_TRUE(nc_create("NetCDFTest.nc", NC_CLOBBER, &ncid) == 0);
  UNITTEST_ASSERT_TRUE(nc_def_dim(ncid, "x", NX, &x_dimid) == 0);
  UNITTEST_ASSERT_TRUE(nc_def_dim(ncid, "y", NY, &y_dimid) == 0);
  
  dimids[0] = x_dimid;
  dimids[1] = y_dimid;
     
  UNITTEST_ASSERT_TRUE(nc_def_var(ncid, "data", NC_FLOAT, 2, dimids, &varid) == 0);     
  UNITTEST_ASSERT_TRUE(nc_enddef(ncid) == 0);
  UNITTEST_ASSERT_TRUE(nc_put_var_float(ncid, varid, &data[0][0]) == 0);
  UNITTEST_ASSERT_TRUE(nc_close(ncid) == 0);


  // now we read the file and validate we got the same data
  int in_ncid;
  int in_varid;
  float in_data[NX][NY];

  UNITTEST_ASSERT_TRUE(nc_open("NetCDFTest.nc", NC_NOWRITE, &in_ncid) == 0);
  UNITTEST_ASSERT_TRUE(nc_inq_varid(in_ncid, "data", &in_varid) == 0);
  UNITTEST_ASSERT_TRUE(nc_get_var_float(in_ncid, in_varid, &in_data[0][0]) == 0);
  UNITTEST_ASSERT_TRUE(nc_close(in_ncid) == 0);

  for (i=0; i < NX; i++)
    for (j=0; j < NY; j++)
      UNITTEST_ASSERT_TRUE(in_data[i][j] == data[i][j]);
}

DECLARE_UNITTEST_END(NetCDFTest);



DECLARE_UNITTEST_BEGIN(GridNetCDFTest);

void run() {
  int orig_nx=31, orig_ny=32, orig_nz=33;
  float orig_hx=.7f, orig_hy=.8f, orig_hz=.9f;

  Grid3DHostF gridf;
  gridf.init(orig_nx,orig_ny, orig_nz, 0,0,0, false);
  Grid3DHostD gridd;
  gridd.init(orig_nx,orig_ny, orig_nz, 0,0,0, false);
  Grid3DHostI gridi;
  gridi.init(orig_nx,orig_ny, orig_nz, 0,0,0, false);

  int i,j,k;
  for (i=0; i < orig_nx; i++)
    for (j=0; j < orig_ny; j++)
      for (k=0; k < orig_nz; k++) {
        gridf.at(i,j,k) = i*3+j*13+k*2;
        gridd.at(i,j,k) = i*3+j*13+k*2;
        gridi.at(i,j,k) = i*3+j*13+k*2;
      }

  NetCDFGrid3DWriter writer;
  UNITTEST_ASSERT_TRUE(writer.open("test_grid.nc", orig_nx, orig_ny, orig_nz, orig_hx, orig_hy, orig_hz));
  UNITTEST_ASSERT_TRUE(writer.define_variable("gridf", NC_FLOAT, GS_CENTER_POINT));
  UNITTEST_ASSERT_TRUE(writer.define_variable("gridi", NC_INT, GS_CENTER_POINT));
  UNITTEST_ASSERT_TRUE(writer.define_variable("gridd", NC_DOUBLE, GS_CENTER_POINT));

  size_t time_level;
  UNITTEST_ASSERT_TRUE(writer.add_time_level(1.0, time_level));
  UNITTEST_ASSERT_TRUE(writer.add_data("gridf", gridf, time_level));
  UNITTEST_ASSERT_TRUE(writer.add_data("gridd", gridd, time_level));
  UNITTEST_ASSERT_TRUE(writer.add_data("gridi", gridi, time_level));
  UNITTEST_ASSERT_TRUE(writer.close());

  NetCDFGrid3DReader reader;
  nc_type type;
  UNITTEST_ASSERT_TRUE(reader.open("test_grid.nc"));
  UNITTEST_ASSERT_TRUE(reader.variable_type("gridf", type));
  UNITTEST_ASSERT_EQUAL_INT(type, NC_FLOAT);
  UNITTEST_ASSERT_TRUE(reader.variable_type("gridd", type));
  UNITTEST_ASSERT_EQUAL_INT(type, NC_DOUBLE);
  UNITTEST_ASSERT_TRUE(reader.variable_type("gridi", type));
  UNITTEST_ASSERT_EQUAL_INT(type, NC_INT);

  int nx, ny, nz;
  UNITTEST_ASSERT_TRUE(reader.variable_size("gridf", nx, ny, nz));
  UNITTEST_ASSERT_EQUAL_INT(nx, orig_nx);
  UNITTEST_ASSERT_EQUAL_INT(ny, orig_ny);
  UNITTEST_ASSERT_EQUAL_INT(nz, orig_nz);

  UNITTEST_ASSERT_EQUAL_INT(reader.num_time_levels(), 1);
  UNITTEST_ASSERT_EQUAL_FLOAT(reader.get_time(0), 1.0f, 0);

  Grid3DHostF read_gridf;
  Grid3DHostD read_gridd;
  Grid3DHostI read_gridi;
  read_gridf.init(nx, ny, nz, 0, 0, 0, false);
  read_gridd.init(nx, ny, nz, 0, 0, 0, false);
  read_gridi.init(nx, ny, nz, 0, 0, 0, false);

  UNITTEST_ASSERT_TRUE(reader.read_variable("gridf", read_gridf));
  UNITTEST_ASSERT_TRUE(reader.read_variable("gridd", read_gridd));
  UNITTEST_ASSERT_TRUE(reader.read_variable("gridi", read_gridi));

  for (i=0; i < orig_nx; i++)
    for (j=0; j < orig_ny; j++)
      for (k=0; k < orig_nz; k++) {
        UNITTEST_ASSERT_EQUAL_DOUBLE(gridd.at(i,j,k), read_gridd.at(i,j,k), 0);
        UNITTEST_ASSERT_EQUAL_FLOAT (gridf.at(i,j,k), read_gridf.at(i,j,k), 0);
        UNITTEST_ASSERT_EQUAL_INT   (gridi.at(i,j,k), read_gridi.at(i,j,k));
      }

}

DECLARE_UNITTEST_END(GridNetCDFTest);

#endif