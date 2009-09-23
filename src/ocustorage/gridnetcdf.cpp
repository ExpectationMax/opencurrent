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

namespace ocu {

int  
NetCDFGrid3DWriter::xcoord(float offset)
{
  if (_xcoordoffset_map.count(offset) == 0) {
    int dimid, dim_varid;
    char coord_name[1024];
    if (offset == .5f) 
      sprintf(coord_name, "XAxisCentered");
    else if (offset == 0)
      sprintf(coord_name, "XAxisUFace");
    else
      sprintf(coord_name, "XAxisOffset%.4f", offset);
    
    size_t var_nx = offset == 0 ? _nx+1 : _nx;
    nc_def_dim(_nc_id, coord_name, var_nx, &dimid);

    nc_def_var(_nc_id, coord_name, NC_FLOAT, 1, &dimid, &dim_varid);
    std::vector<float> dim_values;    
    for (int i=0; i < var_nx; i++)
      dim_values.push_back(i * _hx + offset);
    nc_put_var_float (_nc_id, dim_varid, &*dim_values.begin());

    _xcoordoffset_map[offset] = dimid;
    return dimid;
  }
  
  return _xcoordoffset_map[offset];
}

int  
NetCDFGrid3DWriter::ycoord(float offset)
{
  if (_ycoordoffset_map.count(offset) == 0) {
    int dimid, dim_varid;
    char coord_name[1024];
    if (offset == .5f) 
      sprintf(coord_name, "YAxisCentered");
    else if (offset == 0)
      sprintf(coord_name, "YAxisVFace");
    else
      sprintf(coord_name, "YAxisOffset%.4f", offset);
    
    size_t var_ny = offset == 0 ? _ny+1 : _ny;
    nc_def_dim(_nc_id, coord_name, var_ny, &dimid);

    nc_def_var(_nc_id, coord_name, NC_FLOAT, 1, &dimid, &dim_varid);
    std::vector<float> dim_values;    
    for (int i=0; i < var_ny; i++)
      dim_values.push_back(i * _hy + offset);
    nc_put_var_float (_nc_id, dim_varid, &*dim_values.begin());

    _ycoordoffset_map[offset] = dimid;
    return dimid;
  }
  
  return _ycoordoffset_map[offset];
}

int  
NetCDFGrid3DWriter::zcoord(float offset)
{
  if (_zcoordoffset_map.count(offset) == 0) {
    int dimid, dim_varid;
    char coord_name[1024];
    if (offset == .5f) 
      sprintf(coord_name, "ZAxisCentered");
    else if (offset == 0)
      sprintf(coord_name, "ZAxisWFace");
    else
      sprintf(coord_name, "ZAxisOffset%.4f", offset);
    
    size_t var_nz = offset == 0 ? _nz+1 : _nz;
    nc_def_dim(_nc_id, coord_name, var_nz, &dimid);
    
    nc_def_var(_nc_id, coord_name, NC_FLOAT, 1, &dimid, &dim_varid);
    std::vector<float> dim_values;    
    for (int i=0; i < var_nz; i++)
      dim_values.push_back(i * _hz + offset);
    nc_put_var_float (_nc_id, dim_varid, &*dim_values.begin());
    
    _zcoordoffset_map[offset] = dimid;
    return dimid;
  }
  
  return _zcoordoffset_map[offset];

}


NetCDFGrid3DWriter::NetCDFGrid3DWriter()
{
  _state = ST_CLOSED;
  _nc_id = -1;
  _time_id = -1;
  _nx = _ny = _nz = 0;
  _hx = _hy = _hz =  0;
}

NetCDFGrid3DWriter::~NetCDFGrid3DWriter()
{
  if (is_open()) close();
}

void 
NetCDFGrid3DWriter::add_zero_time_level(int time_level)
{
  if (time_level == 0 && _time_steps.size() == 0) {
    size_t dummy;
    add_time_level(0, dummy);
  }
}

bool
NetCDFGrid3DWriter::add_time_level(float time, size_t &level)
{ 
  if (!is_open()) {
    printf("[ERROR] NetCDFGrid3DWriter::add_time_level - file must be open\n");
    return false;
  }

  if (_state == ST_OPEN_DEFINE_MODE) {
    nc_enddef(_nc_id);
    _state = ST_OPEN_DATA_MODE;
  }

  _time_steps.push_back(time); 

  level = _time_steps.size() - 1;
  int ok = nc_put_var1_float (_nc_id, _time_var_id, &level, &time);
  if (ok != NC_NOERR) {
    printf("[ERROR] NetCDFGrid3DWriter::add_time_level - Error adding time level: %s\n", nc_strerror(ok));
    return false;
  }

  return true; 
}

bool NetCDFGrid3DWriter::open(const char *filename, int nx, int ny, int nz, float hx, float hy, float hz)
{
  if (is_open()) {
    printf("[ERROR] NetCDFGrid3DWriter::open - file already open\n");
    return false;
  }

#ifdef OCU_NETCDF4SUPPORT
  int ok = nc_create(filename, NC_CLOBBER|NC_NETCDF4, &_nc_id);
#else
  int ok = nc_create(filename, NC_CLOBBER, &_nc_id);
#endif // OCU_NETCDF4SUPPORT

  if (ok != NC_NOERR) {
    printf("[ERROR] NetCDFGrid3DWriter::open - Could not open \"%s\" with error %s\n", filename, nc_strerror(ok));
    return false;
  }

  _filename = filename;
  _state = ST_OPEN_DEFINE_MODE;
  _nx = nx;
  _ny = ny;
  _nz = nz;
  _hx = hx;
  _hy = hy;
  _hz = hz;

  nc_def_dim(_nc_id, "Time", NC_UNLIMITED, &_time_id);

  nc_put_att_int(_nc_id, NC_GLOBAL, "nx", NC_INT, 1, &_nx);
  nc_put_att_int(_nc_id, NC_GLOBAL, "ny", NC_INT, 1, &_ny);
  nc_put_att_int(_nc_id, NC_GLOBAL, "nz", NC_INT, 1, &_nz);
  nc_put_att_float(_nc_id, NC_GLOBAL, "hx", NC_FLOAT, 1, &_hx);
  nc_put_att_float(_nc_id, NC_GLOBAL, "hy", NC_FLOAT, 1, &_hy);
  nc_put_att_float(_nc_id, NC_GLOBAL, "hz", NC_FLOAT, 1, &_hz);

  nc_def_var(_nc_id, "Time", NC_FLOAT, 1, &_time_id, &_time_var_id); 

  return true;
}

bool NetCDFGrid3DWriter::define_variable(const char *name, nc_type var_type, GridStaggering stag, float offsetx, float offsety, float offsetz)
{
  if (!is_open()) {
    printf("[ERROR] NetCDFGrid3DWriter::define_variable - file must be opened first\n");
    return false;
  }

  if (var_type != NC_FLOAT && var_type != NC_DOUBLE && var_type != NC_INT) {
    printf("[ERROR] NetCDFGrid3DWriter::define_variable - only NC_FLOAT, NC_DOUBLE, NC_INT types supported\n");
    return false;
  }

  if (_varid_map.count(name) != 0) {
    printf("[ERROR] NetCDFGrid3DWriter::define_variable - variable \"%s\" previously defined\n", name);
    return false;
  }

  if (_state == ST_OPEN_DATA_MODE) {
    nc_redef(_nc_id);
    _state = ST_OPEN_DEFINE_MODE;
  }

  // if staggering is GS_CUSTOM, this is a no-op
  grid_staggering_to_offset(stag, offsetx, offsety, offsetz);

  Variable var;
  var.type = var_type;
  var.coords[0] = _time_id;
  var.coords[1] = xcoord(offsetx);
  var.coords[2] = ycoord(offsety);
  var.coords[3] = zcoord(offsetz);
  grid_staggering_to_dimension(stag, _nx, _ny, _nz, var.nx, var.ny, var.nz);

  int ok = nc_def_var(_nc_id, name, var.type, 4, var.coords, &var.id); 
  if (ok != NC_NOERR) {
    printf("[ERROR] NetCDFGrid3DWriter::define_variable - Failed with error %s\n", nc_strerror(ok));
  }

#ifdef OCU_NETCDF4SUPPORT
  ok = nc_def_var_deflate(_nc_id, var.id, 0, 1, 2); // default compression is 2
  if (ok != NC_NOERR) {
    printf("[ERROR] NetCDFGrid3DWriter::define_variable - nc_def_var_deflate failed with error %s\n", nc_strerror(ok));
  }
#endif // OCU_NETCDF4SUPPORT

  _varid_map[name] = var;

  return true;
}

bool NetCDFGrid3DWriter::add_data_internal(const char *name, const Grid3DUntyped &grid, nc_type data_type, void *data, int time_level)
{
  if (!is_open()) {
    printf("[ERROR] NetCDFGrid3DWriter::add_data_internal - file must be opened first\n");
    return false;
  }


  if (_state == ST_OPEN_DEFINE_MODE) {
    nc_enddef(_nc_id);
    _state = ST_OPEN_DATA_MODE;
  }

  if (!_varid_map.count(name)) {
    printf("[ERROR] NetCDFGrid3DWriter::add_data_internal - variable \"%s\" not found\n", name);
    return false;
  }

  Variable var = _varid_map[name];

  if (grid.nx() != var.nx || grid.ny() != var.ny || grid.nz() != var.nz) {
    printf("[ERROR] NetCDFGrid3DWriter::add_data_internal - dimension mismatch, expected (%d %d %d), grid has (%d %d %d)\n", var.nx, var.ny, var.nz, grid.nx(), grid.ny(), grid.nz());
    return false;
  }

  if (var.type != data_type) {
    printf("[ERROR] NetCDFGrid3DWriter::add_data_internal - variable \"%s\" type does not match data_type\n", name);
    return false;
  }

  // will leave in open data mode
  add_zero_time_level(time_level);

  size_t starts[4] = { time_level, 0, 0, 0 };
  size_t counts[4] = { 1, var.nx, var.ny, var.nz};
  ptrdiff_t strides[4] = {1, 1, 1, 1};
  ptrdiff_t dim_sizes[4] = { grid.xstride()*grid.ystride()*grid.zstride(), grid.xstride(), grid.ystride(), grid.zstride() };

  int ok = NC_NOERR;
  if (data_type == NC_FLOAT)
    ok = nc_put_varm_float(_nc_id, var.id, starts, counts, strides, dim_sizes, (float *)data);
  else if (data_type == NC_DOUBLE) 
    ok = nc_put_varm_double(_nc_id, var.id, starts, counts, strides, dim_sizes, (double *)data);
  else if (data_type == NC_INT) 
    ok = nc_put_varm_int(_nc_id, var.id, starts, counts, strides, dim_sizes, (int *)data);
  else {
    printf("[ERROR] NetCDFGrid3DWriter::add_data_internal - invalid type\n");
    return false;
  }

  if (ok != NC_NOERR) {
    printf("[ERROR] NetCDFGrid3DWriter::add_data_internal - Failed with error %s\n", nc_strerror(ok));
  }
    
  return true;
}


bool NetCDFGrid3DWriter::add_data(const char *name, const Grid3DHostD &grid, int time_level)
{
  return add_data_internal(name, grid, NC_DOUBLE, (void *)&grid.at(0,0,0), time_level);
}

bool NetCDFGrid3DWriter::add_data(const char *name, const Grid3DHostF &grid, int time_level)
{
  return add_data_internal(name, grid, NC_FLOAT, (void *)&grid.at(0,0,0), time_level);
}

bool NetCDFGrid3DWriter::add_data(const char *name, const Grid3DHostI &grid, int time_level)
{
  return add_data_internal(name, grid, NC_INT, (void *)&grid.at(0,0,0), time_level);
}

bool NetCDFGrid3DWriter::close()
{
  int ok = nc_close(_nc_id);
  if (ok != NC_NOERR) {
    printf("[ERROR] NetCDFGrid3DWriter::close - Could not close \"%s\" with error %s\n", _filename.c_str(), nc_strerror(ok));
    return false;
  }

  _state = ST_CLOSED;
  _varid_map.clear();
  _time_steps.clear();
  return true;
}




NetCDFGrid3DReader::NetCDFGrid3DReader()
{
  _nc_id = -1;
  _nx = _ny = _nz = 0;
  _hx = _hy = _hz =  0;
  _is_open = false;
}

NetCDFGrid3DReader::~NetCDFGrid3DReader()
{
  if (is_open()) close();
}




bool 
NetCDFGrid3DReader::open(const char *filename)
{
  if (is_open()) {
    printf("[ERROR] NetCDFGrid3DReader::open - file already open\n");
    return false;
  }

  int ok = nc_open(filename, NC_NOWRITE, &_nc_id);
  if (ok != NC_NOERR) {
    printf("[ERROR] NetCDFGrid3DReader::open - Could not open \"%s\" with error %s\n", filename, nc_strerror(ok));
    return false;
  }

  // read global attrs
  nc_get_att_int(_nc_id, NC_GLOBAL, "nx", &_nx);
  nc_get_att_int(_nc_id, NC_GLOBAL, "ny", &_ny);
  nc_get_att_int(_nc_id, NC_GLOBAL, "nz", &_nz);
  nc_get_att_float(_nc_id, NC_GLOBAL, "hx", &_hx);
  nc_get_att_float(_nc_id, NC_GLOBAL, "hy", &_hy);
  nc_get_att_float(_nc_id, NC_GLOBAL, "hz", &_hz);


  // read time dimension
  ok = nc_inq_dimid (_nc_id, "Time", &_time_id);
  if (ok != NC_NOERR) {
    printf("[ERROR] NetCDFGrid3DReader::open - Could not file \"time\" attribute with error %s\n", nc_strerror(ok));
    return false;
  }

  size_t ntime_levels;
  nc_inq_dimlen  (_nc_id, _time_id, &ntime_levels);
  _time_steps.resize(ntime_levels, 0);
  // read values into time levels
  int time_var_id;
  if (nc_inq_varid (_nc_id, "Time", &time_var_id) == NC_NOERR) {
    // make sure it has ntime_levels values
    int time_ndims, time_dimid;
    nc_inq_varndims(_nc_id, time_var_id, &time_ndims);
    if (time_ndims == 1) {
      nc_inq_vardimid(_nc_id, time_var_id, &time_dimid);

      if (time_dimid == _time_id) {
        // read them into _time_steps
        nc_get_var_float(_nc_id, time_var_id, &*_time_steps.begin());
      }
    }
  }

  // read all variable names / coordinates
  std::vector<int> varids;
  int nvars;

#ifdef OCU_NETCDF4SUPPORT

  nc_inq_varids(_nc_id, &nvars, 0);
  varids.resize(nvars, 0);
  nc_inq_varids(_nc_id, &nvars, &*varids.begin());

#else

  nc_inq_nvars(_nc_id, &nvars);
  // not sure if this is the right thing to do?
  for (int ii=0; ii < nvars; ii++)
    varids.push_back(ii); 
  
#endif // OCU_NETCDF4SUPPORT

  for (int i=0; i < nvars; i++) {
    Variable var;
    var.id = varids[i];

    char var_name[NC_MAX_NAME+1];
    int ndims;
    nc_inq_varname(_nc_id, varids[i], var_name);
    nc_inq_vartype(_nc_id, varids[i], &var.type);
    nc_inq_varndims(_nc_id, varids[i], &ndims);
    
    // only read 4d variables - first dim should be time
    if (ndims != 4)
      continue;

    size_t nx,ny,nz;
    nc_inq_vardimid(_nc_id, varids[i], var.coords);
    nc_inq_dimlen  (_nc_id, var.coords[1], &nx);
    nc_inq_dimlen  (_nc_id, var.coords[2], &ny);
    nc_inq_dimlen  (_nc_id, var.coords[3], &nz);
    var.nx = nx;
    var.ny = ny;
    var.nz = nz;

    _varid_map[var_name] = var;
  }

  return true;
}


std::vector<std::string> 
NetCDFGrid3DReader::list_variables() const
{
  std::vector<std::string> result;

  for (std::map<std::string, Variable>::const_iterator iter = _varid_map.begin(); iter != _varid_map.end(); ++iter) {
    result.push_back(iter->first);
  }

  return result;
}

bool 
NetCDFGrid3DReader::variable_type(const char *name, nc_type &type) const
{
  std::map<std::string, Variable>::const_iterator iter = _varid_map.find(name);
  if (iter == _varid_map.end())  {
    printf("[WARNING] NetCDFGrid3DReader::variable_type - variable \'%s\' not found\n", name);
    return false;
  }

  type = iter->second.type;

  return true;
}

bool 
NetCDFGrid3DReader::variable_size(const char *name, int &nx, int &ny, int &nz) const
{
  std::map<std::string, Variable>::const_iterator iter = _varid_map.find(name);
  if (iter == _varid_map.end())  {
    printf("[WARNING] NetCDFGrid3DReader::variable_size - variable \'%s\' not found\n", name);
    return false;
  }

  nx = iter->second.nx;
  ny = iter->second.ny;
  nz = iter->second.nz;

  return true;
}


bool 
NetCDFGrid3DReader::read_variable_internal(const char *name, const Grid3DUntyped &grid, nc_type data_type, void *data, int time_level) const
{
  std::map<std::string, Variable>::const_iterator iter = _varid_map.find(name);
  if (iter == _varid_map.end())  {
    printf("[ERROR] NetCDFGrid3DReader::read_variable_internal - variable \'%s\' not found\n", name);
    return false;
  }

  const Variable &var = iter->second;
  if (grid.nx() != var.nx || grid.ny() != var.ny || grid.nz() != var.nz) {
    printf("[ERROR] NetCDFGrid3DReader::read_variable_internal - grid dim (%d, %d, %d), variable in file (%d, %d, %d)\n", grid.nx(), grid.ny(), grid.nz(), var.nx, var.ny, var.nz);
    return false;
  }

  if (var.type != data_type) {
    printf("[ERROR] NetCDFGrid3DReader::read_variable_internal - variable '%s' not type %d\n", name, data_type);
    return false;
  }

  size_t starts[4] = { time_level, 0, 0, 0 };
  size_t counts[4] = { 1, var.nx, var.ny, var.nz};
  ptrdiff_t strides[4] = {1, 1, 1, 1};
  ptrdiff_t dim_sizes[4] = { grid.xstride()*grid.ystride()*grid.zstride(), grid.xstride(), grid.ystride(), grid.zstride() };

  int ok = NC_NOERR;
  if (data_type == NC_FLOAT)
    ok = nc_get_varm_float(_nc_id, var.id, starts, counts, strides, dim_sizes, (float *)data);
  else if (data_type == NC_DOUBLE)
    ok = nc_get_varm_double(_nc_id, var.id, starts, counts, strides, dim_sizes, (double *)data);
  else if (data_type == NC_INT)
    ok = nc_get_varm_int(_nc_id, var.id, starts, counts, strides, dim_sizes, (int *)data);
  else {
    printf("[ERROR] NetCDFGrid3DReader::read_variable_internal - Internal error - type not supported\n");
    return false;
  }

  if (ok != NC_NOERR) {
    printf("[ERROR] NetCDFGrid3DReader::read_variable_internal - Error reading variable \"%s\" attribute with error %s\n", name, nc_strerror(ok));
    return false;
  }

  return true;
}
bool 
NetCDFGrid3DReader::read_variable(const char *name, Grid3DHostD &grid, int time_level) const
{
  return read_variable_internal(name, grid, NC_DOUBLE, &grid.at(0,0,0), time_level);
}

bool 
NetCDFGrid3DReader::read_variable(const char *name, Grid3DHostF &grid, int time_level) const
{
  return read_variable_internal(name, grid, NC_FLOAT, &grid.at(0,0,0), time_level);
}

bool 
NetCDFGrid3DReader::read_variable(const char *name, Grid3DHostI &grid, int time_level) const
{
  return read_variable_internal(name, grid, NC_INT, &grid.at(0,0,0), time_level);
}


bool NetCDFGrid3DReader::close()
{
  if (!is_open()) {
    printf("[ERROR] NetCDFGrid3DReader::close - file is not open\n");
    return false;
  }

  int ok = nc_close(_nc_id);
  if (ok != NC_NOERR) {
    printf("[ERROR] NetCDFGrid3DReader::close - Could not close \"%s\" with error %s\n", _filename.c_str(), nc_strerror(ok));
    return false;
  }

  _is_open = false;
  return true;
}


} // end namespace
