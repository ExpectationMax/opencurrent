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

#ifndef __OCU_STORAGE_GRID_NETCFD_H__
#define __OCU_STORAGE_GRID_NETCFD_H__

#include "ocustorage/grid3d.h"
#include "netcdf.h"
#include <map>
#include <string>
#include <vector>

namespace ocu {

enum GridStaggering {
  GS_CENTER_POINT,
  GS_U_FACE,
  GS_V_FACE,
  GS_W_FACE,
  GS_CUSTOM
};

// TODO: lower case names
inline void 
grid_staggering_to_offset(GridStaggering stag, float &xoffset, float &yoffset, float &zoffset) {
  switch(stag) {
    case GS_CENTER_POINT: xoffset = 0.5f; yoffset = 0.5f; zoffset = 0.5f; break;
    case GS_U_FACE:       xoffset = 0.0f; yoffset = 0.5f; zoffset = 0.5f; break;
    case GS_V_FACE:       xoffset = 0.5f; yoffset = 0.0f; zoffset = 0.5f; break;
    case GS_W_FACE:       xoffset = 0.5f; yoffset = 0.5f; zoffset = 0.0f; break;
    default: break;
  }
}

inline void
grid_staggering_to_dimension(GridStaggering stag, int nx, int ny, int nz, int &out_nx, int &out_ny, int &out_nz) {
  out_nx = nx;
  out_ny = ny;
  out_nz = nz;

  switch(stag) {
    case GS_U_FACE:  out_nx = nx+1; break;
    case GS_V_FACE:  out_ny = ny+1; break;
    case GS_W_FACE:  out_nz = nz+1; break;
    default: break;
  }
}

class NetCDFGrid3DWriter {

  enum State {
    ST_CLOSED,
    ST_OPEN_DEFINE_MODE,
    ST_OPEN_DATA_MODE,
  };

  struct Variable {
    int id;
    nc_type type;
    int coords[4]; // time, x, y, z
    int nx, ny, nz;
  };

  //**** MEMBER VARIABLES ****
  std::string _filename;
  int _nc_id;
  int _time_id, _time_var_id; // time coordinate
  std::map<float, int> _xcoordoffset_map; // coodinate (\in [0,1]) to id
  std::map<float, int> _ycoordoffset_map; // coodinate (\in [0,1]) to id
  std::map<float, int> _zcoordoffset_map; // coodinate (\in [0,1]) to id
  std::map<std::string, Variable> _varid_map;
  std::vector<float> _time_steps;
  State _state;
  int _nx, _ny, _nz;
  float _hx, _hy, _hz;

  //**** INTERNAL METHODS ****
  void add_zero_time_level(int time_level); 
  int  xcoord(float offset);
  int  ycoord(float offset);
  int  zcoord(float offset);

  bool add_data_internal(const char *name, const Grid3DUntyped &grid, nc_type data_type, void *data, int time_level);

public:

  //**** MANAGERS ****
  NetCDFGrid3DWriter();
  ~NetCDFGrid3DWriter();

  //**** PUBLIC INTERFACE ****
  bool is_open() const { return _state != ST_CLOSED; }

  bool add_time_level(float time, size_t &level);
  int  get_time_level(float time) const { for (int i=0; i < _time_steps.size(); i++) { if (_time_steps[i] == time) return i; } return -1; }
  float get_time(int time_level)  const { return _time_steps[time_level]; }

  bool open(const char *filename, int nx, int ny, int nz, float hx=1.0f, float hy=1.0f, float hz=1.0f);

  bool define_variable(const char *name, nc_type var_type, GridStaggering stag, float offsetx=0.5f, float offsety=0.5f, float offsetz=0.5f);
  
  bool add_data(const char *name, const Grid3DHostD &grid, int time_level=0);
  bool add_data(const char *name, const Grid3DHostF &grid, int time_level=0);
  bool add_data(const char *name, const Grid3DHostI &grid, int time_level=0);
  bool close();

};


class NetCDFGrid3DReader {

  struct Variable {
    int id;
    nc_type type;
    int coords[4]; // time, x, y, z
    int nx, ny, nz;
  };

  //**** MEMBER VARIABLES ****
  std::string _filename;
  int _nc_id;
  int _time_id;
  std::map<std::string, Variable> _varid_map;
  std::vector<float> _time_steps;
  int _nx, _ny, _nz;
  float _hx, _hy, _hz;
  bool _is_open;

  bool read_variable_internal(const char *name, const Grid3DUntyped &grid, nc_type data_type, void *data, int time_level) const;

public:

  //**** MANAGERS ****
  NetCDFGrid3DReader();
  ~NetCDFGrid3DReader();

  //**** PUBLIC INTERFACE ****
  bool is_open() const { return _is_open; }
  bool open(const char *filename);

  int nx() const { return _nx; }
  int ny() const { return _ny; }
  int nz() const { return _nz; }
  float hx() const { return _hx; }
  float hy() const { return _hy; }
  float hz() const { return _hz; }

  int   num_time_levels()         const { return _time_steps.size(); }
  float get_time(int time_level)  const { return _time_steps[time_level]; }

  std::vector<std::string> list_variables() const;
  bool variable_type(const char *name, nc_type &type) const;
  bool variable_size(const char *name, int &nx, int &ny, int &nz) const;

  bool read_variable(const char *name, Grid3DHostD &grid, int time_level=0) const;
  bool read_variable(const char *name, Grid3DHostF &grid, int time_level=0) const;
  bool read_variable(const char *name, Grid3DHostI &grid, int time_level=0) const;
  
  bool close();

};

} // end namespace

#endif
