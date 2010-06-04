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

#ifndef __OCU_UTIL_IMAGEFILE_H__
#define __OCU_UTIL_IMAGEFILE_H__


namespace ocu {




class ImageFile {

  //**** MEMBER VARIABLES ****
  unsigned char *_red;
  unsigned char *_green;
  unsigned char *_blue;

  int _width;
  int _height;

public:

  //**** MANAGERS ****
  ImageFile() { _red = 0; _green = 0; _blue = 0; _width = _height = 0; }
  ~ImageFile() { delete[] _red; delete[] _green; delete[] _blue; }

  //**** PUBLIC INTERFACE ****
  void clear(unsigned char r=0, unsigned char g=0, unsigned char b=0);

  void allocate(int width, int height);

  unsigned char &red(int x, int y) { return _red[y * _width + x]; }
  const unsigned char &red(int x, int y) const { return _red[y * _width + x]; }

  unsigned char &green(int x, int y) { return _green[y * _width + x]; }
  const unsigned char &green(int x, int y) const { return _green[y * _width + x]; }

  unsigned char &blue(int x, int y) { return _blue[y * _width + x]; }
  const unsigned char &blue(int x, int y) const { return _blue[y * _width + x]; }

  void set_rgb(int x, int y, unsigned char r, unsigned char g, unsigned char b) { red(x,y) = r; green(x,y) = g; blue(x,y) = b; }

  bool write_ppm(const char *filename);
  bool read_ppm(const char *filename);
};






}

#endif


