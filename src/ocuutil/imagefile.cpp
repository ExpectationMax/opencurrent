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

#include <memory.h>
#include <fstream>
#include "ocuutil/imagefile.h"

namespace ocu {

void 
ImageFile::clear(unsigned char r, unsigned char g, unsigned char b)
{
  memset(_red, r, _width * _height);
  memset(_green, g, _width * _height);
  memset(_blue, b, _width * _height);
}

void 
ImageFile::allocate(int width, int height)
{
  if (_red) delete[] _red;
  if (_green) delete[] _green;
  if (_blue) delete[] _blue;

  _red = new unsigned char[width * height];
  _green = new unsigned char[width * height];
  _blue = new unsigned char[width * height];

  _width = width;
  _height = height;
}

bool 
ImageFile::write_ppm(const char *filename)
{
  std::ofstream file;
  file.open(filename, std::ios::binary);

  if (!file) {
    printf("[ERROR] ImageFile::WritePPM - could not open file %s\n", filename);
    return false;
  }

  file << "P6 ";

  file << _width << " ";
  file << _height << " ";
  file << 255 << "\n";

  unsigned char *rptr = _red;
  unsigned char *gptr = _green;
  unsigned char *bptr = _blue;

  const unsigned char *rlast = rptr + (_width * _height);

  while (rptr != rlast) {
    file << *rptr++;
    file << *gptr++;
    file << *bptr++;
  }

  return true;
}


bool 
ImageFile::read_ppm(const char *filename)
{
  return false;
}


} // end namespace
