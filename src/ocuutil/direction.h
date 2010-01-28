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

#ifndef __OCU_UTIL_DIRECTION_H__
#define __OCU_UTIL_DIRECTION_H__


namespace ocu {




enum DirectionType {
  DIR_POSITIVE_FLAG = 0x01,
  DIR_NEGATIVE_FLAG = 0x02,
  DIR_XAXIS_FLAG    = 0x04,
  DIR_YAXIS_FLAG    = 0x08,
  DIR_ZAXIS_FLAG    = 0x10,
  DIR_XPOS = DIR_XAXIS_FLAG | DIR_POSITIVE_FLAG,
  DIR_XNEG = DIR_XAXIS_FLAG | DIR_NEGATIVE_FLAG,
  DIR_YPOS = DIR_YAXIS_FLAG | DIR_POSITIVE_FLAG,
  DIR_YNEG = DIR_YAXIS_FLAG | DIR_NEGATIVE_FLAG,
  DIR_ZPOS = DIR_ZAXIS_FLAG | DIR_POSITIVE_FLAG,
  DIR_ZNEG = DIR_ZAXIS_FLAG | DIR_NEGATIVE_FLAG,

};




} // end namespace

#endif

