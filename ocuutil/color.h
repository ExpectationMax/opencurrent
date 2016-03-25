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

#ifndef __OCU_UTIL_COLOR_H__
#define __OCU_UTIL_COLOR_H__

#include <cuda.h>
#include <vector_functions.h>
#include "ocuutil/float_routines.h"
#include "ocuutil/defines.h"

namespace ocu {



//! Adapted from http://www.cs.rit.edu/~ncs/color/t_convert.html
//! r,g,b values are from 0 to 1
//! h = [0,360], s = [0,1], v = [0,1]
OCU_HOSTDEVICE 
inline float3 rgb_to_hsv( float3 rgb )
{
  float3 hsv;
	float min, max, delta;

	min = min3( rgb.x, rgb.y, rgb.z );
	max = max3( rgb.x, rgb.y, rgb.z );
	
	hsv.z = max;				// v

	delta = max - min;

	if( max != 0 )
		hsv.y = delta / max;		// s
	else {
		// r = g = b = 0		// s = 0, v is undefined
		hsv.y = 0;
		hsv.x = -1;
		return hsv;
	}

	if( rgb.x == max )
		hsv.x = ( rgb.y - rgb.z ) / delta;		// between yellow & magenta
	else if( rgb.y == max )
		hsv.x = 2 + ( rgb.z - rgb.x ) / delta;	// between cyan & yellow
	else
		hsv.x = 4 + ( rgb.x - rgb.y ) / delta;	// between magenta & cyan

	hsv.x *= 60;				// degrees
	if( hsv.x < 0 )
		hsv.x += 360;

  return hsv;
}

OCU_HOSTDEVICE 
inline float3  hsv_to_rgb(float3 hsv)
{
	float f, p, q, t;

  if( hsv.y == 0 ) {
		// achromatic (grey)
		return make_float3(hsv.z, hsv.z, hsv.z);
	}

  hsv.x = fmodf(hsv.x, 360.0f);
  if (hsv.x < 0) hsv.x += 360.0f;

  hsv.x /= 60.0f ;			// sector 0 to 5
	int i = (int) hsv.x;
	f = hsv.x - i;			// factorial part of h
	p = hsv.z * ( 1.0f - hsv.y );
	q = hsv.z * ( 1.0f - hsv.y * f );
	t = hsv.z * ( 1.0f - hsv.y * ( 1.0f  - f ) );

	switch( i ) {
		case 0:	  return make_float3(hsv.z,t,p);
		case 1:   return make_float3(q,hsv.z,p);
		case 2:   return make_float3(p,hsv.z,t);
		case 3:   return make_float3(p,q,hsv.z);
		case 4:   return make_float3(t,p,hsv.z);
		default:  return make_float3(hsv.z,p,q);
	}
}

// Simple mapping from [0,1] to a temperature-like RGB color.  Stolen from RTAPI cutil_math.h
OCU_HOSTDEVICE
inline  float3 pseudo_temperature( float t )
{
  const float b = t < 0.25f ? smoothstep( -0.25f, 0.25f, t ) : 1.0f-smoothstep( 0.25f, 0.5f, t );
  const float g = t < 0.5f  ? smoothstep( 0.0f, 0.5f, t ) : 
             (t < 0.75f ? 1.0f : 1.0f-smoothstep( 0.75f, 1.0f, t ));
  const float r = smoothstep( 0.5f, 0.75f, t );
  return make_float3( r, g, b );
}

} // end namespace

#endif

