/*
 * This file is part of the 'Esthera' bayesian estimation software toolkit.
 * Copyright (C) 2011-2012  Mehdi Chitchian and Alexander S. van Amesfoort
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "pf_model.h"

__constant float NOISE_ANGLES = 0.2f;
__constant float NOISE_X      = 0.2f;
__constant float NOISE_Y      = 0.2f;
__constant float NOISE_V_X    = 2.0f;
__constant float NOISE_V_Y    = 2.0f;

//__constant const float NOISE_ANGLE_MEASUREMENTS = 0.1f;
//__constant const float NOISE_CAMERA_X           = 0.1f;
//__constant const float NOISE_CAMERA_Y           = 0.1f;

__constant float d_arm_lenghts[NUM_ANGLES] = {0.0f, 3.0f, 3.0f, 3.0f, 3.0f};

inline void particle_set(
		__global float4* p1_angles1,
		__global float*  p1_angles2,
		__global float4* p1_pos,
		const int p1,
		__global const float4* p2_angles1,
		__global const float*  p2_angles2,
		__global const float4* p2_pos,
		const int p2,
		const int num_particles,
		const int num_blocks)
{
	p1_angles1[p1] = p2_angles1[p2];
	p1_angles2[p1] = p2_angles2[p2];
	p1_pos[p1]     = p2_pos[p2];
}
/*
 *
 */
inline float sampling_iw(
	__global float4* p_angles1,
	__global float*  p_angles2,
	__global float4* p_pos,
	__constant control* control_data,
	__constant measurement* measurement_data,
	__global const float* const d_random,
	const float dt)
{
	const int idx = get_global_id(0);
	const int glob = get_global_size(0);
	float cameraX,cameraY,cameraZ;
	float value=0.0f;

	float4 angles1 = p_angles1[idx];
	float  angles2 = p_angles2[idx];
	float4 pos     = p_pos    [idx];

	pos.x += (pos.z * dt) + (d_random[idx] * NOISE_X);
	pos.y += (pos.w * dt) + (d_random[glob + idx] * NOISE_Y);

	pos.z += d_random[2*glob + idx] * NOISE_V_X;
	pos.w += d_random[3*glob + idx] * NOISE_V_Y;

	// i=0
	{
		angles1.x += (control_data->angles[0] * dt) + (d_random[4*glob + idx] * NOISE_ANGLES);

		float sf, cf;
		sf = sincos(angles1.x, &cf);
		//sf = sin(angles1.x);
		//cf = cos(angles1.x);
		cameraX = ( pos.x * cf) + (pos.y * sf);
		cameraY = (-pos.x * sf) + (pos.y * cf);
		cameraZ = 0.0f;

		const float e = angles1.x - measurement_data->angles[0];
		value += e * e * NOISE_ANGLE_MEASUREMENTS;
	}
#pragma unroll
	for (int i=1; i < NUM_ANGLES; ++i)
	{
		//float angle;
		float sf, cf;

		if (i==1){ angles1.y += (control_data->angles[i] * dt) + (d_random[5*glob+idx] * NOISE_ANGLES); sf = sincos(angles1.y, &cf); const float e = angles1.y - measurement_data->angles[i]; value += e * e * NOISE_ANGLE_MEASUREMENTS;}
		if (i==2){ angles1.z += (control_data->angles[i] * dt) + (d_random[6*glob+idx] * NOISE_ANGLES); sf = sincos(angles1.z, &cf); const float e = angles1.z - measurement_data->angles[i]; value += e * e * NOISE_ANGLE_MEASUREMENTS;}
		if (i==3){ angles1.w += (control_data->angles[i] * dt) + (d_random[7*glob+idx] * NOISE_ANGLES); sf = sincos(angles1.w, &cf); const float e = angles1.w - measurement_data->angles[i]; value += e * e * NOISE_ANGLE_MEASUREMENTS;}
		if (i==4){ angles2   += (control_data->angles[i] * dt) + (d_random[8*glob+idx] * NOISE_ANGLES); sf = sincos(  angles2, &cf); const float e = angles2 - measurement_data->angles[i]; value += e * e * NOISE_ANGLE_MEASUREMENTS;}

		//float sf, cf;
		//sf = sincos(angle, &cf);
		//sf = sin(angle);
		//cf = cos(angle);

		//const float nA = cameraX;
		const float nB = (cameraY * cf) - (cameraZ * sf);
		const float nC = (cameraY * sf) + (cameraZ * cf);

		//cameraX = nA;
		cameraY = nB;
		cameraZ = nC - d_arm_lenghts[i];

		//const float e = angle - measurement_data->angles[i];
		//value += e * e * NOISE_ANGLE_MEASUREMENTS;
	}

	p_angles1[idx] = angles1;
	p_angles2[idx] = angles2;
	p_pos    [idx] = pos;


	value += (cameraX-measurement_data->x) * (cameraX-measurement_data->x) * NOISE_CAMERA_X;
	value += (cameraY-measurement_data->y) * (cameraY-measurement_data->y) * NOISE_CAMERA_Y;

	//const float norm_factor = 1.0f;//0.0f / powf(2.0f*((float)M_PI), NUM_SENSORS/2);

	return exp(-value);

}

/*
__device__ __forceinline__ float importance_weight(const state_data particle_data, const measurement measurement_data)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	float cameraX,cameraY,cameraZ;
	float value=0;

	// i=0
	{
		const float x = particle_data.x[idx];
		const float y = particle_data.y[idx];
		const float angle = particle_data.angles[0+idx];
		float sf, cf;
		__sincosf(angle, &sf, &cf);
		cameraX = ( x * cf) + (y * sf);
		cameraY = (-x * sf) + (y * cf);
		cameraZ = 0;

		const float e = angle - measurement_data.angles[0];
		value += e * e * NOISE_ANGLE_MEASUREMENTS;

	}

	for (int i=1; i<NUM_ANGLES; ++i)
	{
		float sf, cf;
		const float angle = particle_data.angles[(i*blockDim.x*gridDim.x) + idx];
		__sincosf(angle, &sf, &cf);
		//const float nA = cameraX;
		const float nB = (cameraY * cf) - (cameraZ * sf);
		const float nC = (cameraY * sf) + (cameraZ * cf);

		//cameraX = nA;
		cameraY = nB;
		cameraZ = nC - d_arm_lenghts[i];

		const float e = angle - measurement_data.angles[i];
		value += e * e * NOISE_ANGLE_MEASUREMENTS;

	}

	//const float cameraX = a;
	//const float cameraY = b;

	value += (cameraX-measurement_data.x) * (cameraX-measurement_data.x) * NOISE_CAMERA_X;
	value += (cameraY-measurement_data.y) * (cameraY-measurement_data.y) * NOISE_CAMERA_Y;

	//const float norm_factor = 1.0f;//0.0f / powf(2.0f*((float)M_PI), NUM_SENSORS/2);

	return __expf(-value);
}
*/
