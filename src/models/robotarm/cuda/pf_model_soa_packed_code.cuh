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

#ifndef _PF_MODEL_CODE_CUH
#define _PF_MODEL_CODE_CUH 1

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>

#include "pf_model_soa_packed_data.h"

const float fc_x = 408.65831;
const float fc_y = 409.69059;
const float cc_x = 166.32611;
const float cc_y = 108.93539;
//const float alpha_c = 0;
const float kc_1 = -0.21561;
const float kc_2 = 1.60704;
const float kc_3 = -0.01291;
const float kc_4 = 0.00007;
//const float kc_5 = 0.0;

__device__ __constant__ float d_arm_lenghts[NUM_ANGLES] = {0, 3, 3, 3, 3};

__device__ __forceinline__ void particle_set(
		const state_data particle_data1,
		const int p1,
		const state_data particle_data2,
		const int p2,
		const int num_particles,
		const int num_blocks)
{
	particle_data1.angles1[p1] = particle_data2.angles1[p2];
	particle_data1.angles2[p1] = particle_data2.angles2[p2];
	particle_data1.pos[p1]     = particle_data2.pos[p2];
}
/*
 *
 */
__device__ __forceinline__ float sampling_iw(
	const state_data particle_data,
	const control control_data,
	const measurement measurement_data,
	const float* const d_random,
	const float dt)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float cameraX,cameraY,cameraZ;
	float value=0;

	float4 angles1 = particle_data.angles1[idx];
	float  angles2 = particle_data.angles2[idx];
	float4 pos     = particle_data.pos    [idx];

	pos.x += (pos.z * dt) + (d_random[((NUM_ANGLES+0)*blockDim.x*gridDim.x) + idx] * NOISE_X);
	pos.y += (pos.w * dt) + (d_random[((NUM_ANGLES+1)*blockDim.x*gridDim.x) + idx] * NOISE_Y);

	pos.z += d_random[((NUM_ANGLES+2)*blockDim.x*gridDim.x) + idx] * NOISE_V_X;
	pos.w += d_random[((NUM_ANGLES+3)*blockDim.x*gridDim.x) + idx] * NOISE_V_Y;

	// i=0
	{
		float angle;
		angles1.x += (control_data.angles[0] * dt) + (d_random[0 + idx] * NOISE_ANGLES); angle = angles1.x;

		float sf, cf;
		__sincosf(angle, &sf, &cf);
		cameraX = ( pos.x * cf) + (pos.y * sf);
		cameraY = (-pos.x * sf) + (pos.y * cf);
		cameraZ = 0;

		const float e = angle - measurement_data.angles[0];
		value += e * e * NOISE_ANGLE_MEASUREMENTS;
	}
#pragma unroll
	for (int i=1; i < NUM_ANGLES; ++i)
	{
		float angle;
		if (i==1){ angles1.y += (control_data.angles[i] * dt) + (d_random[(i*blockDim.x*gridDim.x) + idx] * NOISE_ANGLES); angle = angles1.y;}
		if (i==2){ angles1.z += (control_data.angles[i] * dt) + (d_random[(i*blockDim.x*gridDim.x) + idx] * NOISE_ANGLES); angle = angles1.z;}
		if (i==3){ angles1.w += (control_data.angles[i] * dt) + (d_random[(i*blockDim.x*gridDim.x) + idx] * NOISE_ANGLES); angle = angles1.w;}
		if (i==4){ angles2   += (control_data.angles[i] * dt) + (d_random[(i*blockDim.x*gridDim.x) + idx] * NOISE_ANGLES); angle = angles2;}

		float sf, cf;
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

	//normalised
	const float x_n = cameraX/cameraZ;
	const float y_n = cameraY/cameraZ;

	const float dxx = 2 * kc_3 * x_n * y_n + kc_4 * (3 * x_n * x_n + y_n * y_n);
	const float dxy = kc_3 * (3 * y_n * y_n + x_n * x_n) + 2 * kc_4 * x_n * y_n;

	const float r2 = (x_n * x_n) + (y_n * y_n);
	const float xdx = (1 + kc_1 * r2 + kc_2 * r2 * r2 /*+ kc_5 * r2 * r2 * r2*/) * x_n + dxx;
	const float xdy = (1 + kc_1 * r2 + kc_2 * r2 * r2 /*+ kc_5 * r2 * r2 * r2*/) * y_n + dxy;

	const float xp = fc_x * xdx + cc_x;
	const float yp = fc_y * xdy + cc_y;

	particle_data.angles1[idx] = angles1;
	particle_data.angles2[idx] = angles2;
	particle_data.pos    [idx] = pos;


	value += (xp-measurement_data.x) * (xp-measurement_data.x) * NOISE_CAMERA_X;
	value += (yp-measurement_data.y) * (yp-measurement_data.y) * NOISE_CAMERA_Y;

	//const float norm_factor = 1.0f;//0.0f / powf(2.0f*((float)M_PI), NUM_SENSORS/2);

	return __expf(-value);

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

void print_particle(const state particle)
{
	printf("x: %f  y: %f  vX: %f  vY: %f  [%f",
			particle.x,
			particle.y,
			particle.vX,
			particle.vY,
			particle.angles[0]);
	for (int i=1; i < NUM_ANGLES; ++i)
	{
		printf(", %f", particle.angles[i]);
	}
	
	printf("]");
}

void print_particle(const state_data particle, const int index, const int num_particles, const int num_blocks)
{
	printf("x: %f  y: %f  vX: %f  vY: %f  [%f",
			particle.pos[index].x,
			particle.pos[index].y,
			particle.pos[index].z,
			particle.pos[index].w,
			particle.angles1[index].x);
	for (int i=1; i < NUM_ANGLES; ++i)
	{
		float angle=-1;
		if (i==1){angle = particle.angles1[index].y;}
		if (i==2){angle = particle.angles1[index].z;}
		if (i==3){angle = particle.angles1[index].w;}
		if (i==4){angle = particle.angles2[index];}
		printf(", %f", angle);
	}
	
	printf("]");
}
void print_estimate(const state estimate, const float weights, const float error, const state actual_state)
{
	printf("%f %f %f %f %f %f\n", estimate.x, estimate.y, weights, error, actual_state.x, actual_state.y);
}

bool read_trace(FILE* const input_file, measurement* const measurement_data, control* const control_data, state* const actual_state, float* const dt)
{
	if (2 != fscanf(input_file, "%f %f", &measurement_data->x, &measurement_data->y))
		return false;
	for (int i=0; i < NUM_ANGLES; ++i)
	{
		if (1 != fscanf(input_file, "%f", &measurement_data->angles[i]))
			return false;
	}
	for (int i=0; i < NUM_ANGLES; ++i)
	{
		if (1 != fscanf(input_file, "%f", &control_data->angles[i]))
			return false;
	}
	if (3 != fscanf(input_file, "%f %f %f", dt, &actual_state->x, &actual_state->y))
		return false;
	for (int i=0; i < NUM_ANGLES; ++i)
	{
		if (1 != fscanf(input_file, "%f", &actual_state->angles[i]))
			return false;
	}
	actual_state->vX = 0;
	actual_state->vY = 0;
	return true;
	
}

float estimate_error(const state estimate, const state actual)
{
	float error = 0;
	error += (estimate.x-actual.x) * (estimate.x-actual.x) * NOISE_CAMERA_X;
	error += (estimate.y-actual.y) * (estimate.y-actual.y) * NOISE_CAMERA_Y;

	for (int i=0; i<NUM_ANGLES; ++i)
	{
		const float est = estimate.angles[i];
		const float mea = actual.angles[i];
		error += (est - mea) * (est - mea) * NOISE_ANGLE_MEASUREMENTS;
	}
	return error;
}

#endif /* _PF_MODEL_CODE_CUH */
