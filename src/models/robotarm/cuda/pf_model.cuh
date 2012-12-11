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

#ifndef _PF_MODEL_CUH
#define _PF_MODEL_CUH 1

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>

const int NUM_ANGLES = 5;

typedef struct _state
{
	float angles[NUM_ANGLES];
	float x;
	float y;
	float vX;
	float vY;
}
state;

typedef struct _control
{
	float angles[NUM_ANGLES];
}
control;

typedef struct _measurement
{
	float angles[NUM_ANGLES];
	float x;
	float y;
}
measurement;

const int NUM_STATE_VARIABLES = NUM_ANGLES + 4;

const float NOISE_ANGLES = 0.2;
const float NOISE_X      = 0.2;
const float NOISE_Y      = 0.2;
const float NOISE_V_X    = 2.0;
const float NOISE_V_Y    = 2.0;

const float NOISE_ANGLE_MEASUREMENTS = 0.1;
const float NOISE_CAMERA_X           = 0.1;
const float NOISE_CAMERA_Y           = 0.1;

state initial_state =
{
	{0.75 * M_PI,0.75 * M_PI,0.75 * M_PI,0.75 * M_PI,0.75 * M_PI}, //angles
	7,   //x
	0,   //y
	0.0, //vX
	0.0  //vY
};



/* fixed sensor positions */
__device__ __constant__ float d_arm_lenghts[NUM_ANGLES] = {0, 3, 3, 3, 3};

/*
 *
 */
__device__ __forceinline__ void sampling(
	state* const particle_data,
	const control control_data,
	const float* const d_random,
	const float dt)
{
	for (int i=0; i < NUM_ANGLES; ++i)
	{
		particle_data->angles[i] += (control_data.angles[i] * dt) + (d_random[i] * NOISE_ANGLES);
	}

	const float vX = particle_data->vX;
	const float vY = particle_data->vY;

	particle_data->x += (vX * dt) + (d_random[NUM_ANGLES+0] * NOISE_X);
	particle_data->y += (vY * dt) + (d_random[NUM_ANGLES+1] * NOISE_Y);

	particle_data->vX = vX + (d_random[NUM_ANGLES+2] * NOISE_V_X);
	particle_data->vY = vY + (d_random[NUM_ANGLES+3] * NOISE_V_Y);
}

/*
 *
 */
__device__ __forceinline__ float importance_weight(const state* const particle_data, const measurement measurement_data)
{
	float cameraX,cameraY,cameraZ;
	float value=0;

	// i=0
	{
		const float x = particle_data->x;
		const float y = particle_data->y;
		const float angle = particle_data->angles[0];
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
		const float angle = particle_data->angles[i];
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

#endif /* _PF_MODEL_CUH */
