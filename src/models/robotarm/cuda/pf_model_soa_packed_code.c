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

#ifndef _PF_MODEL_CODE_C
#define _PF_MODEL_CODE_C 1

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define float4 float

#include "pf_model_soa_packed_data.h"


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

#endif /* _PF_MODEL_CODE_C */
