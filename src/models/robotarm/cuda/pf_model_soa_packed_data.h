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

#ifndef _PF_MODEL_DATA_H
#define _PF_MODEL_DATA_H 1

#include <stdio.h>

const int MAX_NUM_PARTICLES = 1024;
const int MAX_NUM_BLOCKS = 65536;
const int MAX_NUM_TRANSFER = MAX_NUM_PARTICLES/2;





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

typedef struct _state_data
{
	float4 *angles1;
	float  *angles2;
	float4 *pos;
}
state_data;

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

/*
state initial_state =
{
	{0.75 * M_PI,0.75 * M_PI,0.75 * M_PI,0.75 * M_PI,0.75 * M_PI}, //angles
	7,   //x
	0,   //y
	0.0, //vX
	0.0  //vY
};
*/

void print_particle(const state particle);

void print_particle(const state_data particle, const int index, const int num_particles, const int num_blocks);

void print_estimate(const state estimate, const float weights, const float error, const state actual_state);

bool read_trace(FILE* const input_file, measurement* const measurement_data, control* const control_data, state* const actual_state, float* const dt);

float estimate_error(const state estimate, const state actual);


#endif /* _PF_MODEL_DATA_H */
