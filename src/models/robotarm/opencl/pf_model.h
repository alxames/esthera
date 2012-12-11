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

#ifndef _PF_MODEL_H
#define _PF_MODEL_H 1

#define MAX_NUM_PARTICLES 1024
#define MAX_NUM_BLOCKS 65536
#define MAX_NUM_TRANSFER (MAX_NUM_PARTICLES/2)

#define NUM_ANGLES 5

#define NOISE_ANGLE_MEASUREMENTS 0.1f
#define NOISE_CAMERA_X           0.1f
#define NOISE_CAMERA_Y           0.1f

#define NUM_STATE_VARIABLES (NUM_ANGLES + 4)

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

#endif /* _PF_MODEL_H */
