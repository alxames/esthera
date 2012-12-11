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

__kernel void reduce(
		__global float2* buffer,
		__local float2* scratch,
		__const int length,
		__global float2* result)
{
	int global_index = get_global_id(0);
	float2 accumulator = {-1, -1};
	// Loop sequentially over chunks of input vector
	while (global_index < length)
	{
		float2 element = buffer[global_index];
		accumulator = (accumulator.x > element.x) ? accumulator : element;
		global_index += get_global_size(0);
	}

	// Perform parallel reduction
	int local_index = get_local_id(0);
	scratch[local_index] = accumulator;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int offset = get_local_size(0) / 2;
			offset > 0;
			offset >>= 1)
	{
		if (local_index < offset)
		{
			float2 other = scratch[local_index + offset];
			float2 mine = scratch[local_index];
			scratch[local_index] = (mine.x > other.x) ? mine : other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (local_index == 0)
	{
		result[get_group_id(0)] = scratch[0];
	}
}

/*
__kernel void reduce_local(
		__global float2* buffer,
		__local float2* scratch,
		__const int length,
		__global float2* result)
{
	int global_index = get_global_id(0);
	int local_index = get_local_id(0);
	// Load data into local memory
	if (global_index < length)
	{
		scratch[local_index] = buffer[global_index];
	}
	else
	{
		float2 x = {-1, -1};
		// Infinity is the identity element for the min operation
		scratch[local_index] = x;//{-1, -1};
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int offset = get_local_size(0) / 2;
			offset > 0;
			offset >>= 1)
	{
		if (local_index < offset)
		{
			float2 other = scratch[local_index + offset];
			float2 mine = scratch[local_index];
			scratch[local_index] = (mine.x > other.x) ? mine : other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (local_index == 0)
	{
		result[get_group_id(0)] = scratch[0];
	}
}
*/
