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

#include <pf_model.h>

inline void particle_set(
		__global float4* p1_angles1,
		__global float* p1_angles2,
		__global float4* p1_pos,
		const int p1,
		__global const float4* p2_angles1,
		__global const float* p2_angles2,
		__global const float4* p2_pos,
		const int p2,
		const int num_particles,
		const int num_blocks);

inline float sampling_iw(
	__global float4* p_angles1,
	__global float* p_angles2,
	__global float4* p_pos,
	__constant control* control_data,
	__constant measurement* measurement_data,
	__global const float* const d_random,
	const float dt);

inline void bitonic_sort(
	__local float* const d_weights,
	__local int* const index,
	const int num_values,
	const unsigned int dir);



/*
__device__ int get_max_index(const float* const elements, const int num_elements)
{
	__shared__ int shared_index[MAX_NUM_BLOCKS];
	const int tid = threadIdx.x;

	int offset = 1;

	shared_index[tid] = tid;
	__syncthreads();

	// build max index
	for (int d = num_elements/2; d > 0; d /= 2)
	{ 
		if (tid < d)
		{
			const int ai = offset*(2*tid+1)-1;
			const int bi = offset*(2*tid+2)-1;

			shared_index[bi] = 
				elements[shared_index[ai]] > elements[shared_index[bi]]
				? shared_index[ai]
				: shared_index[bi];
		}
		offset *= 2;

		__syncthreads();
	}

	return shared_index[num_elements-1];
}
*/

/*
__device__ void get_max_min(
	const float* const elements,
	const int num_elements,
	int* const max_index,
	int* const min_index)
{
	__shared__ int shared_index_max[MAX_NUM_PARTICLES];
	__shared__ int shared_index_min[MAX_NUM_PARTICLES];
	const int tid = threadIdx.x;

	int offset = 1;

	shared_index_max[tid] = tid;
	shared_index_min[tid] = tid;
	__syncthreads();

	// build max index
	for (int d = num_elements/2; d > 0; d /= 2)
	{ 
		if (tid < d)
		{
			const int ai = offset*(2*tid+1)-1;
			const int bi = offset*(2*tid+2)-1;

			shared_index_max[bi] = 
				elements[shared_index_max[ai]] > elements[shared_index_max[bi]]
				? shared_index_max[ai]
				: shared_index_max[bi];

			shared_index_min[bi] = 
				elements[shared_index_min[ai]] < elements[shared_index_min[bi]]
				? shared_index_min[ai]
				: shared_index_min[bi];
		}
		offset *= 2;

		__syncthreads();
	}

	*max_index = shared_index_max[num_elements-1];
	*min_index = shared_index_min[num_elements-1];
}
*/
/*
 *
 */
inline float prefix_sum(__local float* const elements, const int num_elements)
{
	const int tid = get_local_id(0);

	int offset = 1;

	/* build sum in place up the tree */
	for (int d = num_elements/2; d > 0; d /= 2)
	{ 
		barrier(CLK_LOCAL_MEM_FENCE);
		if (tid < d)
		{
			const int ai = offset*(2*tid+1)-1;
			const int bi = offset*(2*tid+2)-1;

			elements[bi] += elements[ai];
		}
		offset *= 2;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	const float cum_sum = elements[num_elements - 1];

	/* store and clear the last element */
	if (tid == 0)
	{
		elements[num_elements - 1] = 0;
	}

	/* traverse down tree & build scan */
	for (int d = 1; d < num_elements; d *= 2)
	{
		offset /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (tid < d)                     
		{
			const int ai = offset*(2*tid+1)-1;
			const int bi = offset*(2*tid+2)-1;

			const float t = elements[ai];
			elements[ai] = elements[bi];
			elements[bi] += t; 
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	return cum_sum;
}

/*
 *
 */
inline int binary_search(__local const float* const haystack, const float needle, const int size)
{
	int min=0;
	int max=size-1;

	do
	{
		const int current = min + (max-min+1)/2;
		if (haystack[current] > needle) /* to the left */
		{
			max = current-1;
		}
		else /* to the right (inclusive) */
		{
			min = current;
		}
	} while(min < max);

	return min;
}

/*
 *
 */
__kernel void sampling_importance(
	__global float4* p_angles1,
	__global float* p_angles2,
	__global float4* p_pos,
	__global float* const d_particle_weights,
	__constant control* control_data,
	__constant measurement* measurement_data,
	__global const float* const d_random,
	const float dt)
{
	const int idx = get_global_id(0);

#ifdef TIMING_DEVICE
	const clock_t s0 = clock();
#endif /* TIMING_DEVICE */

	//sampling(d_particle_data, control_data, d_random, dt);

#ifdef TIMING_DEVICE
	const clock_t s1 = clock();
#endif /* TIMING_DEVICE */

	const float weight = sampling_iw(p_angles1, p_angles2, p_pos, control_data, measurement_data, d_random, dt);

#ifdef TIMING_DEVICE
	const clock_t s2 = clock();
#endif /* TIMING_DEVICE */

	d_particle_weights[idx] = weight;

#ifdef TIMING_DEVICE
	const clock_t s3 = clock();
#endif /* TIMING_DEVICE */

#ifdef TIMING_DEVICE
	if (idx == 0)
	{
		d_clocks[0] = s0;
		d_clocks[1] = s1;
		d_clocks[2] = s2;
		d_clocks[3] = s3;
	}
#endif /* TIMING_DEVICE */

}

__kernel void block_sort(
	__global float4* sorted_angles1,
	__global float* sorted_angles2,
	__global float4* sorted_pos,
	__global float* const d_particle_weights,
	__global const float4* unsorted_angles1,
	__global const float* unsorted_angles2,
	__global const float4* unsorted_pos,
	const int num_particles,
	__global float2* d_local_winner_weights)
{
	__local float shared_weights[MAX_NUM_PARTICLES];
	__local int   shared_index  [MAX_NUM_PARTICLES];

	const int blockshift = 2*get_group_id(0)*get_local_size(0);
	const int tid        = get_local_id(0);

	shared_weights[tid             ] = d_particle_weights[blockshift + tid             ];
	shared_weights[tid + get_local_size(0)] = d_particle_weights[blockshift + tid + get_local_size(0)];

	shared_index[tid             ] = tid;
	shared_index[tid + get_local_size(0)] = tid + get_local_size(0);

	bitonic_sort(shared_weights, shared_index, num_particles, 0);

	d_particle_weights[blockshift + tid             ] = shared_weights[tid             ];
	d_particle_weights[blockshift + tid + get_local_size(0)] = shared_weights[tid + get_local_size(0)];

	//d_particle_data_sorted[blockshift + tid             ] = d_particle_data_unsorted[blockshift + shared_index[tid]];
	particle_set(sorted_angles1, sorted_angles2, sorted_pos, blockshift + tid, unsorted_angles1, unsorted_angles2, unsorted_pos, blockshift + shared_index[tid], num_particles, get_num_groups(0));
	//d_particle_data_sorted[blockshift + tid + blockDim.x] = d_particle_data_unsorted[blockshift + shared_index[tid + blockDim.x]];
	particle_set(sorted_angles1, sorted_angles2, sorted_pos, blockshift + tid + get_local_size(0), unsorted_angles1, unsorted_angles2, unsorted_pos, blockshift + shared_index[tid + get_local_size(0)], num_particles, get_num_groups(0));

	if(tid == 0)
	{
		d_local_winner_weights[get_group_id(0)].x = shared_weights[tid];
		d_local_winner_weights[get_group_id(0)].y = get_group_id(0);
	}

}

/*
__global__ void sort_block_winners(
	state* const d_particles,
	state* const d_particles_unsorted,
	float* const d_weights,
	const int num_values,
	const unsigned int dir)
{
	__shared__ float shared_weights[MAX_NUM_BLOCKS * MAX_NUM_TRANSFER];
	__shared__ int   index         [MAX_NUM_BLOCKS * MAX_NUM_TRANSFER];

	const unsigned int blockshift = blockIdx.x*2*blockDim.x;
	const unsigned int tid = threadIdx.x;

	index[tid] = tid;
	index[tid + blockDim.x] = tid + blockDim.x;

	shared_weights  [tid +          0] = d_weights  [blockshift + tid +          0];
	shared_weights  [tid + blockDim.x] = d_weights  [blockshift + tid + blockDim.x];

	bitonic_sort(shared_weights, index, num_values, dir);

	d_weights  [blockshift + tid +          0] = shared_weights  [tid +          0];
	d_weights  [blockshift + tid + blockDim.x] = shared_weights  [tid + blockDim.x];

	d_particles[blockshift + tid +          0] = d_particles_unsorted[index[tid +          0]];
	d_particles[blockshift + tid + blockDim.x] = d_particles_unsorted[index[tid + blockDim.x]];
}
*/

/*
__global__ void exchange_2dtorus(
	state_data* const d_particle_data,
	float* const d_particle_weights,
	const int num_particles,
	const int num_transfer,
	const int dimx,
	const int dimy)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	//const int target_bid = (bid+gridDim.x-1)%gridDim.x;
	//d_particle_data   [(bid*num_particles) + num_particles - tid - 1] = d_particle_data   [(target_bid*num_particles) + tid];
	//d_particle_weights[(bid*num_particles) + num_particles - tid - 1] = d_particle_weights[(target_bid*num_particles) + tid];

	const int x = bid % dimx;
	const int y = bid / dimx;

	const int left   = (y                )*dimx + ((x-1+dimx) % dimx);
	const int right  = (y                )*dimx + ((x+1     ) % dimx);
	const int top    = ((y-1+dimy) % dimy)*dimx + (x                );
	const int bottom = ((y+1     ) % dimy)*dimx + (x                );

	const state p = d_particle_data   [(bid*num_particles)+tid];
	const float w = d_particle_weights[(bid*num_particles)+tid];

	//LEFT
	d_particle_data   [(left*num_particles) + num_particles - tid - 1] = p;
	d_particle_weights[(left*num_particles) + num_particles - tid - 1] = w;

	//RIGHT
	d_particle_data   [(right*num_particles) + num_particles - tid - 1 + num_transfer] = p;
	d_particle_weights[(right*num_particles) + num_particles - tid - 1 + num_transfer] = w;

	//TOP
	d_particle_data   [(top*num_particles) + num_particles - tid - 1 + (2*num_transfer)] = p;
	d_particle_weights[(top*num_particles) + num_particles - tid - 1 + (2*num_transfer)] = w;

	//BOTTOM
	d_particle_data   [(bottom*num_particles) + num_particles - tid - 1 + (3*num_transfer)] = p;
	d_particle_weights[(bottom*num_particles) + num_particles - tid - 1 + (3*num_transfer)] = w;
}
*/

__kernel void exchange_ring(
	__global float4* angles1,
	__global float* angles2,
	__global float4* pos,
	__global float* const d_particle_weights,
	const int num_particles,
	const int num_blocks)
{
	const int tid = get_local_id(0);
	const int bid = get_group_id(0);

	const int target_bid = (bid+get_num_groups(0)-1)%get_num_groups(0);
	//d_particle_data   [(bid*num_particles) + num_particles - tid - 1] = d_particle_data   [(target_bid*num_particles) + tid];
	particle_set(angles1, angles2, pos, (bid*num_particles) + num_particles - tid - 1, angles1, angles2, pos, (target_bid*num_particles) + tid, num_particles, num_blocks);
	d_particle_weights[(bid*num_particles) + num_particles - tid - 1] = d_particle_weights[(target_bid*num_particles) + tid];
}

__kernel void resampling(
	__global float4* angles1,
	__global float* angles2,
	__global float4* pos,
	__global const float4* tmp_angles1,
	__global const float* tmp_angles2,
	__global const float4* tmp_pos,
	__global const float* const d_particle_weights,
	__global const float* d_random,
	const int num_particles,
	const float resampling_freq
#ifdef PRNG_MTGP
	, const int rand_offset
#endif /* PRNG_MTGP */
)
{
#ifdef PRNG_MTGP
	d_random = &d_random[rand_offset];
	if ((d_random[get_local_size(0)*get_num_groups(0) + get_group_id(0)]-1.0f) > resampling_freq)
		return;
#else /* !PRNG_MTGP */
	if (d_random[get_local_size(0)*get_num_groups(0) + get_group_id(0)] > resampling_freq)
		return;
#endif /* PRNG_MTGP */

	const int idx = get_global_id(0);
	const int tid = get_local_id(0);

	const float d_random_resampling = d_random[idx];

	__local float shared_weights[MAX_NUM_PARTICLES];
	shared_weights[tid] = d_particle_weights[idx];
	barrier(CLK_LOCAL_MEM_FENCE);

	const float cum_sum = prefix_sum(shared_weights, num_particles);
#ifdef PRNG_MTGP
	const float selection = (d_random_resampling-1.0f)*cum_sum;
#else /* !PRNG_MTGP */
	const float selection = d_random_resampling*cum_sum;
#endif /* PRNG_MTGP */
	const int selection_index =
		binary_search(shared_weights, selection, num_particles);

	/* write results to device memory, only particle data no weights */
	//d_particle_data[idx] = d_particle_data_tmp[blockIdx.x*blockDim.x+selection_index];
	particle_set(angles1, angles2, pos, idx, tmp_angles1, tmp_angles2, tmp_pos, get_group_id(0)*get_local_size(0)+selection_index, num_particles, get_num_groups(0));
	//d_particle_weights[idx] = shared_weights[tid];
}

/*
__global__ void get_particle_max(
	const float* const d_particle_weights,
	int* const d_global_winner_index,
	const int num_particles,
	const int num_blocks)
{
	const int tid = threadIdx.x;

	__shared__ float shared_weights[MAX_NUM_BLOCKS];
	shared_weights[tid] = d_particle_weights[tid*num_particles];

	const int max_index = get_max_index(shared_weights, num_blocks);
	__syncthreads();

	if (tid == 0)
	{
		*d_global_winner_index = max_index;
	}
}
*/

#ifdef PRNG_MTGP

#ifndef D_2PI
#define D_2PI 6.28318530717958647692f
#endif /* D_2PI */

/* data input is uniformly distributed between [1.0-2.0)
 * output is standard normal distribution
 */
__kernel void boxmuller(__global float* const data, const int num_data)
{
	const int idx = get_global_id(0);

	if ((2*idx) < num_data)
	{
		float u1 = 2.0f - data[2*idx];
		float u2 = 2.0f - data[2*idx+1];

		float r  = sqrt(-2.0f * log(u1));
		float phi = D_2PI * u2;

		float z1 = r * cos(phi);
		float z2 = r * sin(phi);

		data[2*idx]   = z1;
		data[2*idx+1] = z2;
	}
}

#endif /* PRNG_MTGP */

