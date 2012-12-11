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

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <cutil.h>

#ifdef MTGP
#include "mtgp/mtgp32-cuda.h"
#include "mtgp/mtgp32-cuda-common.h"
#else /* !MTGP */
#include <curand.h>
#endif /* MTGP */

typedef struct _robot_state
{
	float x;
	float y;
	float theta;
}
robot_state;

typedef struct _control_input
{
	float velocity;
	float angular_velocity;
}
control_input;

#ifdef MTGP
const int MTGP_GRIDSIZE = 120;
const int BOXMULLER_BLOCKSIZE = 256;
#endif /* MTGP */

const int MAX_NUM_PARTICLES = 512;
const int MAX_NUM_BLOCKS = 512;
const int MAX_NUM_TRANSFER = MAX_NUM_PARTICLES;

const int NUM_SENSORS = 10;
const int NUM_STATE_VARIABLES = 3;

const float NOISE_VELOCITY = 0.5;
const float NOISE_ANGULAR_VELOCITY = 0.1;
const float NOISE_GAMMA = 0.01;


/* fixed sensor positions */
__device__ __constant__ float d_sensor_position_x[NUM_SENSORS] =
				{12, 12, 13, 14, 16, 16, 18, 20, 20, 20};
__device__ __constant__ float d_sensor_position_y[NUM_SENSORS] =
				{12, 37, 25, 18, 12, 37, 30, 12, 25, 37};
__device__ __constant__ float d_sensor_position_z[NUM_SENSORS] =
				{ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3};

__device__ __constant__ float d_sensor_data[NUM_SENSORS];

//__device__ robot_state d_particle_winner;
//__device__ float       d_particle_winner_weight;

#ifdef TIMING_DEVICE
const int num_clocks = 7;
__device__ clock_t d_clocks[MAX_NUM_BLOCKS*num_clocks];
#endif /* TIMING_DEVICE */


__device__ inline void comparator(float* weightA, float* weightB, robot_state* particleA, robot_state* particleB, const unsigned int dir)
{
	float tmpWeight;
	robot_state tmpParticle;
	if((*weightA > *weightB) == dir)
	{
		tmpWeight = *weightA;
		*weightA = *weightB;
		*weightB = tmpWeight;

		tmpParticle = *particleA;
		*particleA = *particleB;
		*particleB = tmpParticle;
	}
}

__device__ inline void bitonic_sort(robot_state* const d_particles, float* const d_weights, const int num_values, const unsigned int dir)
{
	const unsigned int tid = threadIdx.x;

	for(uint32_t size = 2; size < num_values; size <<= 1)
	{
		//Bitonic merge
		uint32_t ddd = dir ^ ( (tid & (size / 2)) != 0 );
		for(uint32_t stride = size / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
			uint32_t pos = 2 * tid - (tid & (stride - 1));
			comparator(&d_weights[pos+0], &d_weights[pos+stride], &d_particles[pos+0], &d_particles[pos+stride], ddd);
		}
	}

	//ddd == dir for the last bitonic merge step
	{
		for(uint32_t stride = num_values / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
			uint32_t pos = 2 * tid - (tid & (stride - 1));
			comparator(&d_weights[pos+0], &d_weights[pos+stride], &d_particles[pos+0], &d_particles[pos+stride], dir);
		}
	}

	__syncthreads();
}

__global__ void sort_particles(robot_state* const d_particles, float* const d_weights, const int num_values, const unsigned int dir)
{
	__shared__ robot_state shared_particles[MAX_NUM_PARTICLES];
	__shared__ float       shared_weights  [MAX_NUM_PARTICLES];

	const unsigned int blockshift = blockIdx.x*2*blockDim.x;
	const unsigned int tid = threadIdx.x;

	shared_particles[tid +          0] = d_particles[blockshift + tid +          0];
	shared_particles[tid + blockDim.x] = d_particles[blockshift + tid + blockDim.x];
	
	shared_weights  [tid +          0] = d_weights  [blockshift + tid +          0];
	shared_weights  [tid + blockDim.x] = d_weights  [blockshift + tid + blockDim.x];

	bitonic_sort(shared_particles, shared_weights, num_values, dir);

	d_particles[blockshift + tid +          0] = shared_particles[tid +          0];
	d_particles[blockshift + tid + blockDim.x] = shared_particles[tid + blockDim.x];
	
	d_weights  [blockshift + tid +          0] = shared_weights  [tid +          0];
	d_weights  [blockshift + tid + blockDim.x] = shared_weights  [tid + blockDim.x];
}

/*
 *
 */
__device__ int get_max_index(const float* const elements, const int num_elements)
{
	__shared__ int shared_index[MAX_NUM_BLOCKS];
	const int tid = threadIdx.x;

	int offset = 1;

	shared_index[tid] = tid;
	__syncthreads();

	/* build max index */
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

/*
 *
 */
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

	/* build max index */
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

/*
 *
 */
__device__ float prefix_sum(float* const elements, const int num_elements)
{
	const int tid = threadIdx.x;

	int offset = 1;

	/* build sum in place up the tree */
	for (int d = num_elements/2; d > 0; d /= 2)
	{ 
		__syncthreads();
		if (tid < d)
		{
			const int ai = offset*(2*tid+1)-1;
			const int bi = offset*(2*tid+2)-1;

			elements[bi] += elements[ai];
		}
		offset *= 2;
	}

	__syncthreads();
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
		__syncthreads();
		if (tid < d)                     
		{
			const int ai = offset*(2*tid+1)-1;
			const int bi = offset*(2*tid+2)-1;

			const float t = elements[ai];
			elements[ai] = elements[bi];
			elements[bi] += t; 
		}
	}
	__syncthreads();

	return cum_sum;
}

/*
 *
 */
__device__ int binary_search(const float* const haystack, const float needle, const int size)
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
__device__ void sampling(
	robot_state* const particle_data,
	const control_input* const control,
	const float* const d_random,
	const float dt)
{
	const float x = particle_data->x;
	const float y = particle_data->y;
	const float theta = particle_data->theta;

	const float velocity = control->velocity + NOISE_VELOCITY * d_random[0];
	const float angular_velocity =
		control->angular_velocity + NOISE_ANGULAR_VELOCITY * d_random[1];
	const float gamma = NOISE_GAMMA * d_random[2];

	particle_data->x = x +
		velocity / angular_velocity * (sinf(theta+angular_velocity*dt) - sinf(theta));
	particle_data->y = y -
		velocity / angular_velocity * (cosf(theta+angular_velocity*dt) - cosf(theta));
	particle_data->theta = theta + angular_velocity*dt + gamma*dt;
}

/*
 *
 */
__device__ float importance_weight(const robot_state* const particle_data)
{
	const float x = particle_data->x;
	const float y = particle_data->y;

	const float norm_factor = 100000.0f / powf(2.0f*((float)M_PI), NUM_SENSORS/2);

	float value=0;

	for (int i=0; i<NUM_SENSORS; ++i)
	{
		const float d1=x-d_sensor_position_x[i];
		const float d2=y-d_sensor_position_y[i];
		const float d3=  d_sensor_position_z[i];

		const float vect=sqrtf(d1*d1+d2*d2+d3*d3) - d_sensor_data[i];
		value += vect*vect;
	}

	return norm_factor*expf(-value);
}

/*
 *
 */
__global__ void sampling_importance(
	robot_state* const d_particle_data,
	float* const d_particle_weights,
//	int* const d_block_weight_max_index,
//	int* const d_block_weight_min_index,
	robot_state* const d_block_winners_data,
	float* const d_block_winners_weights,
	const control_input control,
	const float* const d_random,
	const float dt,
	const int num_particles,
	const int num_transfer)
{
#ifdef TIMING_DEVICE
	const clock_t s0 = clock();
#endif /* TIMING_DEVICE */

	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	const float* const d_random_sampling = &d_random[NUM_STATE_VARIABLES*idx];

	__shared__ robot_state shared_particle_data[MAX_NUM_PARTICLES];
	shared_particle_data[tid] = d_particle_data[idx];

#ifdef TIMING_DEVICE
	const clock_t s1 = clock();
#endif /* TIMING_DEVICE */

	sampling(&shared_particle_data[tid], &control, d_random_sampling, dt);

#ifdef TIMING_DEVICE
	const clock_t s2 = clock();
#endif /* TIMING_DEVICE */

	__shared__ float shared_weights[MAX_NUM_PARTICLES];

#ifdef TIMING_DEVICE
	const clock_t s3 = clock();
#endif /* TIMING_DEVICE */

	const float weight = importance_weight(&shared_particle_data[tid]);

#ifdef TIMING_DEVICE
	const clock_t s4 = clock();
#endif /* TIMING_DEVICE */

	shared_weights[tid] = weight;
	__syncthreads();
	//only first half
	if (tid < num_particles/2)
	{
		bitonic_sort(shared_particle_data, shared_weights, num_particles, 0);
	}

#ifdef TIMING_DEVICE
	const clock_t s5 = clock();
#endif /* TIMING_DEVICE */

	int max_weight_index;
	int min_weight_index;
	get_max_min(shared_weights, num_particles, &max_weight_index, &min_weight_index);
	__syncthreads();

	d_particle_weights[idx] = shared_weights      [tid];
	d_particle_data   [idx] = shared_particle_data[tid];

	if (tid < num_transfer)
	{
		d_block_winners_data   [bid*num_transfer + tid] = shared_particle_data[tid];
		d_block_winners_weights[bid*num_transfer + tid] = shared_weights      [tid];
	}
	__syncthreads();


#ifdef TIMING_DEVICE
	const clock_t s6 = clock();
#endif /* TIMING_DEVICE */

	if (tid == 0)
	{
//		d_block_weight_max_index[bid] = max_weight_index + bid*blockDim.x;
//		d_block_weight_min_index[bid] = min_weight_index + bid*blockDim.x;

#ifdef TIMING_DEVICE
		d_clocks[num_clocks*bid  ] = s0;
		d_clocks[num_clocks*bid+1] = s1;
		d_clocks[num_clocks*bid+2] = s2;
		d_clocks[num_clocks*bid+3] = s3;
		d_clocks[num_clocks*bid+4] = s4;
		d_clocks[num_clocks*bid+5] = s5;
		d_clocks[num_clocks*bid+6] = s6;
#endif /* TIMING_DEVICE */
	}
}

__global__ void resampling(
	robot_state* const d_particle_data,
	const float* const d_particle_weights,
//	int* const d_block_weight_max_index,
//	int* const d_block_weight_min_index,
	const robot_state* const d_block_winners_data,
	const float*       const d_block_winners_weights,
	const float* const d_random,
	const int num_particles,
	const int num_transfer)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	//const int bid = blockIdx.x;

	const float d_random_resampling = d_random[idx];

	__shared__ float shared_weights[MAX_NUM_PARTICLES];
	__shared__ robot_state shared_particle_data[MAX_NUM_PARTICLES];
	shared_weights[tid] = d_particle_weights[idx];
	shared_particle_data[tid] = d_particle_data[idx];
	__syncthreads();

	// replace weakest particle with global winner
	if(tid >= (num_particles - num_transfer))
	{
		shared_particle_data[tid] = d_block_winners_data   [num_particles - tid -1];
		shared_weights      [tid] = d_block_winners_weights[num_particles - tid -1];
		//d_particle_data[d_block_weight_min_index[bid]] = d_particle_winner;
		//shared_particle_data[d_block_weight_min_index[bid]-bid*num_particles] = d_particle_winner;
	}
	__syncthreads();

	const float cum_sum = prefix_sum(shared_weights, num_particles);
#ifdef MTGP
	const float selection = (d_random_resampling-1)*cum_sum;
#else /* !MTGP */
	const float selection = d_random_resampling*cum_sum;
#endif /* MTGP */
	const int selection_index =
		binary_search(shared_weights, selection, num_particles);

	/* write results to device memory, only particle data no weights */
	d_particle_data[idx] = shared_particle_data[tid];//selection_index];
	//d_particle_weights[idx] = shared_weights[tid];
}

/* data input is uniformly distributed between 1.0-2.0
 * output is standard normal distribution
 */
__global__ void boxmuller(float* const data, const int num_data)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if ((2*idx) < num_data)
	{
		float u1 = data[2*idx]-1;
		float u2 = data[2*idx+1]-1;

		float r  = sqrtf(-2.0f * logf(u1));
		float phi = 2 * ((float)M_PI) * u2;

		float z1 = r * cosf(phi);
		float z2 = r * sinf(phi);

		data[2*idx]   = z1;
		data[2*idx+1] = z2;
	}

}

/*
__global__ void get_particle_max(
	const robot_state* const d_particle_data,
	const float* const d_particle_weights,
	const int* const d_block_weight_max_index,
	const int num_blocks)
{
	const int tid = threadIdx.x;

	__shared__ float shared_weights[MAX_NUM_BLOCKS];
	shared_weights[tid] = d_particle_weights[d_block_weight_max_index[tid]];

	const int max_index = get_max_index(shared_weights, num_blocks);
	__syncthreads();

	if (tid == 0)
	{
		d_particle_winner =        d_particle_data   [d_block_weight_max_index[max_index]];
		d_particle_winner_weight = d_particle_weights[d_block_weight_max_index[max_index]];
	}
}
*/

#ifdef MTGP
void generate_normal_random(
	float* const data,
	const int num_samples,
	const int num_samples_boxmuller,
	mtgp32_kernel_status_t* const d_mtgp_status)
{
	mtgp32_single_kernel<<< MTGP_GRIDSIZE, THREAD_NUM >>>(
		d_mtgp_status,
		(uint32_t*) data,
		num_samples / MTGP_GRIDSIZE);
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess) {
		printf("failure in 'mtgp32_single_kernel' kernel call.\n%s\n",
			cudaGetErrorString(e));
		exit(1);
	}
	cudaThreadSynchronize();

	int boxmuller_gridsize = num_samples_boxmuller / (2*BOXMULLER_BLOCKSIZE);
	if (0 != num_samples_boxmuller % (2*BOXMULLER_BLOCKSIZE))
	{
		boxmuller_gridsize++;
	}

	boxmuller <<< boxmuller_gridsize, BOXMULLER_BLOCKSIZE >>> (data, num_samples_boxmuller);
	e = cudaGetLastError();
	if (e != cudaSuccess) {
		printf("failure in 'boxmuller' kernel call.\n%s\n", cudaGetErrorString(e));
		exit(1);
	}
	cudaThreadSynchronize();
}
#endif /* MTGP */


int64_t time_diff(const struct timeval t1, const struct timeval t2)
{
	return ((int64_t)t1.tv_sec-t2.tv_sec) * 1000000 + ((int64_t)t1.tv_usec-t2.tv_usec);
}

int main(int argc, char* argv[])
{
	int num_particles = 256;
	int num_blocks    = 16;
	int num_transfer  = 1;

	int opt;

	while (-1 != (opt = getopt(argc, argv, "m:N:t:"))) {
		switch (opt)
		{
			case 'm':
				num_particles = atoi(optarg);
				break;
			case 'N':
				num_blocks = atoi(optarg);
				break;
			case 't':
				num_transfer = atoi(optarg);
				break;
			default: /* '?' */
				fprintf(stderr, "Usage: %s [-m #particles] [-N #cores] [-t #transfer] input_file\n", argv[0]);
				exit(EXIT_FAILURE);
		}
	}

	if (optind >= argc)
	{
		fprintf(stderr, "no input file given\n");
		exit(EXIT_FAILURE);
	}

	if (
		num_particles > MAX_NUM_PARTICLES ||
		num_blocks    > MAX_NUM_BLOCKS ||
		num_transfer  > MAX_NUM_TRANSFER)
	{
		fprintf(stderr, "exceeds maximum\n");
		exit(EXIT_FAILURE);
	}

	FILE* const input_file = fopen(argv[optind], "r");

	if (input_file == NULL)
	{
		fprintf(stderr, "could not open %s\n", argv[1]);
		exit(1);
	}

#ifdef MTGP
	/* mtgp init */
	mtgp32_kernel_status_t *d_mtgp_status;
	CUDA_SAFE_CALL(cudaMalloc(&d_mtgp_status, sizeof(mtgp32_kernel_status_t) * MTGP_GRIDSIZE));
	make_constant(MTGPDC_PARAM_TABLE, MTGP_GRIDSIZE);
	make_kernel_data(d_mtgp_status, MTGPDC_PARAM_TABLE, MTGP_GRIDSIZE);

	/* num_particles * num_variables for sampling
	 * num_particles for resampling
	 */
	int num_random = num_particles * num_blocks * (NUM_STATE_VARIABLES + 1);

	/* num_random should be multiple of LARGE_SIZE * MTGP_GRIDSIZE */
	const int r = num_random % (LARGE_SIZE * MTGP_GRIDSIZE);
	if (0 != r)
	{
		num_random = num_random + (LARGE_SIZE * MTGP_GRIDSIZE) - r;
	}

	float* d_random;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_random, sizeof(float) * num_random));

	float* d_normal_random = d_random;
	float* d_uniform_random = &d_random[num_blocks*num_particles*NUM_STATE_VARIABLES];

#else /* !MTGP */
	struct timeval t;
	gettimeofday(&t, NULL);
	const uint64_t seed = t.tv_usec;

	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, seed);


	const int num_normal_random = num_particles * num_blocks * NUM_STATE_VARIABLES;
	const int num_uniform_random = num_particles * num_blocks;

	float* d_normal_random;
	float* d_uniform_random;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_normal_random, sizeof(float) * num_normal_random));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_uniform_random, sizeof(float) * num_uniform_random));
#endif /* MTGP */

	/* particles */
	robot_state* d_particle_data;
	CUDA_SAFE_CALL(cudaMalloc(&d_particle_data,
		sizeof(robot_state) * num_particles * num_blocks));

	/* particle weights */
	float* d_particle_weights;
	CUDA_SAFE_CALL(cudaMalloc(&d_particle_weights,
		sizeof(float) * num_particles * num_blocks));

	/* max and min particles per block */
/*
	int* d_block_weight_max_index;
	int* d_block_weight_min_index;
	CUDA_SAFE_CALL(cudaMalloc(&d_block_weight_max_index, sizeof(int)*num_blocks));
	CUDA_SAFE_CALL(cudaMalloc(&d_block_weight_min_index, sizeof(int)*num_blocks));
*/
	robot_state* d_block_winners_data;
	CUDA_SAFE_CALL(cudaMalloc(&d_block_winners_data, sizeof(robot_state) * num_blocks * num_transfer));

	float* d_block_winners_weights;
	CUDA_SAFE_CALL(cudaMalloc(&d_block_winners_weights, sizeof(float) * num_blocks * num_transfer));

	/* init particle data */
	{
		robot_state local_particle_data[num_blocks*num_particles];

		for (int i=0; i< num_blocks*num_particles; ++i)
		{
			local_particle_data[i].x = 20;
			local_particle_data[i].y = 40;
			local_particle_data[i].theta = -M_PI/2;
		}
		CUDA_SAFE_CALL(cudaMemcpy(
			d_particle_data,
			local_particle_data,
			sizeof(robot_state)*num_blocks*num_particles,
			cudaMemcpyHostToDevice));
	}

	/* sensor/control input */
	float sensor_data[NUM_SENSORS];
	control_input control;
	control.velocity = 0;
	control.angular_velocity = 0;
	float dt;

	robot_state actual_state;
	int sample_count=0;

#ifdef DEBUG_ESTIMATE
	float error_sum = 0;
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
	int64_t s1_total=0;
	int64_t s2_total=0;
	int64_t s3_total=0;
#ifdef TIMING_DEVICE
	int64_t s3_1_total=0;
	int64_t s3_2_total=0;
	int64_t s3_3_total=0;
#endif /* TIMING_DEVICE */
	int64_t s4_total=0;
	int64_t s5_total=0;
#endif /* TIMING_HOST */

	while (16 == fscanf(input_file, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
			&sensor_data[0],
			&sensor_data[1],
			&sensor_data[2],
			&sensor_data[3],
			&sensor_data[4],
			&sensor_data[5],
			&sensor_data[6],
			&sensor_data[7],
			&sensor_data[8],
			&sensor_data[9],
			&control.velocity,
			&control.angular_velocity,
			&dt,
			&actual_state.x,
			&actual_state.y,
			&actual_state.theta)) /*has data*/
	{
		cudaError_t e;

#ifdef TIMING_HOST
		struct timeval t1,t2;
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */

		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_sensor_data, sensor_data,
			sizeof(float)*NUM_SENSORS, 0, cudaMemcpyHostToDevice));

#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s1 = time_diff(t2,t1);
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */

#ifdef MTGP
		generate_normal_random(
			d_random,
			num_random,
			num_blocks*num_particles*NUM_STATE_VARIABLES,
			d_mtgp_status);
#else /* !MTGP */
		curandGenerateNormal(generator, d_normal_random, num_normal_random, 0, 1);
		curandGenerateUniform(generator, d_uniform_random, num_uniform_random);
#endif /* MTGP */

#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s2 = time_diff(t2,t1);
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */

		sampling_importance<<< num_blocks, num_particles >>>(
			d_particle_data,
			d_particle_weights,
			//d_block_weight_max_index,
			//d_block_weight_min_index,
			d_block_winners_data,
			d_block_winners_weights,
			control,
			d_normal_random,
			dt,
			num_particles,
			num_transfer);
		e = cudaGetLastError();
		if (e != cudaSuccess) {
			printf("failure in 'sampling_importance' kernel call.\n%s\n",
				cudaGetErrorString(e));
			exit(1);
		}
		CUDA_SAFE_CALL(cudaThreadSynchronize());

#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s3 = time_diff(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		robot_state local_particle_data[num_blocks*num_particles];
		float local_particle_weights[num_blocks*num_particles];
		CUDA_SAFE_CALL(cudaMemcpy(
			local_particle_data,
			d_particle_data,
			sizeof(robot_state)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(
			local_particle_weights,
			d_particle_weights,
			sizeof(float)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));

		printf("SAMPLING\n");
		for (int i=0; i<num_blocks; ++i)
		{
			printf("block %i\n", i);
			for (int j=0; j<num_particles; ++j)
			{
				robot_state* current =
					&local_particle_data[i*num_particles+j];
				printf("%2i  x: %10f  y: %10f  theta: %10f   | w: %.32f\n",
					i*num_particles+j,
					current->x,
					current->y,
					current->theta,
					local_particle_weights[i*num_particles+j]);

			}
			printf("-------------------------\n");
		}
/*
		int lbwmaxi[MAX_NUM_BLOCKS];
		int lbwmini[MAX_NUM_BLOCKS];
		CUDA_SAFE_CALL(cudaMemcpy(
			lbwmaxi,
			d_block_weight_max_index,
			sizeof(int)*num_blocks,
			cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(
			lbwmini,
			d_block_weight_min_index,
			sizeof(int)*num_blocks,
			cudaMemcpyDeviceToHost));
		for (int i=0; i<num_blocks; ++i)
		{
			int maxi=lbwmaxi[i];
			int mini=lbwmini[i];
			robot_state* cmaxi = &local_particle_data[maxi];
			robot_state* cmini = &local_particle_data[mini];
			printf("block %2i winner: %2i x: %f y: %f theta: %f w: %.32f\n",
				i,
				maxi,
				cmaxi->x,
				cmaxi->y,
				cmaxi->theta,
				local_particle_weights[maxi]);
			printf("block %2i  loser: %2i x: %f y: %f theta: %f w: %.32f\n",
				i,
				mini,
				cmini->x,
				cmini->y,
				cmini->theta,
				local_particle_weights[mini]);
		}
		printf("-------------------------\n");
*/
#endif /* DEBUG */

#ifdef TIMING_DEVICE
#ifdef TIMING_HOST
		clock_t lc[num_blocks*num_clocks];
		CUDA_SAFE_CALL(cudaMemcpyFromSymbol(
			lc,
			d_clocks,
			sizeof(clock_t)*num_blocks*num_clocks,
			0,
			cudaMemcpyDeviceToHost));

		//for (int i=0; i<num_blocks; ++i)
		const int i=0;
		clock_t total, t_sampling, t_importance_weights, t_maxmin;
		t_sampling           = lc[i*num_clocks+3]-lc[i*num_clocks+0];
		t_importance_weights = lc[i*num_clocks+5]-lc[i*num_clocks+3];
		t_maxmin             = lc[i*num_clocks+6]-lc[i*num_clocks+5];
		total = t_sampling + t_importance_weights + t_maxmin;

		const int64_t s3_1 = t_sampling * s3 / total;
		const int64_t s3_2 = t_importance_weights * s3 / total;
		const int64_t s3_3 = t_maxmin * s3 / total;

		/*printf("TT %li(%5.2lf) %li(%5.2lf) %li(%5.2lf) %li(%5.2lf) %li(%5.2lf) %li(%5.2lf)\n",
			s1,((double)s1)*100/total,
			s2,((double)s2)*100/total,
			s3,((double)s3)*100/total,
			s4,((double)s4)*100/total,
			s5,((double)s5)*100/total,
			s6,((double)s6)*100/total);*/
#endif /* TIMING_HOST */
#endif /* TIMING_DEVICE */

#ifdef TIMING_HOST
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */

#ifdef DEBUG
		{
			printf("BLOCK WINNERS  PRE SORT\n");
			robot_state local_winners_data   [MAX_NUM_BLOCKS * MAX_NUM_TRANSFER];
			float       local_winners_weights[MAX_NUM_BLOCKS * MAX_NUM_TRANSFER];

			CUDA_SAFE_CALL(cudaMemcpy(
						local_winners_data,
						d_block_winners_data,
						sizeof(robot_state) * num_blocks * num_transfer,
						cudaMemcpyDeviceToHost));

			CUDA_SAFE_CALL(cudaMemcpy(
						local_winners_weights,
						d_block_winners_weights,
						sizeof(float) * num_blocks * num_transfer,
						cudaMemcpyDeviceToHost));

			for (int i=0; i < num_blocks; ++i)
			{
				for (int j=0; j < num_transfer; ++j)
				{
					printf("%2i  x: %10f  y: %10f  theta: %10f   | w: %.32f\n",
						i,
						local_winners_data   [i*num_transfer + j].x,
						local_winners_data   [i*num_transfer + j].y,
						local_winners_data   [i*num_transfer + j].theta,
						local_winners_weights[i*num_transfer + j]);
				}
			}
			printf("\n");
		}
#endif /* DEBUG */

		/*get_particle_max<<< 1, num_blocks >>>(
			d_particle_data,
			d_particle_weights,
			d_block_weight_max_index,
			num_blocks);*/
		sort_particles<<< 1, num_blocks * num_transfer / 2>>>(
			d_block_winners_data,
			d_block_winners_weights,
			num_blocks * num_transfer,
			0);
		e = cudaGetLastError();
		if (e != cudaSuccess) {
			printf("failure in 'sort_particles' kernel call.\n%s\n",
				cudaGetErrorString(e));
			exit(1);
		}

		CUDA_SAFE_CALL(cudaThreadSynchronize());
#ifdef DEBUG
		{
			printf("BLOCK WINNERS  POST SORT\n");
			robot_state local_winners_data   [MAX_NUM_BLOCKS * MAX_NUM_TRANSFER];
			float       local_winners_weights[MAX_NUM_BLOCKS * MAX_NUM_TRANSFER];

			CUDA_SAFE_CALL(cudaMemcpy(
						local_winners_data,
						d_block_winners_data,
						sizeof(robot_state) * num_blocks * num_transfer,
						cudaMemcpyDeviceToHost));

			CUDA_SAFE_CALL(cudaMemcpy(
						local_winners_weights,
						d_block_winners_weights,
						sizeof(float) * num_blocks * num_transfer,
						cudaMemcpyDeviceToHost));

			for (int i=0; i < num_blocks; ++i)
			{
				for (int j=0; j < num_transfer; ++j)
				{
					printf("%2i  x: %10f  y: %10f  theta: %10f   | w: %.32f\n",
						i,
						local_winners_data   [i*num_transfer + j].x,
						local_winners_data   [i*num_transfer + j].y,
						local_winners_data   [i*num_transfer + j].theta,
						local_winners_weights[i*num_transfer + j]);
				}
			}
			printf("\n");
		}

#endif /* DEBUG */

#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s4 = time_diff(t2,t1);
#endif /* TIMING_HOST */


#ifdef DEBUG_ESTIMATE
		float lpww;
		robot_state lpw;
		CUDA_SAFE_CALL(cudaMemcpy(
			&lpw,
			d_block_winners_data,
			sizeof(robot_state),
			cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(
			&lpww,
			d_block_winners_weights,
			sizeof(float),
			cudaMemcpyDeviceToHost));
		float estimate_error = sqrtf((lpw.x-actual_state.x)*(lpw.x-actual_state.x)+(lpw.y-actual_state.y)*(lpw.y-actual_state.y));
		error_sum += estimate_error;
#ifndef GLOBAL_ONLY
		printf("EST x: %10f  y: %10f  theta: %10f  w: %.32f\n",
			lpw.x,
			lpw.y,
			lpw.theta,
			lpww);
		printf("ACT x: %10f  y: %10f  theta: %10f  | err: %f\n\n",
			actual_state.x,
			actual_state.y,
			actual_state.theta,
			estimate_error);
#endif /* GLOBAL_ONLY */
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */

		resampling<<< num_blocks, num_particles >>>(
			d_particle_data,
			d_particle_weights,
			//d_block_weight_max_index,
			//d_block_weight_min_index,
			d_block_winners_data,
			d_block_winners_weights,
			d_uniform_random,
			num_particles,
			num_transfer);

		e = cudaGetLastError();
		if (e != cudaSuccess) {
			printf("failure in 'resampling' kernel call.\n%s\n", cudaGetErrorString(e));
			exit(1);
		}

		CUDA_SAFE_CALL(cudaThreadSynchronize());

#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s5 = time_diff(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		printf("RESAMPLING\n");
		CUDA_SAFE_CALL(cudaMemcpy(
			local_particle_data,
			d_particle_data,
			sizeof(robot_state)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(
			local_particle_weights,
			d_particle_weights,
			sizeof(float)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));

		for (int i=0; i<num_blocks; ++i)
		{
			printf("block %i\n", i);
			for (int j=0; j<num_particles; ++j)
			{
				robot_state* current =
					&local_particle_data[i*num_particles+j];
				printf("%2i  x: %10f  y: %10f  theta: %10f   | w: %.32f\n",
					i*num_particles+j,
					current->x,
					current->y,
					current->theta,
					local_particle_weights[i*num_particles+j]);

			}
			printf("-------------------------\n");
		}
		printf("\n");
#endif /* DEBUG */

#ifdef TIMING_HOST
#ifndef GLOBAL_ONLY
		printf("%2ld %4ld %4ld "
#ifdef TIMING_DEVICE
				"(%3ld %3ld %3ld) "
#endif /* TIMING_DEVICE */
				"%3ld %4ld | %ld\n",
				s1,//time_diff(s2,s1),//((double)time_diff(s2,s1)*100)/time_diff(s6,s1),
				s2,//time_diff(s3,s2),//((double)time_diff(s3,s2)*100)/time_diff(s6,s1),
				s3,//time_diff(s4,s3),//((double)time_diff(s4,s3)*100)/time_diff(s6,s1),
#ifdef TIMING_DEVICE
				s3_1,
				s3_2,
				s3_3,
#endif /* TIMING_DEVICE */
				s4,//time_diff(s5,s4),//((double)time_diff(s5,s4)*100)/time_diff(s6,s1),
				s5,//time_diff(s6,s5),//((double)time_diff(s6,s5)*100)/time_diff(s6,s1),
				s1+s2+s3+s4+s5);//time_diff(s6,s1));
#endif /* GLOBAL_ONLY */

		s1_total += s1;
		s2_total += s2;
		s3_total += s3;
#ifdef TIMING_DEVICE
		s3_1_total += s3_1;
		s3_2_total += s3_2;
		s3_3_total += s3_3;
#endif /* TIMING_DEVICE */
		s4_total += s4;
		s5_total += s5;
#endif /* TIMING_HOST */

		sample_count++;
	}

#ifdef DEBUG_ESTIMATE
	printf("%d %d %d %.16f\n", num_blocks, num_particles, num_transfer, error_sum/sample_count);
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
		printf("%d %d %ld %ld %ld "
#ifdef TIMING_DEVICE
				"%ld %ld %ld "
#endif /* TIMING_DEVICE */
				"%ld %ld %ld\n",
				num_blocks,
				num_particles,
				s1_total/sample_count,
				s2_total/sample_count,
				s3_total/sample_count,
#ifdef TIMING_DEVICE
				s3_1_total/sample_count,
				s3_2_total/sample_count,
				s3_3_total/sample_count,
#endif /* TIMING_DEVICE */
				s4_total/sample_count,
				s5_total/sample_count,
				(s1_total+s2_total+s3_total+s4_total+s5_total)/sample_count);
#endif /* TIMING_HOST */

#ifdef MTGP
	cudaFree(d_mtgp_status);
	cudaFree(d_random);
#else /* !MTGP */
	cudaFree(d_normal_random);
	cudaFree(d_uniform_random);
	curandDestroyGenerator(generator);
#endif /* MTGP */
	cudaFree(d_particle_data);
	cudaFree(d_particle_weights);
	//cudaFree(d_block_weight_max_index);
	//cudaFree(d_block_weight_min_index);

	fclose(input_file);

	return 0;
}

