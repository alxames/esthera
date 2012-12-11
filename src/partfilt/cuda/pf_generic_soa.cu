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

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#ifdef MTGP
#include "mtgp/mtgp32-cuda.h"
#include "mtgp/mtgp32-cuda-common.h"
#else /* !MTGP */
#include <curand.h>
#endif /* MTGP */


typedef struct _state state;
typedef struct _control control;
typedef struct _measurement measurement;

#include <pf_model_soa.cuh>

#ifdef MTGP
const int MTGP_GRIDSIZE = 120;
const int BOXMULLER_BLOCKSIZE = 256;
#endif /* MTGP */

const int MAX_NUM_PARTICLES = 1024;
const int MAX_NUM_BLOCKS = 65536;
const int MAX_NUM_TRANSFER = MAX_NUM_PARTICLES/2;

extern const int NUM_STATE_VARIABLES;

#define CUDA_CALL(call) do { \
	cudaError_t err = call; \
	if (cudaSuccess != err) { \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
				__FILE__, __LINE__, cudaGetErrorString( err) ); \
		exit(EXIT_FAILURE);} \
} while(0)


#define CURAND_CALL(call) do { \
	curandStatus_t stat = call; \
	if (CURAND_STATUS_SUCCESS != stat) { \
		printf("Curand error %d in file  '%s' in line %i\n",stat,__FILE__,__LINE__); \
		exit(EXIT_FAILURE); } \
} while(0)

#ifdef TIMING_DEVICE
const int NUM_CLOCKS = 4;
__device__ clock_t d_clocks[NUM_CLOCKS];
#endif /* TIMING_DEVICE */

void print_particle(const state particle);
void print_particle(const state_data particle, const int index, const int num_particles, const int num_blocks);
bool read_trace(FILE* const input_file, measurement* const measurement_data, control* const control_data, state* const actual_state, float* const dt);
float estimate_error(const state estimate, const state actual);
//__device__ __forceinline__ void sampling(state_data* const particle_data, const control control_data, const float* const d_random, const float dt);
//__device__ __forceinline__ float importance_weight(const state_data* const particle_data, const measurement measurement_data);


#include "bitonic-sort.cu"



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
__global__ void sampling_importance(
	const state_data d_particle_data,
	float* const d_particle_weights,
	const control control_data,
	const measurement measurement_data,
	const float* const d_random,
	const float dt)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

#ifdef TIMING_DEVICE
	const clock_t s0 = clock();
#endif /* TIMING_DEVICE */

	//sampling(d_particle_data, control_data, d_random, dt);

#ifdef TIMING_DEVICE
	const clock_t s1 = clock();
#endif /* TIMING_DEVICE */

	const float weight = sampling_iw(d_particle_data, control_data, measurement_data, d_random, dt);

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

__global__ void block_sort(
	const state_data d_particle_data_sorted,
	float* const d_particle_weights,
	const state_data d_particle_data_unsorted,
	const int num_particles,
	float* d_local_winner_weights)
{
	__shared__ float shared_weights[MAX_NUM_PARTICLES];
	__shared__ int   shared_index  [MAX_NUM_PARTICLES];

	const int blockshift = 2*blockIdx.x*blockDim.x;
	const int tid        = threadIdx.x;

	shared_weights[tid             ] = d_particle_weights[blockshift + tid             ];
	shared_weights[tid + blockDim.x] = d_particle_weights[blockshift + tid + blockDim.x];

	shared_index[tid             ] = tid;
	shared_index[tid + blockDim.x] = tid + blockDim.x;

	bitonic_sort(shared_weights, shared_index, num_particles, 0);

	d_particle_weights[blockshift + tid             ] = shared_weights[tid             ];
	d_particle_weights[blockshift + tid + blockDim.x] = shared_weights[tid + blockDim.x];

	//d_particle_data_sorted[blockshift + tid             ] = d_particle_data_unsorted[blockshift + shared_index[tid]];
	particle_set(d_particle_data_sorted, blockshift + tid, d_particle_data_unsorted, blockshift + shared_index[tid], num_particles, gridDim.x);
	//d_particle_data_sorted[blockshift + tid + blockDim.x] = d_particle_data_unsorted[blockshift + shared_index[tid + blockDim.x]];
	particle_set(d_particle_data_sorted, blockshift + tid + blockDim.x, d_particle_data_unsorted, blockshift + shared_index[tid + blockDim.x], num_particles, gridDim.x);

	if(tid == 0)
	{
		d_local_winner_weights[blockIdx.x] = shared_weights[tid];
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

__global__ void exchange_ring(
	const state_data d_particle_data,
	float* const d_particle_weights,
	const int num_particles,
	const int num_blocks)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;

	const int target_bid = (bid+gridDim.x-1)%gridDim.x;
	//d_particle_data   [(bid*num_particles) + num_particles - tid - 1] = d_particle_data   [(target_bid*num_particles) + tid];
	particle_set(d_particle_data, (bid*num_particles) + num_particles - tid - 1, d_particle_data, (target_bid*num_particles) + tid, num_particles, num_blocks);
	d_particle_weights[(bid*num_particles) + num_particles - tid - 1] = d_particle_weights[(target_bid*num_particles) + tid];
}

__global__ void resampling(
	const state_data d_particle_data,
	const state_data d_particle_data_tmp,
	const float* const d_particle_weights,
	const float* const d_random,
	const int num_particles,
	const float resampling_freq)
{
#ifdef MTGP
	if ((d_random[blockDim.x*gridDim.x + blockIdx.x]-1.0f) > resampling_freq)
		return;
#else /* !MTGP */
	if (d_random[blockDim.x*gridDim.x + blockIdx.x] > resampling_freq)
		return;
#endif /* MTGP */

	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;

	const float d_random_resampling = d_random[idx];

	__shared__ float shared_weights[MAX_NUM_PARTICLES];
	shared_weights[tid] = d_particle_weights[idx];
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
	//d_particle_data[idx] = d_particle_data_tmp[blockIdx.x*blockDim.x+selection_index];
	particle_set(d_particle_data, idx, d_particle_data_tmp, blockIdx.x*blockDim.x+selection_index, num_particles, gridDim.x);
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

#ifdef MTGP
/* data input is uniformly distributed between [1.0-2.0)
 * output is standard normal distribution
 */
__global__ void boxmuller(float* const data, const int num_data)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if ((2*idx) < num_data)
	{
		float u1 = 2.0f - data[2*idx];
		float u2 = 2.0f - data[2*idx+1];

		float r  = sqrtf(-2.0f * logf(u1));
		float phi = 2.0f * ((float)M_PI) * u2;

		float z1 = r * cosf(phi);
		float z2 = r * sinf(phi);

		data[2*idx]   = z1;
		data[2*idx+1] = z2;
	}

}

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
	CUDA_CALL(cudaThreadSynchronize());

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
	CUDA_CALL(cudaThreadSynchronize());
}
#endif /* MTGP */

static void init_particles(const state init_state, state* const d_particle_data, const int num_blocks, const int num_particles)
{
	state *init_particle_data = (state*)malloc(sizeof(state)*num_blocks*num_particles);

	for (int i=0; i < (num_blocks*num_particles); ++i)
	{
		init_particle_data[i] = init_state;
	}
	CUDA_CALL(cudaMemcpy(
		d_particle_data,
		init_particle_data,
		sizeof(state)*num_blocks*num_particles,
		cudaMemcpyHostToDevice));
	free(init_particle_data);
}

#ifdef TIMING_HOST
static inline int64_t time_diff(const struct timeval t1, const struct timeval t2)
{
	return ((int64_t)t1.tv_sec-t2.tv_sec) * 1000000 + ((int64_t)t1.tv_usec-t2.tv_usec);
}
#endif /* TIMING_HOST */

static inline void parse_range(const char* str, int* left, int* right)
{
	const char* pos = strchr(str,':');
	if (NULL != pos)
	{
		char* l_str = strdup(str);
		const char* r_str = pos + 1;
		l_str[pos-str] = '\0';
		*left  = atoi(l_str);
		*right = atoi(r_str);
		free(l_str);
	}
	else
	{
		const int val = atoi(str);
		*left  = val;
		*right = val;
	}

}

int particle_filter(const int num_particles, const int num_blocks, const int num_transfer, const float resampling_freq, const char* input_file_str);

int main(int argc, char* argv[])
{
	int num_transfer_start  = 1;
	int num_transfer_end    = 1;

	int num_blocks_start    = 16;
	int num_blocks_end      = 16;

	int num_particles_start = 256;
	int num_particles_end   = 256;

	int loop_count          = 1;

	float resampling_freq  = 1.0;

	int device = 0;

	int opt;

	while (-1 != (opt = getopt(argc, argv, "m:N:t:r:l:d:"))) {
		switch (opt)
		{
			case 'm':
				parse_range(optarg, &num_particles_start, &num_particles_end);
				break;
			case 'N':
				parse_range(optarg, &num_blocks_start, &num_blocks_end);
				break;
			case 't':
				parse_range(optarg, &num_transfer_start, &num_transfer_end);
				break;
			case 'r':
				resampling_freq = atof(optarg);
				break;
			case 'l':
				loop_count = atoi(optarg);
				break;
			case 'd':
				device = atoi(optarg);
				break;
			default: /* '?' */
				fprintf(stderr, "Usage: %s [-m #particles] [-N #cores] [-t #transfer] [-r resamplingChance] input_file\n", argv[0]);
				exit(EXIT_FAILURE);
		}
	}

	if (optind >= argc)
	{
		fprintf(stderr, "no input file given\n");
		exit(EXIT_FAILURE);
	}

	CUDA_CALL(cudaSetDevice(device));

#ifdef DEBUG_ESTIMATE
#ifdef DEBUG_TRACE
	printf("ex ey ew ee ax ay\n");
#else
	printf("N m t e\n");
#endif /* DEBUG_TRACE */
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
	printf("N m t t1 t2 t3 "
#ifdef TIMING_DEVICE
		"t3_1 t3_2 t3_3 "
#endif /* TIMING_DEVICE */
		"t4 t5 t6 t7 total\n");
#endif /* TIMING_HOST */

	for (int num_transfer=num_transfer_start; num_transfer <= num_transfer_end; ++num_transfer)
	{
		for (int num_blocks=num_blocks_start; num_blocks <= num_blocks_end; num_blocks*=2)
		{
			for (int num_particles=num_particles_start; num_particles <= num_particles_end; num_particles*=2)
			{
				for (int i=0; i < loop_count; ++i)
				{


	if (
		num_particles > MAX_NUM_PARTICLES ||
		num_blocks    > MAX_NUM_BLOCKS ||
		num_transfer  > MAX_NUM_TRANSFER ||
		resampling_freq < 0.0 ||
		resampling_freq > 1.0)
	{
		fprintf(stderr, "exceeds maximum\n");
		exit(EXIT_FAILURE);
	}

//	if ((3*num_transfer) > num_particles)
//		continue;

	particle_filter(num_particles, num_blocks, num_transfer, resampling_freq, argv[optind]);


				}
			}
		}
	}
	return 0;
}

int particle_filter(const int num_particles, const int num_blocks, const int num_transfer, const float resampling_freq, const char* input_file_str)
{
	/* for 2d torus */
/*
	const int griddimx = exp2(floor(log2(sqrtf(num_blocks))));
	const int griddimy = num_blocks/griddimx;
*/

	FILE* const input_file = fopen(input_file_str, "r");

	if (input_file == NULL)
	{
		fprintf(stderr, "could not open %s\n", input_file_str);
		exit(1);
	}


	const int num_normal_random = num_particles * num_blocks * NUM_STATE_VARIABLES;
	const int num_uniform_random = (num_particles+1) * num_blocks;

#ifdef MTGP
	/* mtgp init */
	mtgp32_kernel_status_t *d_mtgp_status;
	CUDA_CALL(cudaMalloc(&d_mtgp_status, sizeof(mtgp32_kernel_status_t) * MTGP_GRIDSIZE));
	make_constant(MTGPDC_PARAM_TABLE, MTGP_GRIDSIZE);
	make_kernel_data(d_mtgp_status, MTGPDC_PARAM_TABLE, MTGP_GRIDSIZE);

	/* num_random should be multiple of LARGE_SIZE * MTGP_GRIDSIZE */
	const int r = (num_normal_random + num_uniform_random) % (LARGE_SIZE * MTGP_GRIDSIZE);
	const int num_random = (num_normal_random + num_uniform_random) +
		((0 != r) ? ((LARGE_SIZE * MTGP_GRIDSIZE) - r) : 0);

	float* d_random;
	CUDA_CALL(cudaMalloc((void**)&d_random, sizeof(float) * num_random));

	float* d_normal_random = d_random;
	float* d_uniform_random = &d_random[num_normal_random];

#else /* !MTGP */
	struct timeval t;
	gettimeofday(&t, NULL);
	const uint64_t seed = t.tv_usec;

	curandGenerator_t generator;
	CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));

	float* d_normal_random;
	float* d_uniform_random;
	CUDA_CALL(cudaMalloc((void**)&d_normal_random, sizeof(float) * num_normal_random));
	CUDA_CALL(cudaMalloc((void**)&d_uniform_random, sizeof(float) * num_uniform_random));
#endif /* MTGP */

	/* particles */
	state_data d_particle_data;
	{
		CUDA_CALL(cudaMalloc(&d_particle_data.angles, sizeof(float) * NUM_ANGLES * num_particles * num_blocks));
		CUDA_CALL(cudaMalloc(&d_particle_data.x, sizeof(float) * num_particles * num_blocks));
		CUDA_CALL(cudaMalloc(&d_particle_data.y, sizeof(float) * num_particles * num_blocks));
		CUDA_CALL(cudaMalloc(&d_particle_data.vX, sizeof(float) * num_particles * num_blocks));
		CUDA_CALL(cudaMalloc(&d_particle_data.vY, sizeof(float) * num_particles * num_blocks));

		state_data init_particle_data;// = (state_data*)malloc(sizeof(state_data));

		init_particle_data.angles = (float*)malloc(sizeof(float) * NUM_ANGLES * num_particles * num_blocks);
		init_particle_data.x      = (float*)malloc(sizeof(float) * num_particles * num_blocks);
		init_particle_data.y      = (float*)malloc(sizeof(float) * num_particles * num_blocks);
		init_particle_data.vX     = (float*)malloc(sizeof(float) * num_particles * num_blocks);
		init_particle_data.vY     = (float*)malloc(sizeof(float) * num_particles * num_blocks);

		for (int i=0; i < (num_blocks*num_particles); ++i)
		{
			//init_particle_data[i] = init_state;
			for (int j=0; j < NUM_ANGLES; ++j)
			{
				init_particle_data.angles[(j*num_particles*num_blocks) + i] = 0.75 * M_PI;
			}
			init_particle_data.x [i] = 7;
			init_particle_data.y [i] = 0;
			init_particle_data.vX[i] = 0;
			init_particle_data.vY[i] = 0;
		}
		CUDA_CALL(cudaMemcpy(d_particle_data.angles, init_particle_data.angles, sizeof(float)*NUM_ANGLES*num_particles*num_blocks, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(d_particle_data.x, init_particle_data.x, sizeof(float)*num_particles*num_blocks, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(d_particle_data.y, init_particle_data.y, sizeof(float)*num_particles*num_blocks, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(d_particle_data.vX, init_particle_data.vX, sizeof(float)*num_particles*num_blocks, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(d_particle_data.vY, init_particle_data.vY, sizeof(float)*num_particles*num_blocks, cudaMemcpyHostToDevice));

		free(init_particle_data.angles);
		free(init_particle_data.x);
		free(init_particle_data.y);
		free(init_particle_data.vX);
		free(init_particle_data.vY);

	}

	state_data d_particle_data_tmp;
	{
		CUDA_CALL(cudaMalloc(&d_particle_data_tmp.angles, sizeof(float) * NUM_ANGLES * num_particles * num_blocks));
		CUDA_CALL(cudaMalloc(&d_particle_data_tmp.x, sizeof(float) * num_particles * num_blocks));
		CUDA_CALL(cudaMalloc(&d_particle_data_tmp.y, sizeof(float) * num_particles * num_blocks));
		CUDA_CALL(cudaMalloc(&d_particle_data_tmp.vX, sizeof(float) * num_particles * num_blocks));
		CUDA_CALL(cudaMalloc(&d_particle_data_tmp.vY, sizeof(float) * num_particles * num_blocks));
	}

	/* particle weights */
	float* d_particle_weights;
	CUDA_CALL(cudaMalloc(&d_particle_weights,
		sizeof(float) * num_particles * num_blocks));

	thrust::device_vector<float> d_local_winner_weights(num_blocks);
	//int* d_global_winner_index;
	//CUDA_CALL(cudaMalloc(&d_global_winner_index, sizeof(int)));

	//init_particles(initial_state, d_particle_data, num_blocks, num_particles);

	/* sensor/control input */
	measurement measurement_data;
	control control_data;
	state actual_state;
	float dt;

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
	int64_t s6_total=0;
	int64_t s7_total=0;
#endif /* TIMING_HOST */

#ifdef DEBUG
	state_data local_particle_data;
	local_particle_data.angles = (float*)malloc(sizeof(float)*NUM_ANGLES*num_blocks*num_particles);
	local_particle_data.x      = (float*)malloc(sizeof(float)*num_blocks*num_particles);
	local_particle_data.y      = (float*)malloc(sizeof(float)*num_blocks*num_particles);
	local_particle_data.vX     = (float*)malloc(sizeof(float)*num_blocks*num_particles);
	local_particle_data.vY     = (float*)malloc(sizeof(float)*num_blocks*num_particles);

	//state* local_particle_data     = (state*)malloc(sizeof(state)*num_blocks*num_particles);
	//state* local_particle_data_tmp = (state*)malloc(sizeof(state)*num_blocks*num_particles);
	float* local_particle_weights  = (float*)malloc(sizeof(float)*num_blocks*num_particles);
#endif /* DEBUG */

	while (read_trace(input_file, &measurement_data, &control_data, &actual_state, &dt))
	{
		cudaError_t e;

#ifdef TIMING_HOST
		struct timeval t1,t2;
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */


#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s1 = time_diff(t2,t1);
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */

#ifdef MTGP
		generate_normal_random(
			d_random,
			num_random,
			num_normal_random,
			d_mtgp_status);
#else /* !MTGP */
		CURAND_CALL(curandGenerateNormal(generator, d_normal_random, num_normal_random, 0, 1));
		CURAND_CALL(curandGenerateUniform(generator, d_uniform_random, num_uniform_random));
		CUDA_CALL(cudaThreadSynchronize());
#endif /* MTGP */

#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s2 = time_diff(t2,t1);
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */

		sampling_importance<<< num_blocks, num_particles >>>(
			d_particle_data,
			d_particle_weights,
			control_data,
			measurement_data,
			d_normal_random,
			dt);
		e = cudaGetLastError();
		if (e != cudaSuccess) {
			printf("failure in 'sampling_importance' kernel call.\n%s\n",
				cudaGetErrorString(e));
			exit(1);
		}
		CUDA_CALL(cudaThreadSynchronize());

#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s3 = time_diff(t2,t1);
#endif /* TIMING_HOST */

#ifdef TIMING_DEVICE
#ifdef TIMING_HOST
		clock_t lc[NUM_CLOCKS];
		CUDA_CALL(cudaMemcpyFromSymbol(
			lc,
			d_clocks,
			sizeof(clock_t)*NUM_CLOCKS,
			0,
			cudaMemcpyDeviceToHost));

		clock_t total, t_sampling, t_importance_weights, t_write;
		t_sampling           = lc[1]-lc[0];
		t_importance_weights = lc[2]-lc[1];
		t_write              = lc[3]-lc[2];
		total = t_sampling + t_importance_weights + t_write;

		const int64_t s3_1 = t_sampling * s3 / total;
		const int64_t s3_2 = t_importance_weights * s3 / total;
		const int64_t s3_3 = t_write * s3 / total;
#endif /* TIMING_HOST */
#endif /* TIMING_DEVICE */

#ifdef DEBUG
		/*CUDA_CALL(cudaMemcpy(
			local_particle_data,
			d_particle_data,
			sizeof(state)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));*/
		CUDA_CALL(cudaMemcpy(local_particle_data.angles, d_particle_data.angles, sizeof(float)*NUM_ANGLES*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.x, d_particle_data.x, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.y, d_particle_data.y, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.vX, d_particle_data.vX, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.vY, d_particle_data.vY, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(
			local_particle_weights,
			d_particle_weights,
			sizeof(float)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));

		printf("SAMPLING %d\n", sample_count);
		for (int i=0; i<num_blocks; ++i)
		{
			printf("block %i\n", i);
			for (int j=0; j<num_particles; ++j)
			{
				/*state* current =
					&local_particle_data[i*num_particles+j];*/
				printf("%2i ", i*num_particles+j);
				print_particle(local_particle_data, i*num_particles+j, num_particles, num_blocks);
				printf("| w: %.32f\n", local_particle_weights[i*num_particles+j]);

			}
			printf("-------------------------\n");
		}
#endif /* DEBUG */

#ifdef TIMING_HOST
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */

		block_sort<<< num_blocks, num_particles/2>>>(
				d_particle_data_tmp,
				d_particle_weights,
				d_particle_data,
				num_particles,
				thrust::raw_pointer_cast(&d_local_winner_weights[0]));
		e = cudaGetLastError();
		if (e != cudaSuccess) {
			printf("failure in 'block_sort' kernel call.\n%s\n",
				cudaGetErrorString(e));
			exit(1);
		}
		CUDA_CALL(cudaThreadSynchronize());

#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s4 = time_diff(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		/*CUDA_CALL(cudaMemcpy(
			local_particle_data,
			d_particle_data_tmp,
			sizeof(state)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));*/
		CUDA_CALL(cudaMemcpy(local_particle_data.angles, d_particle_data_tmp.angles, sizeof(float)*NUM_ANGLES*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.x, d_particle_data_tmp.x, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.y, d_particle_data_tmp.y, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.vX, d_particle_data_tmp.vX, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.vY, d_particle_data_tmp.vY, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));

		CUDA_CALL(cudaMemcpy(
			local_particle_weights,
			d_particle_weights,
			sizeof(float)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));

		printf("BLOCK_SORT\n");
		for (int i=0; i<num_blocks; ++i)
		{
			printf("block %i\n", i);
			for (int j=0; j<num_particles; ++j)
			{
				/*state* current =
					&local_particle_data[i*num_particles+j];*/
				printf("%2i ", i*num_particles+j);
				print_particle(local_particle_data, i*num_particles+j, num_particles, num_blocks);
				printf("| w: %.32f\n", local_particle_weights[i*num_particles+j]);

			}
			printf("-------------------------\n");
		}
#endif /* DEBUG */

#ifdef TIMING_HOST
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */

		int global_winner_index = thrust::max_element(
				d_local_winner_weights.begin(),
				d_local_winner_weights.end())-
			d_local_winner_weights.begin();

/*
		get_particle_max<<<1, num_blocks>>>(d_particle_weights, d_global_winner_index, num_particles, num_blocks);
		e = cudaGetLastError();
		if (e != cudaSuccess) {
			printf("failure in 'get_particle_max' kernel call.\n%s\n",
				cudaGetErrorString(e));
			exit(1);
		}
		CUDA_CALL(cudaThreadSynchronize());
*/

#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s5 = time_diff(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG_ESTIMATE
		//int global_winner_index;
		float lpww;
		state lpw;
		/*CUDA_CALL(cudaMemcpy(
			&global_winner_index,
			d_global_winner_index,
			sizeof(int),
			cudaMemcpyDeviceToHost));*/
		/*CUDA_CALL(cudaMemcpy(
			&lpw,
			&d_particle_data_tmp[global_winner_index*num_particles],
			sizeof(state),
			cudaMemcpyDeviceToHost));*/
		for (int i=0; i < NUM_ANGLES; ++i)
		{
			CUDA_CALL(cudaMemcpy(&lpw.angles[i], &d_particle_data_tmp.angles[(i*num_particles*num_blocks)+(global_winner_index*num_particles)], sizeof(float), cudaMemcpyDeviceToHost));
		}
		CUDA_CALL(cudaMemcpy(&lpw.x, &d_particle_data_tmp.x[global_winner_index*num_particles], sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(&lpw.y, &d_particle_data_tmp.y[global_winner_index*num_particles], sizeof(float), cudaMemcpyDeviceToHost));

		CUDA_CALL(cudaMemcpy(
			&lpww,
			&d_particle_weights[global_winner_index*num_particles],
			sizeof(float),
			cudaMemcpyDeviceToHost));

		float est_error = estimate_error(lpw, actual_state);
		error_sum += est_error;
#ifndef GLOBAL_ONLY

#ifdef DEBUG_TRACE
		print_estimate(lpw, lpww, est_error, actual_state);
#else
		printf("EST ");
		print_particle(lpw);
		printf(" w: %.32f\n", lpww);
		printf("ACT ");
		print_particle(actual_state);
		printf("| err: %f\n\n", est_error);
#endif /* DEBUG_TRACE */
#endif /* GLOBAL_ONLY */
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */

		if (num_transfer > 0)
		{
			exchange_ring<<< num_blocks, num_transfer>>>(d_particle_data_tmp, d_particle_weights, num_particles, num_blocks);
			//exchange_2dtorus<<< num_blocks, num_transfer>>>(d_particle_data_tmp, d_particle_weights, num_particles, num_transfer, griddimx, griddimy);
			e = cudaGetLastError();
			if (e != cudaSuccess) {
				printf("failure in 'exchange_ring' kernel call.\n%s\n",
						cudaGetErrorString(e));
				exit(1);
			}
			CUDA_CALL(cudaThreadSynchronize());
		}

#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s6 = time_diff(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		/*CUDA_CALL(cudaMemcpy(
			local_particle_data,
			d_particle_data_tmp,
			sizeof(state)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));*/
		CUDA_CALL(cudaMemcpy(local_particle_data.angles, d_particle_data_tmp.angles, sizeof(float)*NUM_ANGLES*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.x, d_particle_data_tmp.x, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.y, d_particle_data_tmp.y, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.vX, d_particle_data_tmp.vX, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.vY, d_particle_data_tmp.vY, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));

		CUDA_CALL(cudaMemcpy(
			local_particle_weights,
			d_particle_weights,
			sizeof(float)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));

		printf("EXCHANGE\n");
		for (int i=0; i<num_blocks; ++i)
		{
			printf("block %i\n", i);
			for (int j=0; j<num_particles; ++j)
			{
				/*state* current =
					&local_particle_data[i*num_particles+j];*/
				printf("%2i ", i*num_particles+j);
				print_particle(local_particle_data, i*num_particles+j, num_particles, num_blocks);
				printf("| w: %.32f\n", local_particle_weights[i*num_particles+j]);

			}
			printf("-------------------------\n");
		}
#endif /* DEBUG */

#ifdef TIMING_HOST
		gettimeofday(&t1, NULL);
#endif /* TIMING_HOST */

		resampling<<< num_blocks, num_particles >>>(
			d_particle_data,
			d_particle_data_tmp,
			d_particle_weights,
			d_uniform_random,
			num_particles,
			resampling_freq);

		e = cudaGetLastError();
		if (e != cudaSuccess) {
			printf("failure in 'resampling' kernel call.\n%s\n", cudaGetErrorString(e));
			exit(1);
		}

		CUDA_CALL(cudaThreadSynchronize());

#ifdef TIMING_HOST
		gettimeofday(&t2, NULL);
		const int64_t s7 = time_diff(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		printf("RESAMPLING\n");
		/*CUDA_CALL(cudaMemcpy(
			local_particle_data,
			d_particle_data,
			sizeof(state)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));*/
		CUDA_CALL(cudaMemcpy(local_particle_data.angles, d_particle_data.angles, sizeof(float)*NUM_ANGLES*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.x, d_particle_data.x, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.y, d_particle_data.y, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.vX, d_particle_data.vX, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(local_particle_data.vY, d_particle_data.vY, sizeof(float)*num_blocks*num_particles, cudaMemcpyDeviceToHost));

		CUDA_CALL(cudaMemcpy(
			local_particle_weights,
			d_particle_weights,
			sizeof(float)*num_blocks*num_particles,
			cudaMemcpyDeviceToHost));

		for (int i=0; i<num_blocks; ++i)
		{
			printf("block %i\n", i);
			for (int j=0; j<num_particles; ++j)
			{
				/*state* current =
					&local_particle_data[i*num_particles+j];*/
				printf("%2i ", i*num_particles+j);
				print_particle(local_particle_data, i*num_particles+j, num_particles, num_blocks);
				printf("| w: %.32f\n", local_particle_weights[i*num_particles+j]);

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
				"%3ld %4ld %4ld %4ld | %ld\n",
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
				s6,
				s7,
				s1+s2+s3+s4+s5+s6+s7);//time_diff(s6,s1));
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
		s6_total += s6;
		s7_total += s7;
#endif /* TIMING_HOST */

		sample_count++;
	}

#ifdef DEBUG_ESTIMATE
#ifndef DEBUG_TRACE
	printf("%d %d %d %.16f\n", num_blocks, num_particles, num_transfer, error_sum/sample_count);
#endif /* DEBUG_TRACE */
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
		printf("%d %d %d %ld %ld %ld "
#ifdef TIMING_DEVICE
				"[%ld %ld %ld] "
#endif /* TIMING_DEVICE */
				"%ld %ld %ld %ld %ld\n",
				num_blocks,
				num_particles,
				num_transfer,
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
				s6_total/sample_count,
				s7_total/sample_count,
				(s1_total+s2_total+s3_total+s4_total+s5_total+s6_total+s7_total)/sample_count);
#endif /* TIMING_HOST */

#ifdef DEBUG
	free(local_particle_data.angles);
	free(local_particle_data.x);
	free(local_particle_data.y);
	free(local_particle_data.vX);
	free(local_particle_data.vY);
	//free(local_particle_data_tmp);
	//free(local_particle_weights);
#endif /* DEBUG */

#ifdef MTGP
	CUDA_CALL(cudaFree(d_mtgp_status));
	CUDA_CALL(cudaFree(d_random));
#else /* !MTGP */
	CURAND_CALL(curandDestroyGenerator(generator));
	CUDA_CALL(cudaFree(d_normal_random));
	CUDA_CALL(cudaFree(d_uniform_random));
#endif /* MTGP */
	CUDA_CALL(cudaFree(d_particle_data.angles));
	CUDA_CALL(cudaFree(d_particle_data.x));
	CUDA_CALL(cudaFree(d_particle_data.y));
	CUDA_CALL(cudaFree(d_particle_data.vX));
	CUDA_CALL(cudaFree(d_particle_data.vY));

	CUDA_CALL(cudaFree(d_particle_data_tmp.angles));
	CUDA_CALL(cudaFree(d_particle_data_tmp.x));
	CUDA_CALL(cudaFree(d_particle_data_tmp.y));
	CUDA_CALL(cudaFree(d_particle_data_tmp.vX));
	CUDA_CALL(cudaFree(d_particle_data_tmp.vY));

	CUDA_CALL(cudaFree(d_particle_weights));
	//CUDA_CALL(cudaFree(d_global_winner_index));

	fclose(input_file);

	return 0;
}

