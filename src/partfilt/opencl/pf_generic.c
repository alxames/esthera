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
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <time.h>
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW 4 /* since Linux 2.6.28, but not in glibc headers in Ubuntu 10.04 */
#endif
#include <unistd.h>
#include <math.h>
#include <CL/cl.h>

//#include <clpp/clpp.h>
//#include <clpp/clppSort.h>

#include <pf_model.h>
#include <pf_model_host.h>

#ifdef PRNG_MTGP
#include <mtgp-util.h>
#include <mtgp32-fast.h>
#ifdef PRNG_MTGP_TEX
#include <mtgp32-opencl-tex.h>
#else /* !PRNG_MTGP_TEX */
#include <mtgp32-opencl.h>
#endif /* PRNG_MTGP_TEX */
#else /* !PRNG_MTGP */
#ifdef PRNG_CURAND
#include <curand.h>
#else /* !PRNG_CURAND */
#ifdef PRNG_MWC64X
#endif /* PRNG_MWC64X */
#endif /* PRNG_CURAND */
#endif /* PRNG_MTGP */

#ifdef PRNG_MTGP
const int MTGP_WORKGROUPS = 96;
const int BOXMULLER_WORKGROUPS = 256;
extern mtgp32_params_fast_t mtgp32dc_params_fast_11213[];
void make_kernel_data32(cl_command_queue commands, cl_mem d_status, mtgp32_params_fast_t params[], const uint32_t seed, int block_num);
void make_constant(cl_command_queue commands, cl_mem d_params, const mtgp32_params_fast_t params[], int block_num);
#ifdef PRNG_MTGP_TEX
void make_texture(cl_command_queue commands, cl_mem d_tex_param, cl_mem d_tex_single_temper, const mtgp32_params_fast_t params[], int block_num);
#endif /* PRNG_MTGP_TEX */
#else /* !PRNG_MTGP */
#ifdef PRNG_CURAND
#else /* !PRNG_CURAND */
#ifdef PRNG_MWC64X
const size_t MWC64X_LOCAL_WORK_SIZE = 64;
const int MWC64X_SAMPLES_PER_WORK_ITEM = 2048;
#endif /* PRNG_MWC64X */
#endif /* PRNG_CURAND */
#endif /* PRNG_MTGP */

const int REDUCE_WORK_ITEMS = 128;

//const int MAX_NUM_PARTICLES = 1024;
//const int MAX_NUM_BLOCKS = 65536;
//const int MAX_NUM_TRANSFER = MAX_NUM_PARTICLES/2;

//extern const int NUM_STATE_VARIABLES;

#define CLERR \
	if (clerr != CL_SUCCESS) { \
		printf("opencl error %d,\t file: %s   line: %d\n", clerr,__FILE__,__LINE__); \
		exit(-1); \
	}

cl_platform_id   platform_id;
cl_device_id     device_id; /* compute device id */
cl_context       context;   /* compute context */
cl_command_queue commands;  /* compute command queue */
cl_program       program;

cl_kernel        sampling_importance_kernel;
cl_kernel        block_sort_kernel;
cl_kernel        exchange_ring_kernel;
cl_kernel        resampling_kernel;
cl_kernel        reduce_kernel;
#ifdef PRNG_MTGP
cl_kernel        mtgp32_single_kernel;
cl_kernel        boxmuller_kernel;
#else /* !PRNG_MTGP */
#ifdef PRNG_CURAND
#else /* !PRNG_CURAND */
#ifdef PRNG_MWC64X
cl_kernel        mwc64x_init_kernel;
cl_kernel        mwc64x_normal_rand_kernel;
cl_kernel        mwc64x_uniform_rand_kernel;
#endif /* PRNG_MWC64X */
#endif /* PRNG_CURAND */
#endif /* PRNG_MTGP */

#ifdef TIMING_HOST
static inline int64_t time_diff(const struct timeval t1, const struct timeval t2)
{
	return ((int64_t)t1.tv_sec-t2.tv_sec) * 1000000 + ((int64_t)t1.tv_usec-t2.tv_usec);
}

static inline int64_t time_diff_nano(const struct timespec t1, const struct timespec t2)
{
	return ((int64_t)t1.tv_sec-t2.tv_sec) * 1000000 + ((int64_t)(t1.tv_nsec-t2.tv_nsec)/1000);
}
#endif /* TIMING_HOST */

static size_t toMultipleOf(size_t num, size_t base)
{
	return base * ((num / base) + (((num % base) == 0) ? 0 : 1));
}

static size_t div_ceil(size_t num, size_t base)
{
	return (num / base) + (((num % base) == 0) ? 0 : 1);
}

static int to_multiple_int(int num, int base)
{
	return base * ((num / base) + (((num % base) == 0) ? 0 : 1));
}

#ifdef DEBUG
static void print_device_particle(cl_mem angles1, cl_mem angles2, cl_mem pos, cl_mem weights, const size_t num_particles, const size_t num_blocks)
{
	state_data local_particle;
	float* l_weights;
	cl_int clerr;

	local_particle.angles1 = (cl_float4*) malloc(sizeof(cl_float4) * num_particles * num_blocks);
	local_particle.angles2 =  (cl_float*) malloc(sizeof (cl_float) * num_particles * num_blocks);
	local_particle.pos     = (cl_float4*) malloc(sizeof(cl_float4) * num_particles * num_blocks);
	l_weights              =     (float*) malloc(sizeof    (float) * num_particles * num_blocks);

	clerr = clEnqueueReadBuffer(commands, angles1, CL_TRUE, 0, sizeof(cl_float4)*num_particles*num_blocks, local_particle.angles1, 0, NULL, NULL); CLERR;
	clerr = clEnqueueReadBuffer(commands, angles2, CL_TRUE, 0,  sizeof(cl_float)*num_particles*num_blocks, local_particle.angles2, 0, NULL, NULL); CLERR;
	clerr = clEnqueueReadBuffer(commands,     pos, CL_TRUE, 0, sizeof(cl_float4)*num_particles*num_blocks,     local_particle.pos, 0, NULL, NULL); CLERR;
	clerr = clEnqueueReadBuffer(commands, weights, CL_TRUE, 0,     sizeof(float)*num_particles*num_blocks,              l_weights, 0, NULL, NULL); CLERR;

	for (unsigned int i=0; i < num_blocks; ++i)
	{
		printf(">block %i\n", i);
		for (unsigned int j=0; j < num_particles; ++j)
		{
			const int index = i*num_particles + j;

			printf("%3i: (%.8f) %.8f %.8f %.8f %.8f [%.8f %.8f %.8f %.8f %.8f]\n",
					index,
					l_weights[index],
					local_particle.pos[index].x,
					local_particle.pos[index].y,
					local_particle.pos[index].z,
					local_particle.pos[index].w,
					local_particle.angles1[index].x,
					local_particle.angles1[index].y,
					local_particle.angles1[index].z,
					local_particle.angles1[index].w,
					local_particle.angles2[index]);
		}
		printf("-------------------------\n");
	}

	free(local_particle.angles1);
	free(local_particle.angles2);
	free(local_particle.pos);
	free(l_weights);
}
#endif /* DEBUG */

#ifdef PRNG_MTGP
static void generate_normal_random(
			cl_mem d_normal_random,
			cl_mem d_mtgp_status,
			cl_mem d_mtgp_param,
#ifdef PRNG_MTGP_TEX
			cl_mem d_tex_param,
			cl_mem d_tex_single_temper,
#endif /* PRNG_MTGP_TEX */
			const int num_random,
			const int num_normal_random)
{
	cl_int clerr;
	cl_event ev1, ev2;
	size_t global_work_size;
	size_t local_work_size;
	int size;

	global_work_size = THREAD_NUM * MTGP_WORKGROUPS;
	local_work_size = THREAD_NUM;
	size = num_random / MTGP_WORKGROUPS;

	clerr = clSetKernelArg(mtgp32_single_kernel, 0, sizeof(cl_mem), &d_mtgp_status); CLERR;
	clerr = clSetKernelArg(mtgp32_single_kernel, 1, sizeof(cl_mem), &d_normal_random); CLERR;
	clerr = clSetKernelArg(mtgp32_single_kernel, 2, sizeof(cl_uint) * LARGE_SIZE, NULL); CLERR;
	clerr = clSetKernelArg(mtgp32_single_kernel, 3, sizeof(cl_mem), &d_mtgp_param); CLERR;
#ifdef PRNG_MTGP_TEX
	clerr = clSetKernelArg(mtgp32_single_kernel, 4, sizeof(cl_mem), &d_tex_param); CLERR;
	clerr = clSetKernelArg(mtgp32_single_kernel, 5, sizeof(cl_mem), &d_tex_single_temper); CLERR;
	clerr = clSetKernelArg(mtgp32_single_kernel, 6, sizeof(cl_int), &size); CLERR;
#else /* !PRNG_MTGP_TEX */
	clerr = clSetKernelArg(mtgp32_single_kernel, 4, sizeof(cl_int), &size); CLERR;
#endif /* PRNG_MTGP_TEX */
	clerr = clEnqueueNDRangeKernel(commands, mtgp32_single_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev1); CLERR;

	global_work_size = to_multiple_int(num_normal_random, 2*BOXMULLER_WORKGROUPS) / 2;
	local_work_size = BOXMULLER_WORKGROUPS;

	clerr = clSetKernelArg(boxmuller_kernel, 0, sizeof(cl_mem), &d_normal_random); CLERR;
	clerr = clSetKernelArg(boxmuller_kernel, 1, sizeof(cl_int), &num_normal_random); CLERR;
	clerr = clEnqueueNDRangeKernel(commands, boxmuller_kernel, 1, NULL, &global_work_size, &local_work_size, 1, &ev1, &ev2); CLERR;
	clerr = clWaitForEvents(1, &ev2); CLERR;
}
#endif /* PRNG_MTGP */

static uint64_t get_seed() {
#ifdef DETERMINISTIC
	return 0x11111DEAF000000DULL;
#else
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_usec;
#endif
}

int particle_filter(
		const unsigned int num_particles,
		const unsigned int num_blocks,
		const unsigned int num_transfer,
		const float resampling_threshold,
		const char* input_file_str)
{
	/* for 2d torus */
/*
	const int griddimx = exp2(floor(log2(sqrtf(num_blocks))));
	const int griddimy = num_blocks/griddimx;
*/
	cl_int clerr;
	FILE* const input_file = fopen(input_file_str, "r");

	if (input_file == NULL)
	{
		fprintf(stderr, "could not open %s\n", input_file_str);
		exit(1);
	}


	const int num_normal_random = num_particles * num_blocks * NUM_STATE_VARIABLES;
#ifdef VALIAS_RESAMPLING
#  ifdef ADAPTIVE_RESAMPLING
	const int num_uniform_random = 2*num_particles * num_blocks;
#  else
	const int num_uniform_random = (2*num_particles+1) * num_blocks;
#  endif /* ADAPTIVE_RESAMPLING */
#else
	const int num_uniform_random = (num_particles+1) * num_blocks;
#endif

#ifdef PRNG_MTGP
	const uint64_t seed = get_seed();

	/* mtgp_num_random should be multiple of LARGE_SIZE * MTGP_GRIDSIZE */
	const int mtgp_num_random = to_multiple_int(num_normal_random + num_uniform_random, LARGE_SIZE * MTGP_WORKGROUPS);
	cl_mem d_mtgp_status = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(struct mtgp32_kernel_status_t) * MTGP_WORKGROUPS, NULL, &clerr); CLERR;
	cl_mem d_mtgp_param = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(struct mtgp32_param_t), NULL, &clerr); CLERR;

#ifdef PRNG_MTGP_TEX
	cl_image_format image_format;
	image_format.image_channel_order = CL_R;
	image_format.image_channel_data_type = CL_UNSIGNED_INT32;
	cl_mem d_tex_param         = clCreateImage2D(context, CL_MEM_READ_ONLY, &image_format, MTGP_WORKGROUPS, TBL_SIZE, 0, NULL, &clerr); CLERR;
	cl_mem d_tex_single_temper = clCreateImage2D(context, CL_MEM_READ_ONLY, &image_format, MTGP_WORKGROUPS, TBL_SIZE, 0, NULL, &clerr); CLERR;
#endif /* PRNG_MTGP_TEX */
	cl_mem d_normal_random = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * mtgp_num_random, NULL, &clerr); CLERR;
	cl_mem d_uniform_random = d_normal_random;
	make_constant(commands, d_mtgp_param, MTGPDC_PARAM_TABLE, MTGP_WORKGROUPS);
#ifdef PRNG_MTGP_TEX
	make_texture(commands, d_tex_param,/* d_tex_temper,*/ d_tex_single_temper, MTGPDC_PARAM_TABLE, MTGP_WORKGROUPS);
#endif /* PRNG_MTGP_TEX */
	make_kernel_data32(commands, d_mtgp_status, MTGPDC_PARAM_TABLE, seed, MTGP_WORKGROUPS);

#else /* !PRNG_MTGP */
#ifdef PRNG_CURAND
	const uint64_t seed = get_seed();

	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, seed);

	float* d_normal_random;
	float* d_uniform_random;
	cudaMalloc((void**)&d_normal_random,  sizeof(float) * num_normal_random);
	cudaMalloc((void**)&d_uniform_random, sizeof(float) * num_uniform_random);
#else /* !PRNG_CURAND */
#ifdef PRNG_MWC64X
	size_t rand_normal_global_work_size  = div_ceil(num_normal_random,  MWC64X_SAMPLES_PER_WORK_ITEM*MWC64X_LOCAL_WORK_SIZE) * MWC64X_LOCAL_WORK_SIZE;
	size_t rand_uniform_global_work_size = div_ceil(num_uniform_random, MWC64X_SAMPLES_PER_WORK_ITEM*MWC64X_LOCAL_WORK_SIZE) * MWC64X_LOCAL_WORK_SIZE;

	size_t mwc64x_num_normal  = rand_normal_global_work_size  * MWC64X_SAMPLES_PER_WORK_ITEM;
	size_t mwc64x_num_uniform = rand_uniform_global_work_size * MWC64X_SAMPLES_PER_WORK_ITEM;

	const cl_ulong seed = get_seed();

	cl_mem d_normal_random  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * mwc64x_num_normal,  NULL, &clerr); CLERR;
	cl_mem d_uniform_random = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * mwc64x_num_uniform, NULL, &clerr); CLERR;

	cl_ulong mwc64x_normal_skip  = seed;
	cl_ulong mwc64x_uniform_skip = seed;
#endif /* PRNG_MWC64X */
#endif /* PRNG_CURAND */
#endif /* PRNG_MTGP */

	/* particles */
//	state_data d_particle_data;
	cl_mem angles1, angles2, pos;
	{
		angles1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * num_particles * num_blocks, NULL, &clerr); CLERR;
		angles2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)  * num_particles * num_blocks, NULL, &clerr); CLERR;
		pos     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * num_particles * num_blocks, NULL, &clerr); CLERR;

		state_data init_particle_data;// = (state_data*)malloc(sizeof(state_data));

		init_particle_data.angles1 = (cl_float4*)malloc(sizeof(cl_float4) * num_particles * num_blocks);
		init_particle_data.angles2 =  (cl_float*)malloc( sizeof(cl_float) * num_particles * num_blocks);
		init_particle_data.pos     = (cl_float4*)malloc(sizeof(cl_float4) * num_particles * num_blocks);

		for (int i=0; i < (num_blocks*num_particles); ++i)
		{
			//init_particle_data[i] = init_state;
			for (int j=0; j < NUM_ANGLES; ++j)
			{
				if      (j==0){init_particle_data.angles1[i].x = 0.75f * M_PI;}
				else if (j==1){init_particle_data.angles1[i].y = 0.75f * M_PI;}
				else if (j==2){init_particle_data.angles1[i].z = 0.75f * M_PI;}
				else if (j==3){init_particle_data.angles1[i].w = 0.75f * M_PI;}
				else if (j==4){init_particle_data.angles2[i]   = 0.75f * M_PI;}
			}
			init_particle_data.pos[i].x = 7.0f;
			init_particle_data.pos[i].y = 0.0f;
			init_particle_data.pos[i].z = 0.0f;
			init_particle_data.pos[i].w = 0.0f;
		}
		clerr = clEnqueueWriteBuffer(commands, angles1, CL_TRUE, 0, sizeof(cl_float4)*num_particles*num_blocks, init_particle_data.angles1, 0, NULL, NULL); CLERR;
		clerr = clEnqueueWriteBuffer(commands, angles2, CL_TRUE, 0, sizeof(cl_float) *num_particles*num_blocks, init_particle_data.angles2, 0, NULL, NULL); CLERR;
		clerr = clEnqueueWriteBuffer(commands, pos,     CL_TRUE, 0, sizeof(cl_float4)*num_particles*num_blocks, init_particle_data.pos,     0, NULL, NULL); CLERR;

		free(init_particle_data.angles1);
		free(init_particle_data.angles2);
		free(init_particle_data.pos);

	}

//	state_data d_particle_data_tmp;
	cl_mem tmp_angles1, tmp_angles2, tmp_pos;
	{
		tmp_angles1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * num_particles * num_blocks, NULL, &clerr); CLERR;
		tmp_angles2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)  * num_particles * num_blocks, NULL, &clerr); CLERR;
		tmp_pos     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * num_particles * num_blocks, NULL, &clerr); CLERR;
	}

	/* particle weights */
	cl_mem d_particle_weights = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_particles * num_blocks, NULL, &clerr); CLERR;

	cl_mem d_local_winner_weights = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * num_blocks, NULL, &clerr); CLERR;
	cl_mem d_global_winner_weight = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2),   NULL, &clerr); CLERR;
	cl_mem d_control_data         = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(control),     NULL, &clerr); CLERR;
	cl_mem d_measurement_data     = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(measurement), NULL, &clerr); CLERR;
//	clppContext clpp_context;
//	clpp_context.setup(platform_id, device_id, context, commands);

//	clppSort* clpp_sort = clpp::createBestSortKV(&clpp_context, num_blocks, 32);
//	clpp_sort->pushCLDatas(d_local_winner_weights, num_blocks);
	//sort_init(program, device_id, context, commands, 32, d_local_winner_weights, num_blocks);
	//int* d_global_winner_index;
	//CUDA_CALL(cudaMalloc(&d_global_winner_index, sizeof(int)));

	//init_particles(initial_state, d_particle_data, num_blocks, num_particles);

	/* sensor/control input */
	control control_data;
	measurement measurement_data;
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
	local_particle_data.angles1 = (cl_float4*)malloc(sizeof(cl_float4)*num_blocks*num_particles);
	local_particle_data.angles2 = (cl_float*) malloc(sizeof(cl_float) *num_blocks*num_particles);
	local_particle_data.pos     = (cl_float4*)malloc(sizeof(cl_float4)*num_blocks*num_particles);

	//state* local_particle_data     = (state*)malloc(sizeof(state)*num_blocks*num_particles);
	//state* local_particle_data_tmp = (state*)malloc(sizeof(state)*num_blocks*num_particles);
	float* local_particle_weights  = (float*)malloc(sizeof(float)*num_blocks*num_particles);
#endif /* DEBUG */

	clFinish(commands);

	while (read_trace(input_file, &measurement_data, &control_data, &actual_state, &dt) == 0)
	{
		cl_event event1, event2;
#ifdef TIMING_HOST
		struct timespec t1,t2;
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */


#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s1 = time_diff_nano(t2,t1);
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

#ifdef PRNG_MTGP
		generate_normal_random(
			d_normal_random,
			d_mtgp_status,
			d_mtgp_param,
#ifdef PRNG_MTGP_TEX
			d_tex_param,
			d_tex_single_temper,
#endif /* PRNG_MTGP_TEX */
			mtgp_num_random,
			num_normal_random);
#else /* !PRNG_MTGP */
#ifdef PRNG_CURAND
		curandGenerateNormal(generator, d_normal_random, num_normal_random, 0, 1);
		curandGenerateUniform(generator, d_uniform_random, num_uniform_random);
		cudaThreadSynchronize();
#else /* !PRNG_CURAND */
#ifdef PRNG_MWC64X
		cl_event events[2];

		clerr = clSetKernelArg(mwc64x_normal_rand_kernel, 0, sizeof(cl_uint),  &mwc64x_num_normal); CLERR;
		clerr = clSetKernelArg(mwc64x_normal_rand_kernel, 1, sizeof(cl_mem),   &d_normal_random); CLERR;
		clerr = clSetKernelArg(mwc64x_normal_rand_kernel, 2, sizeof(cl_ulong), &mwc64x_normal_skip); CLERR;
		clerr = clEnqueueNDRangeKernel(commands, mwc64x_normal_rand_kernel, 1, NULL, &rand_normal_global_work_size, &MWC64X_LOCAL_WORK_SIZE, 0, NULL, &events[0]); CLERR;

		clerr = clSetKernelArg(mwc64x_uniform_rand_kernel, 0, sizeof(cl_uint),  &mwc64x_num_uniform); CLERR;
		clerr = clSetKernelArg(mwc64x_uniform_rand_kernel, 1, sizeof(cl_mem),   &d_uniform_random); CLERR;
		clerr = clSetKernelArg(mwc64x_uniform_rand_kernel, 2, sizeof(cl_ulong), &mwc64x_uniform_skip); CLERR;
		clerr = clEnqueueNDRangeKernel(commands, mwc64x_uniform_rand_kernel, 1, NULL, &rand_uniform_global_work_size, &MWC64X_LOCAL_WORK_SIZE, 0, NULL, &events[1]); CLERR;
		clWaitForEvents(2, events);
		mwc64x_normal_skip  += mwc64x_num_normal;
		mwc64x_uniform_skip += mwc64x_num_uniform;
/*
		{
			float* local_normal = (float*) malloc(sizeof(float) * mwc64x_num_normal);

			clerr = clEnqueueReadBuffer(commands, d_normal_random, CL_TRUE, 0, sizeof(float)*mwc64x_num_normal, local_normal, 0, NULL, NULL); CLERR;

			for (int i=0; i < mwc64x_num_normal; i+=8)
			{
				fprintf(stdout, "%i (%i): %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n",
						i/8, i,
						local_normal[i],
						local_normal[i+1],
						local_normal[i+2],
						local_normal[i+3],
						local_normal[i+4],
						local_normal[i+5],
						local_normal[i+6],
						local_normal[i+7]);
			}

			free(local_normal);
		}
*/
#endif /* PRNG_MWC64X */
#endif /* PRNG_CURAND */
#endif /* PRNG_MTGP */

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s2 = time_diff_nano(t2,t1);
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

		clerr = clEnqueueWriteBuffer(commands, d_control_data,     CL_TRUE, 0, sizeof(control),     &control_data,     0, NULL, NULL); CLERR;
		clerr = clEnqueueWriteBuffer(commands, d_measurement_data, CL_TRUE, 0, sizeof(measurement), &measurement_data, 0, NULL, NULL); CLERR;

		clerr = clSetKernelArg(sampling_importance_kernel, 0, sizeof(cl_mem), &angles1); CLERR;
		clerr = clSetKernelArg(sampling_importance_kernel, 1, sizeof(cl_mem), &angles2); CLERR;
		clerr = clSetKernelArg(sampling_importance_kernel, 2, sizeof(cl_mem), &pos); CLERR;
		clerr = clSetKernelArg(sampling_importance_kernel, 3, sizeof(cl_mem), &d_particle_weights); CLERR;
		clerr = clSetKernelArg(sampling_importance_kernel, 4, sizeof(cl_mem), &d_control_data); CLERR;
		clerr = clSetKernelArg(sampling_importance_kernel, 5, sizeof(cl_mem), &d_measurement_data); CLERR;
		clerr = clSetKernelArg(sampling_importance_kernel, 6, sizeof(cl_mem), &d_normal_random); CLERR;
		clerr = clSetKernelArg(sampling_importance_kernel, 7, sizeof(float),  &dt); CLERR;

		size_t local_work_size = num_particles;
		size_t global_work_size = num_blocks * num_particles;

		clerr = clEnqueueNDRangeKernel(commands, sampling_importance_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &event1); CLERR;

#ifdef TIMING_HOST
		clWaitForEvents(1, &event1);
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s3 = time_diff_nano(t2,t1);
#endif /* TIMING_HOST */

#ifdef TIMING_DEVICE
#ifdef TIMING_HOST
		clock_t lc[NUM_CLOCKS];
		cudaMemcpyFromSymbol(
			lc,
			d_clocks,
			sizeof(clock_t)*NUM_CLOCKS,
			0,
			cudaMemcpyDeviceToHost);

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
		printf("%i: SAMPLING\n", sample_count);
		print_device_particle(angles1, angles2, pos, d_particle_weights, num_particles, num_blocks);
#endif /* DEBUG */

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

		clerr = clSetKernelArg(block_sort_kernel, 0, sizeof(cl_mem), &tmp_angles1); CLERR;
		clerr = clSetKernelArg(block_sort_kernel, 1, sizeof(cl_mem), &tmp_angles2); CLERR;
		clerr = clSetKernelArg(block_sort_kernel, 2, sizeof(cl_mem), &tmp_pos); CLERR;
		clerr = clSetKernelArg(block_sort_kernel, 3, sizeof(cl_mem), &d_particle_weights); CLERR;
		clerr = clSetKernelArg(block_sort_kernel, 4, sizeof(cl_mem), &angles1); CLERR;
		clerr = clSetKernelArg(block_sort_kernel, 5, sizeof(cl_mem), &angles2); CLERR;
		clerr = clSetKernelArg(block_sort_kernel, 6, sizeof(cl_mem), &pos); CLERR;
		clerr = clSetKernelArg(block_sort_kernel, 7, sizeof(int),    &num_particles); CLERR;
		clerr = clSetKernelArg(block_sort_kernel, 8, sizeof(cl_mem), &d_local_winner_weights); CLERR;

		local_work_size = num_particles/2;
		global_work_size = num_blocks * num_particles / 2;

		clerr = clEnqueueNDRangeKernel(commands, block_sort_kernel, 1, NULL, &global_work_size, &local_work_size, 1, &event1, &event2); CLERR;

#ifdef TIMING_HOST
		clWaitForEvents(1, &event2);
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s4 = time_diff_nano(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		printf("%i: BLOCK SORT\n", sample_count);
		print_device_particle(tmp_angles1, tmp_angles2, tmp_pos, d_particle_weights, num_particles, num_blocks);
#endif /* DEBUG */

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

//		clpp_sort->sort();
//		clpp_sort->waitCompletion();
		size_t reduce_local_work_size = REDUCE_WORK_ITEMS;
		size_t reduce_global_work_size = REDUCE_WORK_ITEMS;

		clerr = clSetKernelArg(reduce_kernel, 0, sizeof(cl_mem), &d_local_winner_weights); CLERR;
		clerr = clSetKernelArg(reduce_kernel, 1, sizeof(cl_float2)*reduce_local_work_size, NULL); CLERR;
		clerr = clSetKernelArg(reduce_kernel, 2, sizeof(cl_int), &num_blocks); CLERR;
		clerr = clSetKernelArg(reduce_kernel, 3, sizeof(cl_mem), &d_global_winner_weight); CLERR;

		clerr = clEnqueueNDRangeKernel(commands, reduce_kernel, 1, NULL, &reduce_global_work_size, &reduce_local_work_size, 1, &event2, &event1); CLERR;
		cl_float2 global_winner;
		clerr = clEnqueueReadBuffer(commands, d_global_winner_weight, CL_TRUE, 0, sizeof(cl_float2), &global_winner, 1, &event1, NULL); CLERR;
		int global_winner_index = global_winner.y;


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
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s5 = time_diff_nano(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG_ESTIMATE
		//int global_winner_index;
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
		//for (int i=0; i < NUM_ANGLES; ++i)
		{
			float lpww;
			state lpw;

			cl_float4 local_angles1;
			clerr = clEnqueueReadBuffer(commands, tmp_angles1, CL_TRUE, global_winner_index*num_particles*sizeof(cl_float4), sizeof(cl_float4), &local_angles1, 0, NULL, NULL ); CLERR;
			lpw.angles[0] = local_angles1.x;
			lpw.angles[1] = local_angles1.y;
			lpw.angles[2] = local_angles1.z;
			lpw.angles[3] = local_angles1.w;

			clerr = clEnqueueReadBuffer(commands, tmp_angles2, CL_TRUE, global_winner_index*num_particles*sizeof(float), sizeof(float), &lpw.angles[4], 0, NULL, NULL ); CLERR;
			cl_float4 local_pos;
			clerr = clEnqueueReadBuffer(commands, tmp_pos, CL_TRUE, global_winner_index*num_particles*sizeof(cl_float4), sizeof(cl_float4), &local_pos, 0, NULL, NULL ); CLERR;
			lpw.x  = local_pos.x;
			lpw.y  = local_pos.y;
			lpw.vX = 0;
			lpw.vY = 0;

			clerr = clEnqueueReadBuffer(commands, d_particle_weights, CL_TRUE, global_winner_index*num_particles*sizeof(float), sizeof(float), &lpww, 0, NULL, NULL ); CLERR;

			float est_error = estimate_error(lpw, actual_state);
			error_sum += est_error;
#ifndef GLOBAL_ONLY

#ifdef DEBUG_TRACE
			print_estimate(lpw, lpww, est_error, actual_state);
#else
			printf("%i: ESTIMATE ::%f,%f %i\n", sample_count, global_winner.x, global_winner.y, global_winner_index);
			printf(">>>(%.8f) %.8f %.8f %.8f %.8f [%.8f %.8f %.8f %.8f %.8f]\n",
					lpww,
					lpw.x,
					lpw.y,
					lpw.vX,
					lpw.vY,
					lpw.angles[0],
					lpw.angles[1],
					lpw.angles[2],
					lpw.angles[3],
					lpw.angles[4]);
//			printf("EST ");
//			print_particle(lpw);
//			printf(" w: %.32f\n", lpww);
//			printf("ACT ");
//			print_particle(actual_state);
//			printf("| err: %f\n\n", est_error);
#endif /* DEBUG_TRACE */
#endif /* GLOBAL_ONLY */
		}
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

		if (num_transfer > 0)
		{
			clerr = clSetKernelArg(exchange_ring_kernel, 0, sizeof(cl_mem), &tmp_angles1); CLERR;
			clerr = clSetKernelArg(exchange_ring_kernel, 1, sizeof(cl_mem), &tmp_angles2); CLERR;
			clerr = clSetKernelArg(exchange_ring_kernel, 2, sizeof(cl_mem), &tmp_pos); CLERR;
			clerr = clSetKernelArg(exchange_ring_kernel, 3, sizeof(cl_mem), &d_particle_weights); CLERR;
			clerr = clSetKernelArg(exchange_ring_kernel, 4, sizeof(int),    &num_particles); CLERR;
			clerr = clSetKernelArg(exchange_ring_kernel, 5, sizeof(int),    &num_blocks); CLERR;

			local_work_size = num_transfer;
			global_work_size = num_blocks * num_transfer;

			clerr = clEnqueueNDRangeKernel(commands, exchange_ring_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &event1); CLERR;
		}

#ifdef TIMING_HOST
		clWaitForEvents(1, &event1);
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s6 = time_diff_nano(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		printf("%i: EXCHANGE\n", sample_count);
		print_device_particle(tmp_angles1, tmp_angles2, tmp_pos, d_particle_weights, num_particles, num_blocks);
#endif /* DEBUG */

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

		cl_uint aidx = 0;
		clerr = clSetKernelArg(resampling_kernel, aidx++, sizeof(cl_mem), &angles1); CLERR;
		clerr = clSetKernelArg(resampling_kernel, aidx++, sizeof(cl_mem), &angles2); CLERR;
		clerr = clSetKernelArg(resampling_kernel, aidx++, sizeof(cl_mem), &pos); CLERR;
		clerr = clSetKernelArg(resampling_kernel, aidx++, sizeof(cl_mem), &tmp_angles1); CLERR;
		clerr = clSetKernelArg(resampling_kernel, aidx++, sizeof(cl_mem), &tmp_angles2); CLERR;
		clerr = clSetKernelArg(resampling_kernel, aidx++, sizeof(cl_mem), &tmp_pos); CLERR;
		clerr = clSetKernelArg(resampling_kernel, aidx++, sizeof(cl_mem), &d_particle_weights); CLERR;
		clerr = clSetKernelArg(resampling_kernel, aidx++, sizeof(cl_mem), &d_uniform_random); CLERR;
		clerr = clSetKernelArg(resampling_kernel, aidx++, sizeof(int),    &num_particles); CLERR;
		clerr = clSetKernelArg(resampling_kernel, aidx++, sizeof(float),  &resampling_threshold); CLERR;
#ifdef PRNG_MTGP
		clerr = clSetKernelArg(resampling_kernel, aidx++, sizeof(cl_int), &num_normal_random); CLERR;
#endif /* PRNG_MTGP */

#ifdef VALIAS_RESAMPLING
		// __local memory
		clerr = clSetKernelArg(resampling_kernel, aidx++, num_particles * sizeof(float),   NULL); CLERR;  // prob (particle weights)
		clerr = clSetKernelArg(resampling_kernel, aidx++, num_particles * sizeof(cl_uint), NULL); CLERR; // alias
		clerr = clSetKernelArg(resampling_kernel, aidx++, num_particles * sizeof(cl_uint), NULL); CLERR; // scratch/scratchf; actually, sz: n*max(sz(ui),sz(float))
#endif

		local_work_size = num_particles;
		global_work_size = num_blocks * num_particles;

		clerr = clEnqueueNDRangeKernel(commands, resampling_kernel, 1, NULL, &global_work_size, &local_work_size, 1, &event1, &event2); CLERR;

#ifdef TIMING_HOST
		clWaitForEvents(1, &event2);
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s7 = time_diff_nano(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		printf("%i: RESAMPLING\n", sample_count);
		print_device_particle(angles1, angles2, pos, d_particle_weights, num_particles, num_blocks);
#endif /* DEBUG */

#ifdef TIMING_HOST
#ifndef GLOBAL_ONLY
		printf("%2ld %4ld %4ld "
#ifdef TIMING_DEVICE
				"(%3ld %3ld %3ld) "
#endif /* TIMING_DEVICE */
				"%3ld %4ld %4ld %4ld | %ld\n",
				s1,//time_diff_nano(s2,s1),//((double)time_diff_nano(s2,s1)*100)/time_diff_nano(s6,s1),
				s2,//time_diff_nano(s3,s2),//((double)time_diff_nano(s3,s2)*100)/time_diff_nano(s6,s1),
				s3,//time_diff_nano(s4,s3),//((double)time_diff_nano(s4,s3)*100)/time_diff_nano(s6,s1),
#ifdef TIMING_DEVICE
				s3_1,
				s3_2,
				s3_3,
#endif /* TIMING_DEVICE */
				s4,//time_diff_nano(s5,s4),//((double)time_diff_nano(s5,s4)*100)/time_diff_nano(s6,s1),
				s5,//time_diff_nano(s6,s5),//((double)time_diff_nano(s6,s5)*100)/time_diff_nano(s6,s1),
				s6,
				s7,
				s1+s2+s3+s4+s5+s6+s7);//time_diff_nano(s6,s1));
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

		clFinish(commands);
		sample_count++;
	}

	fclose(input_file);

#ifdef DEBUG_ESTIMATE
#ifndef DEBUG_TRACE
	printf("%d %d %f %d %.16f\n", num_blocks, num_particles, resampling_threshold, num_transfer, error_sum/sample_count);
#endif /* DEBUG_TRACE */
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
		printf("%d %d %f %d %ld %ld %ld "
#ifdef TIMING_DEVICE
				"[%ld %ld %ld] "
#endif /* TIMING_DEVICE */
				"%ld %ld %ld %ld %ld\n",
				num_blocks,
				num_particles,
				resampling_threshold,
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
	free(local_particle_data.angles1);
	free(local_particle_data.angles2);
	free(local_particle_data.pos);
	//free(local_particle_data_tmp);
	//free(local_particle_weights);
#endif /* DEBUG */

#ifdef PRNG_MTGP
	clReleaseMemObject(d_mtgp_status);
	clReleaseMemObject(d_mtgp_param);
#ifdef PRNG_MTGP_TEX
	clReleaseMemObject(d_tex_param);
	clReleaseMemObject(d_tex_single_temper);
#endif /* PRNG_MTGP_TEX */
	clReleaseMemObject(d_normal_random);
#else /* !PRNG_MTGP */
#ifdef PRNG_CURAND
	curandDestroyGenerator(generator);
	cudaFree(d_normal_random);
	cudaFree(d_uniform_random);
#else /* !PRNG_CURAND */
#ifdef PRNG_MWC64X
	clReleaseMemObject(d_normal_random);
	clReleaseMemObject(d_uniform_random);
#endif /* PRNG_MWC64X */
#endif /* PRNG_CURAND */
#endif /* PRNG_MTGP */
	clReleaseMemObject(angles1);
	clReleaseMemObject(angles2);
	clReleaseMemObject(pos);

	clReleaseMemObject(tmp_angles1);
	clReleaseMemObject(tmp_angles2);
	clReleaseMemObject(tmp_pos);

	clReleaseMemObject(d_particle_weights);
//	delete clpp_sort;
	clReleaseMemObject(d_local_winner_weights);
	clReleaseMemObject(d_global_winner_weight);
	clReleaseMemObject(d_control_data);
	clReleaseMemObject(d_measurement_data);

	return 0;
}

char* load_program_source(const char *filename)
{
	struct stat statbuf;
	FILE *fh;
	char *source;
	fh = fopen(filename, "r");
	if (NULL == fh)
	{
		printf("could not open: %s\n", filename);
		return NULL;
	}

	stat(filename, &statbuf);
	source = (char *) malloc(statbuf.st_size + 1);
	ssize_t nread = (ssize_t)fread(source, statbuf.st_size, 1, fh);
	if (nread != 1)
	{
		printf("could not read complete source file: %s\n", filename);
		return NULL;
	}
	source[statbuf.st_size] = '\0';

	return source;
}

static void print_platform(cl_platform_id platform_id)
{
	char str[1024];
	cl_int clerr;

	clerr = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(str), str, NULL);
	if (CL_SUCCESS != clerr) strncpy(str, "Unknown", 1024);
	printf("\tName:\t\t%s\n", str);

	clerr = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, sizeof(str), str, NULL);
	if (CL_SUCCESS != clerr) strncpy(str, "Unknown", 1024);
	printf("\tVendor:\t\t%s\n", str);

	clerr = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str), str, NULL);
	if (CL_SUCCESS != clerr) strncpy(str, "Unknown", 1024);
	printf("\tVersion:\t%s\n", str);

	clerr = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, sizeof(str), str, NULL);
	if (CL_SUCCESS != clerr) strncpy(str, "Unknown", 1024);
	printf("\tExtensions:\t%s\n", str);
}

static void print_device(cl_device_id device_id)
{
	char str[1024];
	cl_device_type device_type;
	cl_bool image_support;
	cl_int clerr;

	clerr = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(str), str, NULL);
	if (CL_SUCCESS != clerr) strncpy(str, "Unknown", 1024);
	printf("\tName:\t\t%s\n", str);

	clerr = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(str), str, NULL);
	if (CL_SUCCESS != clerr) strncpy(str, "Unknown", 1024);
	printf("\tVendor:\t\t%s\n", str);

	clerr = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
	if (CL_SUCCESS != clerr) strncpy(str, "Unknown", 1024);
	else 
	{
		unsigned int nprinted = 0;
		if (device_type & CL_DEVICE_TYPE_CPU)
		{
			strncpy(str + nprinted, "CPU ", 1024 - 1 - nprinted);
			nprinted += strlen("CPU ");
		}
		if (device_type & CL_DEVICE_TYPE_GPU)
		{
			strncpy(str + nprinted, "GPU ", 1024 - 1 - nprinted);
			nprinted += strlen("GPU ");
		}
		if (device_type & CL_DEVICE_TYPE_ACCELERATOR)
		{
			strncpy(str + nprinted, "Accelerator ", 1024 - 1 - nprinted);
			nprinted += strlen("Accelerator ");
		}
		/*if (device_type & CL_DEVICE_TYPE_CUSTOM) // only in OpenCL 1.2
		{
			strncpy(str + nprinted, "Custom ", 1024 - 1 - nprinted);
			nprinted += strlen("Custom ");
		}*/
		if (device_type & CL_DEVICE_TYPE_CPU)
		{
			strncpy(str + nprinted, "(Default) ", 1024 - 1 - nprinted);
			nprinted += strlen("(Default) ");
		}
	}
	printf("\tType(s):\t%s\n", str);

	clerr = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
	if (CL_SUCCESS != clerr) strncpy(str, "Unknown", 1024);
	else strncpy(str, image_support ? "yes" : "no", 1024);
	printf("\tImage support:\t%s\n", str);

	clerr = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(str), str, NULL);
	if (CL_SUCCESS != clerr) strncpy(str, "Unknown", 1024);
	printf("\tExtensions:\t%s\n", str);
}

static void print_platforms_and_devices(cl_platform_id *platform_ids, unsigned int num_platforms)
{
	cl_int clerr;

	printf("Detected %u OpenCL platform%s\n", num_platforms, num_platforms == 1 ? "" : "s");
	for (unsigned int p=0; p < num_platforms; ++p)
	{
		printf("Platform %u:\n", p);
		print_platform(platform_ids[p]);

		cl_device_id device_ids[32];
		cl_uint num_devices;
		const cl_device_type all_types = CL_DEVICE_TYPE_ALL/* | CL_DEVICE_TYPE_CUSTOM*/;
		clerr = clGetDeviceIDs(platform_ids[p], all_types, 32, device_ids, &num_devices);
		if (CL_SUCCESS != clerr)
		{
			printf("Failed to get devices for platform %u: %u\n", p, clerr);
			printf("\n");
			continue;
		}

		for (unsigned int d=0; d < num_devices; ++d)
		{
			printf("Device %u:%u:\n", p, d);
			print_device(device_ids[d]);
			printf("\n");
		}
	}
}

static inline void parse_range(const char* str, int* left, int* right)
{
	const char* pos = strchr(str,':');
	if (NULL != pos)
	{
		size_t len = strlen(str);
		char* l_str = (char*)malloc(len);
		memcpy(l_str, str, len);
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

static inline void parse_frange(const char* str, double* left, double* right)
{
	const char* pos = strchr(str,':');
	if (NULL != pos)
	{
		size_t len = strlen(str);
		char* l_str = (char*)malloc(len);
		memcpy(l_str, str, len);
		const char* r_str = pos + 1;
		l_str[pos-str] = '\0';
		*left  = atof(l_str);
		*right = atof(r_str);
		free(l_str);
	}
	else
	{
		const double val = atof(str);
		*left  = val;
		*right = val;
	}

}

int main(int argc, char* argv[])
{
	int num_transfer_start  = 1;
	int num_transfer_end    = 1;

	int num_blocks_start    = 16;
	int num_blocks_end      = 16;

	int num_particles_start = 256;
	int num_particles_end   = 256;

	int loop_count          = 1;

	double resampling_threshold_start = 1.0;
	double resampling_threshold_end   = 1.0;
	double resampling_inc             = 0.1;
#ifdef ADAPTIVE_RESAMPLING
	// For the adaptive resampling stategy, this theshold is cmp to the _reciprocal_ effective sampling size.
	// This means that the range is inverted, i.e. always: 0.0; never: 1.0 (but fp precision).
	resampling_threshold_start = 0.0;
	resampling_threshold_end = 0.0;
	resampling_inc = 0.01; // differences are small, so have a smaller default inc
#endif

	int device_gpu          = 1;

	int platform            = -1;
	int device;

	int verbose             = 0;

	int opt;

	while (-1 != (opt = getopt(argc, argv, "m:N:t:r:l:d:cv"))) {
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
				parse_frange(optarg, &resampling_threshold_start, &resampling_threshold_end);
				break;
			case 'l':
				loop_count = atoi(optarg);
				break;
			case 'd':
				if (sscanf(optarg, "%u:%u", &platform, &device) != 2)
				{
					fprintf(stderr, "Failed to parse -d option value.\n");
					exit(EXIT_FAILURE);
				}
				break;
			case 'c':
				device_gpu = 0;
				break;
			case 'v':
				verbose = 1;
				break;
			default: /* '?' */
				fprintf(stderr, "Usage: %s [-m #particles] [-N #cores] [-t #transfer] [-r resampling_threshold] [-l loop_count] [-d platform_id:device_id] [-c] [-v] input_file\n", argv[0]);
				exit(EXIT_FAILURE);
		}
	}

	if (optind >= argc)
	{
		fprintf(stderr, "no input file given\n");
		exit(EXIT_FAILURE);
	}

	const char *source_files[] = {
		"pf_generic.cl",
		"bitonic-sort.cl",
#ifdef VALIAS_RESAMPLING
		"resampling-valias.cl",
#endif
		"pf_model.cl",
#ifdef PRNG_MWC64X
		"mwc64x_kernel.cl",
#endif
#ifdef PRNG_MTGP
#ifdef PRNG_MTGP_TEX
		"mtgp/mtgp32-opencl-tex.cl",
#else
		"mtgp/mtgp32-opencl.cl",
#endif /* PRNG_MTGP_TEX */
#endif /* PRNG_MTGP */
		"reduce.cl"
	};
	const unsigned int num_sources = sizeof(source_files) / sizeof(char*);
	const char *buildOptions="-I../../models/robotarm/opencl -Imwc64x/cl -Imtgp -cl-single-precision-constant -Werror -cl-fast-relaxed-math"
#ifdef PRNG_MTGP
		" -DPRNG_MTGP"
#endif /* PRNG_MTGP */
#ifdef ADAPTIVE_RESAMPLING
		" -DADAPTIVE_RESAMPLING"
#endif
#if defined DEBUG || defined _DEBUG
		" -DDEBUG -cl-opt-disable -g"
#endif
		;
	const char *program_source[num_sources];

	cl_int clerr;
	cl_platform_id platform_ids[32];

	unsigned int num_platforms;

	clerr = clGetPlatformIDs(32, platform_ids, &num_platforms); 
	CLERR;
	if (verbose)
	{
		print_platforms_and_devices(platform_ids, num_platforms);
	}

	if (platform != -1)
	{
		platform_id = platform_ids[platform];
		cl_device_id device_ids[32];
		cl_uint num_devices;
		clerr = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 32, device_ids, &num_devices);
		device_id = device_ids[device];
	}

	for (unsigned int i=0; i < num_platforms; ++i)
	{
		if (platform != -1) break;

		clerr = clGetDeviceIDs(platform_ids[i], device_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
		if (CL_SUCCESS == clerr)
		{
			platform_id = platform_ids[i];
			break;
		}
		else if (CL_DEVICE_NOT_FOUND == clerr)
			continue;
		CLERR;

	}
	if (platform_id == NULL)
	{
		// clGetPlatformInfo() may return an error if platform_id=0 (implementation defined behavior), so exit now.
		// This happens e.g. if asking for GPU devices on a system whose first platforms has no GPUs.
		fprintf(stderr, "No %s devices found in the first OpenCL platform; exiting\n", device_gpu ? "GPU" : "CPU");
		exit(EXIT_FAILURE);
	}

	{
		char platform_name[1024];
		char platform_vendor[1024];
		char device_name[1024];

		clerr = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
		CLERR;

		clerr = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
		CLERR;

		clerr = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
		CLERR;

		printf("Platform name: %s\nPlatform vendor: %s\nDevice name: %s\n", platform_name, platform_vendor, device_name);
	}

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &clerr);
	CLERR;

	commands = clCreateCommandQueue(context, device_id, 0, &clerr);
	CLERR;

	for (unsigned int i=0; i < num_sources; ++i)
	{
		program_source[i] = load_program_source(source_files[i]);
	}
	program = clCreateProgramWithSource(context, num_sources, program_source, NULL, &clerr);
	CLERR;

	clerr = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
	//CLERR;

	size_t log_size;
	clerr = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	CLERR;

	char* build_log = (char*) malloc(log_size);
	clerr = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	CLERR;

	build_log[log_size-1] = '\0';
	printf("BUILD LOG %s\n", build_log);
	free(build_log);

	for (unsigned int i=0; i < num_sources; ++i)
	{
		free((void*)program_source[i]);
	}

	sampling_importance_kernel = clCreateKernel(program, "sampling_importance", &clerr);
	CLERR;

	block_sort_kernel = clCreateKernel(program, "block_sort", &clerr);
	CLERR;

	exchange_ring_kernel = clCreateKernel(program, "exchange_ring", &clerr);
	CLERR;

#ifdef VALIAS_RESAMPLING
	resampling_kernel = clCreateKernel(program, "resampling_valias", &clerr);
#else /* !VALIAS_RESAMPLING */
	resampling_kernel = clCreateKernel(program, "resampling", &clerr);
#endif /* VALIAS_RESAMPLING */
	CLERR;

#ifdef PRNG_MTGP
	mtgp32_single_kernel = clCreateKernel(program, "mtgp32_single_kernel", &clerr); CLERR;
	boxmuller_kernel = clCreateKernel(program, "boxmuller", &clerr); CLERR;
#else /* !PRNG_MTGP */
#ifdef PRNG_CURAND
#else /* !PRNG_CURAND */
#ifdef PRNG_MWC64X
	mwc64x_init_kernel = clCreateKernel(program, "rand_init", &clerr); CLERR;
	mwc64x_normal_rand_kernel = clCreateKernel(program, "normal_rand", &clerr); CLERR;
	mwc64x_uniform_rand_kernel = clCreateKernel(program, "uniform_rand_float", &clerr); CLERR;
#endif /* PRNG_MWC64X */
#endif /* PRNG_CURAND */
#endif /* PRNG_MTGP */

	reduce_kernel = clCreateKernel(program, "reduce", &clerr); CLERR;

#ifdef DEBUG_ESTIMATE
#ifdef DEBUG_TRACE
	printf("ex ey ew ee ax ay\n");
#else
	printf("N m r t e\n");
#endif /* DEBUG_TRACE */
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
	printf("N m r t t1 t2 t3 "
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
				for (double resampling_threshold=resampling_threshold_start; resampling_threshold <= resampling_threshold_end; resampling_threshold+=resampling_inc)
				{
					for (int i=0; i < loop_count; ++i)
					{


	if (
		num_particles > MAX_NUM_PARTICLES ||
		num_blocks    > MAX_NUM_BLOCKS ||
		num_transfer  > MAX_NUM_TRANSFER ||
		resampling_threshold < 0.0 ||
		resampling_threshold > 1.0)
	{
		fprintf(stderr, "exceeds maximum\n");
		exit(EXIT_FAILURE);
	}

//	if ((3*num_transfer) > num_particles)
//		continue;

	particle_filter(num_particles, num_blocks, num_transfer, (float)resampling_threshold, argv[optind]);


					}
				}
			}
		}
	}

	/*Close connection with devices*/
	clReleaseKernel(sampling_importance_kernel);
	clReleaseKernel(block_sort_kernel);
	clReleaseKernel(exchange_ring_kernel);
	clReleaseKernel(resampling_kernel);
#ifdef PRNG_MTGP
	clReleaseKernel(mtgp32_single_kernel);
	clReleaseKernel(boxmuller_kernel);
#else /* !PRNG_MTGP */
#ifdef PRNG_CURAND
#else /* !PRNG_CURAND */
#ifdef PRNG_MWC64X
	clReleaseKernel(mwc64x_init_kernel);
	clReleaseKernel(mwc64x_normal_rand_kernel);
	clReleaseKernel(mwc64x_uniform_rand_kernel);
#endif /* PRNG_MWC64X */
#endif /* PRNG_CURAND */
#endif /* PRNG_MTGP */
	clReleaseKernel(reduce_kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}

