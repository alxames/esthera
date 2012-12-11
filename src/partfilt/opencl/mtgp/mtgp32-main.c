/*
 * @file mtgp32-cuda.cu
 *
 * @brief Sample Program for CUDA 3.2 and 4.0
 *
 * MTGP32-11213
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>11213</sup>-1.
 *
 * This also generates single precision floating point numbers
 * uniformly distributed in the range [1, 2). (float r; 1.0 <= r < 2.0)
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
#include <unistd.h>
#include <CL/cl.h>

#include "mtgp-util.h"
#include "mtgp32-fast.h"
#include "mtgp32-opencl.h"

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

cl_kernel        mtgp32_uint32_kernel;
cl_kernel        mtgp32_single_kernel;

extern mtgp32_params_fast_t mtgp32dc_params_fast_11213[];
void make_kernel_data32(cl_command_queue commands, cl_mem d_status, mtgp32_params_fast_t params[], int block_num);
void make_constant(cl_command_queue commands, cl_mem d_params, const mtgp32_params_fast_t params[], int block_num);

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
	fread(source, statbuf.st_size, 1, fh);
	source[statbuf.st_size] = '\0';

	return source;
}

static inline int64_t time_diff(const struct timeval t1, const struct timeval t2)
{
	return ((int64_t)t1.tv_sec-t2.tv_sec) * 1000000 + ((int64_t)t1.tv_usec-t2.tv_usec);
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] d_status kernel I/O data.
 * @param[in] num_data number of data to be generated.
 */
void make_uint32_random(cl_mem d_status,
		cl_mem d_params,
		int num_data,
		int block_num) {
	cl_mem d_data;
	uint32_t* h_data;
	cl_int clerr;
	cl_event ev;
	size_t global_work_size;
	size_t local_work_size;
	int size;
	struct timeval t1, t2;

	printf("generating 32-bit unsigned random numbers.\n");
	d_data = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * num_data, NULL, &clerr); CLERR;
//	ccudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data);

	h_data = (uint32_t *) malloc(sizeof(uint32_t) * num_data);
	if (h_data == NULL) {
		printf("failure in allocating host memory for output data.\n");
		exit(1);
	}

	gettimeofday(&t1, NULL);
	global_work_size = THREAD_NUM * block_num;
	local_work_size = THREAD_NUM;
	size = num_data / block_num;

	clerr = clSetKernelArg(mtgp32_uint32_kernel, 0, sizeof(cl_mem), &d_status); CLERR;
	clerr = clSetKernelArg(mtgp32_uint32_kernel, 1, sizeof(cl_mem), &d_data); CLERR;
	clerr = clSetKernelArg(mtgp32_uint32_kernel, 2, sizeof(cl_uint) * LARGE_SIZE, NULL); CLERR;
	clerr = clSetKernelArg(mtgp32_uint32_kernel, 3, sizeof(cl_mem), &d_params); CLERR;
	clerr = clSetKernelArg(mtgp32_uint32_kernel, 4, sizeof(cl_int), &size); CLERR;
	clerr = clEnqueueNDRangeKernel(commands, mtgp32_uint32_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev); CLERR;

	clerr = clWaitForEvents(1, &ev); CLERR;
	gettimeofday(&t2, NULL);
	/* kernel call */
//	mtgp32_uint32_kernel<<< block_num, THREAD_NUM>>>(
//			d_status, d_data, num_data / block_num);
//	cudaThreadSynchronize();

	clerr = clEnqueueReadBuffer(commands, d_data, CL_TRUE, 0, sizeof(cl_uint) * num_data, h_data, 0, NULL, NULL); CLERR;
//	ccudaMemcpy(h_data, d_data, sizeof(uint32_t) * num_data, cudaMemcpyDeviceToHost);

	print_uint32_array(h_data, num_data, block_num);
	printf("generated numbers: %d\n", num_data);
	printf("Processing time: %lu (ms)\n", time_diff(t2, t1));
	printf("Samples per second: %E \n", num_data / (time_diff(t2, t1) * 0.000001));

	//free memories
	free(h_data);
	clerr = clReleaseMemObject(d_data); CLERR;
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] d_status kernel I/O data.
 * @param[in] num_data number of data to be generated.
 */
void make_single_random(cl_mem d_status,
		cl_mem d_params,
		int num_data,
		int block_num) {
	cl_mem d_data;
	float* h_data;
	cl_int clerr;
	cl_event ev;
	size_t global_work_size;
	size_t local_work_size;
	int size;
	struct timeval t1, t2;

	printf("generating single precision floating point random numbers.\n");
	d_data = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * num_data, NULL, &clerr); CLERR;
//	ccudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data);

	h_data = (float*) malloc(sizeof(float) * num_data);
	if (h_data == NULL) {
		printf("failure in allocating host memory for output data.\n");
		exit(1);
	}

	gettimeofday(&t1, NULL);
	global_work_size = THREAD_NUM * block_num;
	local_work_size = THREAD_NUM;
	size = num_data / block_num;

	clerr = clSetKernelArg(mtgp32_single_kernel, 0, sizeof(cl_mem), &d_status); CLERR;
	clerr = clSetKernelArg(mtgp32_single_kernel, 1, sizeof(cl_mem), &d_data); CLERR;
	clerr = clSetKernelArg(mtgp32_single_kernel, 2, sizeof(cl_uint) * LARGE_SIZE, NULL); CLERR;
	clerr = clSetKernelArg(mtgp32_single_kernel, 3, sizeof(cl_mem), &d_params); CLERR;
	clerr = clSetKernelArg(mtgp32_single_kernel, 4, sizeof(cl_int), &size); CLERR;
	clerr = clEnqueueNDRangeKernel(commands, mtgp32_single_kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &ev); CLERR;

	clerr = clWaitForEvents(1, &ev); CLERR;
	gettimeofday(&t2, NULL);
	/* kernel call */
//	mtgp32_uint32_kernel<<< block_num, THREAD_NUM>>>(
//			d_status, d_data, num_data / block_num);
//	cudaThreadSynchronize();

	clerr = clEnqueueReadBuffer(commands, d_data, CL_TRUE, 0, sizeof(cl_float) * num_data, h_data, 0, NULL, NULL); CLERR;
//	ccudaMemcpy(h_data, d_data, sizeof(uint32_t) * num_data, cudaMemcpyDeviceToHost);

	print_float_array(h_data, num_data, block_num);
	printf("generated numbers: %d\n", num_data);
	printf("Processing time: %lu (ms)\n", time_diff(t2, t1));
	printf("Samples per second: %E \n", num_data / (time_diff(t2, t1) * 0.000001));

	//free memories
	free(h_data);
	clerr = clReleaseMemObject(d_data); CLERR;
}

int main(int argc, char* argv[])
{
	int device_gpu = 1;

	const char *source_files[1] = {
		"mtgp32-opencl.cl"};
	const char *buildOptions="-I. -Werror";
	const char *program_source[1];

	cl_int clerr;
	cl_platform_id   platform_ids[32];

	unsigned int num_platforms;

	clerr = clGetPlatformIDs(32, platform_ids, &num_platforms); 
	CLERR;

	for (unsigned int i=0; i < num_platforms; ++i)
	{
		clerr = clGetDeviceIDs (platform_ids[i], device_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
		if (CL_SUCCESS == clerr)
		{
			platform_id = platform_ids[i];
			break;
		}
		else if (CL_DEVICE_NOT_FOUND == clerr)
			continue;
		CLERR;
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

	program_source[0] = load_program_source(source_files[0]);
	program = clCreateProgramWithSource(context, 1, program_source, NULL, &clerr);
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

	free((void*)program_source[0]);

	mtgp32_uint32_kernel = clCreateKernel(program, "mtgp32_uint32_kernel", &clerr); CLERR;
	mtgp32_single_kernel = clCreateKernel(program, "mtgp32_single_kernel", &clerr); CLERR;

	// LARGE_SIZE is a multiple of 16
	int num_data = 10000000;
	int block_num;
	int num_unit;
	int r;
	cl_mem d_status;
	cl_mem d_params;
	int mb, mp;


	block_num = 96;
/*	if (argc >= 2) {
		errno = 0;
		block_num = strtol(argv[1], NULL, 10);
		if (errno) {
			printf("%s number_of_block number_of_output\n", argv[0]);
			return 1;
		}
		if (block_num < 1 || block_num > BLOCK_NUM_MAX) {
			printf("%s block_num should be between 1 and %d\n",
					argv[0], BLOCK_NUM_MAX);
			return 1;
		}
		errno = 0;
		num_data = strtol(argv[2], NULL, 10);
		if (errno) {
			printf("%s number_of_block number_of_output\n", argv[0]);
			return 1;
		}
		argc -= 2;
		argv += 2;
	} else {
		printf("%s number_of_block number_of_output\n", argv[0]);
		block_num = get_suitable_block_num(device,
				&mb,
				&mp,
				sizeof(uint32_t),
				THREAD_NUM,
				LARGE_SIZE);
		if (block_num <= 0) {
			printf("can't calculate sutable number of blocks.\n");
			return 1;
		}
		printf("the suitable number of blocks for device 0 "
				"will be multiple of %d, or multiple of %d\n", block_num,
				(mb - 1) * mp);
		return 1;
	}
*/
	num_unit = LARGE_SIZE * block_num;
	d_status = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(struct mtgp32_kernel_status_t) * block_num, NULL, &clerr); CLERR;
	d_params = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(struct mtgp32_param_t), NULL, &clerr); CLERR;
//	ccudaMalloc((void**)&d_status, sizeof(mtgp32_kernel_status_t) * block_num);
	r = num_data % num_unit;
	if (r != 0) {
		num_data = num_data + num_unit - r;
	}
	make_constant(commands, d_params, MTGPDC_PARAM_TABLE, block_num);
	make_kernel_data32(commands, d_status, MTGPDC_PARAM_TABLE, block_num);
	make_uint32_random(d_status, d_params, num_data, block_num);
	make_single_random(d_status, d_params, num_data, block_num);

	clReleaseMemObject(d_status);
	clReleaseMemObject(d_params);

	/*Close connection with devices*/
	clReleaseKernel(mtgp32_uint32_kernel);
	clReleaseKernel(mtgp32_single_kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
}

