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

#include <CL/cl.h>

#include "mtgp-util.h"
#include "mtgp32-fast.h"
#include "mtgp32-opencl-tex.h"

#define CLERR \
	if (clerr != CL_SUCCESS) { \
		printf("opencl error %d,\t file: %s   line: %d\n", clerr,__FILE__,__LINE__); \
		exit(-1); \
	}

/**
 * This function initializes kernel I/O data.
 * @param d_status output kernel I/O data.
 * @param params MTGP32 parameters. needed for the initialization.
 */
void make_kernel_data32(cl_command_queue commands,
		cl_mem d_status,
		mtgp32_params_fast_t params[],
		const uint32_t seed,
		int block_num)
{
	int i;
	struct mtgp32_kernel_status_t* h_status
		= (struct mtgp32_kernel_status_t *) malloc(
				sizeof(struct mtgp32_kernel_status_t) * block_num);

	if (h_status == NULL) {
		printf("failure in allocating host memory for kernel I/O data.\n");
		exit(8);
	}
	for (i = 0; i < block_num; i++) {
		mtgp32_init_state(&(h_status[i].status[0]), &params[i], seed + i + 1);
	}
#if defined(DEBUG)
	printf("h_status[0].status[0]:%08u\n", h_status[0].status[0]);
	printf("h_status[0].status[1]:%08u\n", h_status[0].status[1]);
	printf("h_status[0].status[2]:%08u\n", h_status[0].status[2]);
	printf("h_status[0].status[3]:%08u\n", h_status[0].status[3]);
#endif
//	ccudaMemcpy(d_status, h_status,
//			sizeof(mtgp32_kernel_status_t) * block_num,
//			cudaMemcpyHostToDevice);
	cl_int clerr = clEnqueueWriteBuffer(commands, d_status, CL_TRUE, 0, sizeof(struct mtgp32_kernel_status_t) * block_num, h_status, 0, NULL, NULL); CLERR;
	free(h_status);
}

/**
 * This function sets constants in device memory.
 * @param[in] params input, MTGP32 parameters.
 */
void make_constant(cl_command_queue commands, cl_mem d_params, const mtgp32_params_fast_t params[], int block_num) {
	struct mtgp32_param_t h_params;
	h_params.mask[0] = params[0].mask;
	for (int i = 0; i < block_num; i++) {
		h_params.pos_tbl[i] = params[i].pos;
		h_params.sh1_tbl[i] = params[i].sh1;
		h_params.sh2_tbl[i] = params[i].sh2;
	}

	cl_int clerr = clEnqueueWriteBuffer(commands, d_params, CL_TRUE, 0, sizeof(struct mtgp32_param_t), &h_params, 0, NULL, NULL); CLERR;
}

/**
 * This function sets texture lookup table.
 * @param[in] params input, MTGP32 parameters.
 */
void make_texture(cl_command_queue commands, cl_mem d_tex_param, /*cl_mem d_tex_temper,*/ cl_mem d_tex_single_temper, const mtgp32_params_fast_t params[], int block_num) {
	cl_uint *h_param         = (cl_uint*)malloc(sizeof(cl_uint) * block_num * TBL_SIZE);//[BLOCK_NUM_MAX][TBL_SIZE];
//	cl_uint *h_temper        = (cl_uint*)malloc(sizeof(cl_uint) * block_num * TBL_SIZE);//[BLOCK_NUM_MAX][TBL_SIZE];
	cl_uint *h_single_temper = (cl_uint*)malloc(sizeof(cl_uint) * block_num * TBL_SIZE);//[BLOCK_NUM_MAX][TBL_SIZE];
	// row major [TBL_SIZE][block_num]
	for (int i = 0; i < block_num; i++) {
		for (int j = 0; j < TBL_SIZE; j++) {
			h_param        [j*block_num + i] = params[i].tbl[j];
//			h_temper       [j*block_num + i] = params[i].tmp_tbl[j];
			h_single_temper[j*block_num + i] = params[i].flt_tmp_tbl[j];
		}
	}
	const size_t origin[3] = {0, 0, 0};
	const size_t region[3] = {block_num, TBL_SIZE, 1};
	cl_int clerr;
	clerr = clEnqueueWriteImage(commands, d_tex_param, CL_TRUE, origin, region, 0, 0, h_param, 0, NULL, NULL); CLERR;
//	clerr = clEnqueueWriteImage(commands, d_tex_temper, CL_TRUE, origin, region, 0, 0, h_temper, 0, NULL, NULL); CLERR;
	clerr = clEnqueueWriteImage(commands, d_tex_single_temper, CL_TRUE, origin, region, 0, 0, h_single_temper, 0, NULL, NULL); CLERR;

	free(h_param);
	free(h_single_temper);
}
