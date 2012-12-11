#define uint32_t uint

#include "mtgp32-opencl.h"

/**
 * Shared memory
 * The generator's internal status vector.
 */
//__local uint32_t status[LARGE_SIZE];

/**
 * The function of the recursion formula calculation.
 *
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @param[in] bid block id.
 * @return output
 */
/*__device__*/
uint32_t para_rec(uint32_t X1, uint32_t X2, uint32_t Y, int bid, __constant struct mtgp32_param_t *params) {
	uint32_t X = (X1 & params->mask[0]) ^ X2;
	uint32_t MAT;

	X ^= X << params->sh1_tbl[bid];
	Y = X ^ (Y >> params->sh2_tbl[bid]);
	MAT = params->param_tbl[bid][Y & 0x0f];
	return Y ^ MAT;
}

/**
 * The tempering function.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered value.
 */
/*__device__*/
uint32_t temper(uint32_t V, uint32_t T, int bid, __constant struct mtgp32_param_t *params) {
	uint32_t MAT;

	T ^= T >> 16;
	T ^= T >> 8;
	MAT = params->temper_tbl[bid][T & 0x0f];
	return V ^ MAT;
}

/**
 * The tempering and converting function.
 * By using the preset-ted table, converting to IEEE format
 * and tempering are done simultaneously.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered and converted value.
 */
/*__device__*/
float temper_single(uint32_t V, uint32_t T, int bid, __constant struct mtgp32_param_t *params) {
	uint32_t MAT;
	uint32_t r;

	T ^= T >> 16;
	T ^= T >> 8;
	MAT = params->single_temper_tbl[bid][T & 0x0f];
	r = (V >> 9) ^ MAT;
	return as_float(r);
}

/**
 * Read the internal state vector from kernel I/O data, and
 * put them into shared memory.
 *
 * @param[out] status shared memory.
 * @param[in] d_status kernel I/O data
 * @param[in] bid block id
 * @param[in] tid thread id
 */
/*__device__*/
void status_read(
		__local uint32_t *status,
		__global const struct mtgp32_kernel_status_t *d_status,
		int bid,
		int tid) {
	status[LARGE_SIZE - N + tid] = d_status[bid].status[tid];
	if (tid < N - THREAD_NUM) {
		status[LARGE_SIZE - N + THREAD_NUM + tid]
			= d_status[bid].status[THREAD_NUM + tid];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

/**
 * Read the internal state vector from shared memory, and
 * write them into kernel I/O data.
 *
 * @param[out] d_status kernel I/O data
 * @param[in] status shared memory.
 * @param[in] bid block id
 * @param[in] tid thread id
 */
/*__device__*/
void status_write(
		__global struct mtgp32_kernel_status_t *d_status,
		__local const uint32_t *status,
		int bid,
		int tid) {
	d_status[bid].status[tid] = status[LARGE_SIZE - N + tid];
	if (tid < N - THREAD_NUM) {
		d_status[bid].status[THREAD_NUM + tid]
			= status[4 * THREAD_NUM - N + tid];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

/**
 * kernel function.
 * This function generates 32-bit unsigned integers in d_data
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output
 * @param[in] size number of output data requested.
 */
__kernel void mtgp32_uint32_kernel(
		__global struct mtgp32_kernel_status_t* d_status,
		__global uint32_t* d_data,
		__local uint32_t* status,
		__constant struct mtgp32_param_t* params,
		int size) {
	const int bid = get_group_id(0);
	const int tid = get_local_id(0);
	int pos = params->pos_tbl[bid];
	uint32_t r;
	uint32_t o;

	// copy status data from global memory to shared memory.
	status_read(status, d_status, bid, tid);

	// main loop
	for (int i = 0; i < size; i += LARGE_SIZE) {

		r = para_rec(status[LARGE_SIZE - N + tid],
				status[LARGE_SIZE - N + tid + 1],
				status[LARGE_SIZE - N + tid + pos],
				bid,
				params);
		status[tid] = r;
		o = temper(r, status[LARGE_SIZE - N + tid + pos - 1], bid, params);
		d_data[size * bid + i + tid] = o;
		barrier(CLK_LOCAL_MEM_FENCE);
		r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
				status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
				status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
				bid,
				params);
		status[tid + THREAD_NUM] = r;
		o = temper(r,
				status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
				bid,
				params);
		d_data[size * bid + THREAD_NUM + i + tid] = o;
		barrier(CLK_LOCAL_MEM_FENCE);
		r = para_rec(status[2 * THREAD_NUM - N + tid],
				status[2 * THREAD_NUM - N + tid + 1],
				status[2 * THREAD_NUM - N + tid + pos],
				bid,
				params);
		status[tid + 2 * THREAD_NUM] = r;
		o = temper(r, status[tid + pos - 1 + 2 * THREAD_NUM - N], bid, params);
		d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	// write back status for next call
	status_write(d_status, status, bid, tid);
}

/**
 * kernel function.
 * This function generates single precision floating point numbers in d_data.
 *
 * @param[in,out] d_status kernel I/O data
 * @param[out] d_data output. IEEE single precision format.
 * @param[in] size number of output data requested.
 */
__kernel void mtgp32_single_kernel(
		__global struct mtgp32_kernel_status_t* d_status,
		__global float* d_data,
		__local uint32_t* status,
		__constant struct mtgp32_param_t* params,
		int size)
{

	const int bid = get_group_id(0);
	const int tid = get_local_id(0);
	int pos = params->pos_tbl[bid];
	uint32_t r;
	float o;

	// copy status data from global memory to shared memory.
	status_read(status, d_status, bid, tid);

	// main loop
	for (int i = 0; i < size; i += LARGE_SIZE) {
		r = para_rec(status[LARGE_SIZE - N + tid],
				status[LARGE_SIZE - N + tid + 1],
				status[LARGE_SIZE - N + tid + pos],
				bid,
				params);
		status[tid] = r;
		o = temper_single(r, status[LARGE_SIZE - N + tid + pos - 1], bid, params);
		d_data[size * bid + i + tid] = o;
		barrier(CLK_LOCAL_MEM_FENCE);
		r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
				status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
				status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
				bid,
				params);
		status[tid + THREAD_NUM] = r;
		o = temper_single(
				r,
				status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
				bid,
				params);
		d_data[size * bid + THREAD_NUM + i + tid] = o;
		barrier(CLK_LOCAL_MEM_FENCE);
		r = para_rec(status[2 * THREAD_NUM - N + tid],
				status[2 * THREAD_NUM - N + tid + 1],
				status[2 * THREAD_NUM - N + tid + pos],
				bid,
				params);
		status[tid + 2 * THREAD_NUM] = r;
		o = temper_single(r,
				status[tid + pos - 1 + 2 * THREAD_NUM - N],
				bid,
				params);
		d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	// write back status for next call
	status_write(d_status, status, bid, tid);
}

