#ifndef MTGP32_OPENCL_H
#define MTGP32_OPENCL_H 1

#define MTGPDC_MEXP 11213
#define MTGPDC_N 351
#define MTGPDC_FLOOR_2P 256
#define MTGPDC_CEIL_2P 512
#define MTGPDC_PARAM_TABLE mtgp32dc_params_fast_11213
#define MEXP 11213
#define THREAD_NUM MTGPDC_FLOOR_2P
#define LARGE_SIZE (THREAD_NUM * 3)
#define BLOCK_NUM_MAX 200
#define TBL_SIZE 16
#define N MTGPDC_N

/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
struct mtgp32_kernel_status_t {
	uint32_t status[MTGPDC_N];
};

/*
 * Generator Parameters.
 */
struct mtgp32_param_t {
	unsigned int pos_tbl[BLOCK_NUM_MAX];
	uint32_t param_tbl[BLOCK_NUM_MAX][TBL_SIZE];
	uint32_t temper_tbl[BLOCK_NUM_MAX][TBL_SIZE];
	uint32_t single_temper_tbl[BLOCK_NUM_MAX][TBL_SIZE];
	uint32_t sh1_tbl[BLOCK_NUM_MAX];
	uint32_t sh2_tbl[BLOCK_NUM_MAX];
	uint32_t mask[1];
};

#endif /* MTGP32_OPENCL_H */
