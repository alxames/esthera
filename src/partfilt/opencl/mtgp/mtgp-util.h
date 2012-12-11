#ifndef MTGP_UTIL_CU
#define MTGP_UTIL_CU
/*
 * mtgp-util.h
 *
 * Some utility functions for Sample Programs
 *
 */
#include <stdint.h>
#include <inttypes.h>
//#include "test-tool.hpp"

int get_suitable_block_num(int device, int *max_block_num,
			   int *mp_num, int word_size,
			   int thread_num, int large_size);
void print_max_min(uint32_t data[], int size);
void print_float_array(const float array[], int size, int block);
void print_uint32_array(const uint32_t array[], int size, int block);
void print_double_array(const double array[], int size, int block);
void print_uint64_array(const uint64_t array[], int size, int block);

#endif
