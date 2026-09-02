#include "box_muller.h"
#include <pf_model.h>

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

#ifdef PRNG_SFMT
#include <SFMT.h>
#else /* !PRNG_SFMT */
#error "no PRNG set"
#endif /* PRNG_SFMT */

extern const particle_state init_state;

void sampling_importance(
	particle_state* const particle_data,
	const control control_data,
	const measurement measurement_data,
	const float* const d_random,
	const float dt);

int read_trace(
	FILE* const input_file,
	control* const control_data,
	measurement* const measurement_data,
	particle_state* const actual_state,
	float* const dt);

float estimate_error(
	const particle_state estimate,
	const particle_state actual);

void print_particle(const particle_state particle_data);



static void resampling(
	particle_state* const new_particle_data,
	particle_state* const old_particle_data,
	const float* const uniform_random,
	const int num_particles)
{
	float sum_weight = 0;

	/* build up prefix sum */
	for (int i=0; i < num_particles; ++i)
	{
		sum_weight += old_particle_data[i].weight;
		old_particle_data[i].weight = sum_weight;
	}

	for (int i=0; i < num_particles; ++i)
	{
		const float target = uniform_random[i]*sum_weight;

		int j=0;

		while (j < num_particles && old_particle_data[j].weight < target)
		{
			++j;
		}

		new_particle_data[i] = old_particle_data[j];
	}
}

static void resampling_vose(
	particle_state* const new_particle_data,
	particle_state* const old_particle_data,
	const float* const uniform_random,
	const int num_particles,
	int* const alias,
	float* const prob,
	int* const small,
	int* const large)
{
	int num_small = 0;
	int num_large = 0;

	float sum_weight = 0.0f;

	for (int i=0; i < num_particles; ++i)
	{
		sum_weight += old_particle_data[i].weight;
	}

	/* TODO: handle sum_weight == 0.0f */

	for (int i=0; i < num_particles; ++i)
	{
		alias[i] = i;

		old_particle_data[i].weight *= num_particles / sum_weight;

		if (old_particle_data[i].weight < 1.0f)
		{
			small[num_small++] = i;
		}
		else
		{
			large[num_large++] = i;
		}
	}

	while (num_small > 0 && num_large > 0)
	{
		int l = small[num_small-1];
		int g = large[num_large-1];

		--num_small;
		--num_large;

		prob[l] = old_particle_data[l].weight;
		alias[l] = g;
		old_particle_data[g].weight = (old_particle_data[g].weight + old_particle_data[l].weight) - 1.0f;

		if (old_particle_data[g].weight < 1.0f)
		{
			small[num_small++] = g;
		}
		else
		{
			large[num_large++] = g;
		}
	}

/*
	while (num_large > 0)
	{
		prob[large[--num_large]] = 1.0f;
	}

	while (num_small > 0)
	{
		prob[small[--num_small]] = 1.0f;
	}
*/

	for (int i=0; i < num_particles; ++i)
	{
/*
		if (uniform_random[2*i] < 0.0f || uniform_random[2*i] > 1.0f)
		{
			fprintf(stderr, "(%i) bad rand: %f\n", 2*i, uniform_random[2*i]);
			exit(1);
		}
*/
		int col = (int)floorf(uniform_random[2*i] * num_particles);

		/* uniform random from SFMT can be 1.0f */
		if (col == num_particles)
		{
			--col;
		}
/*
		if (uniform_random[2*i+1] < 0.0f || uniform_random[2*i+1] > 1.0f)
		{
			fprintf(stderr, "(%i) bad rand: %f\n", 2*i+1, uniform_random[2*i+1]);
			exit(1);
		}
*/

		if (uniform_random[2*i+1] < prob[col])
		{
			new_particle_data[i] = old_particle_data[col];
		}
		else
		{
			new_particle_data[i] = old_particle_data[alias[col]];
		}
	}
}

static int get_max_particle(const particle_state *particle_data, const int num_particles)
{
	float max_weight = -1;
	float max_index = -1;

	for (int i=0; i < num_particles; ++i)
	{
		if (particle_data[i].weight > max_weight)
		{
			max_weight = particle_data[i].weight;
			max_index = i;
		}
	}

	return max_index;
}

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

static uint64_t get_seed() {
#ifdef DETERMINISTIC
        return 0x11111DEAF000000DULL;
#else
        struct timeval t;
        gettimeofday(&t, NULL);
        return t.tv_usec;
#endif
}

static void print_particles(const particle_state* const particle_data, const int num_particles)
{
	for (int i=0; i < num_particles; ++i)
	{
		print_particle(particle_data[i]);
	}
}

int particle_filter(const int num_particles, const char* input_file_str)
{
	/* hack to prevent get_max_particle() from getting optimised out
	 * when not printing the estimate
	 */
	static int acc = 0;

	FILE* const input_file = fopen(input_file_str, "r");

	if (input_file == NULL)
	{
		fprintf(stderr, "could not open %s\n", input_file_str);
		exit(1);
	}

	const int num_normal_random = num_particles * NUM_STATE_VARIABLES;
#ifdef VALIAS_RESAMPLING
	const int num_uniform_random = 2*num_particles;
#else /* !VALIAS_RESAMPLING */
	const int num_uniform_random = num_particles;
#endif /* VALIAS_RESAMPLING */

	float* rand_normal;
	float* rand_uniform;

	posix_memalign((void **)&rand_normal,  16, sizeof(float) * num_normal_random);
	posix_memalign((void **)&rand_uniform, 16, sizeof(float) * num_uniform_random);

#ifdef PRNG_SFMT
	init_gen_rand(get_seed());

	const int sfmt_min_size = get_min_array_size32();
	const int sfmt_num_random = (num_normal_random + num_uniform_random) <  sfmt_min_size ? sfmt_min_size : (num_normal_random + num_uniform_random);
	uint32_t *sfmt_array;
	posix_memalign((void **)&sfmt_array, 16, sizeof(float) * sfmt_num_random);
#endif /* PRNG_SFMT */

	/* particles */
	particle_state *particle_data;
	particle_data = malloc(sizeof(particle_state) * num_particles);
	particle_state *tmp_particle_data;
	tmp_particle_data = malloc(sizeof(particle_state) * num_particles);

	for (int i=0; i < num_particles; ++i)
	{
		particle_data[i] = init_state;
	}

#ifdef VALIAS_RESAMPLING
	int*   resampling_alias = malloc(sizeof(int)   * num_particles);
	float* resampling_prob  = malloc(sizeof(float) * num_particles);
	int*   resampling_small = malloc(sizeof(int)   * num_particles);
	int*   resampling_large = malloc(sizeof(int)   * num_particles);
#endif /* VALIAS_RESAMPLING */

	/* sensor/control input */
	control control_data;
	measurement measurement_data;
	particle_state actual_state;
	float dt;

	int sample_count=0;

#ifdef DEBUG_ESTIMATE
	float error_sum = 0;
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
	int64_t s1_total=0;
	int64_t s2_total=0;
	int64_t s3_total=0;
	int64_t s4_total=0;
	int64_t s5_total=0;
	int64_t s6_total=0;
	int64_t s7_total=0;
#endif /* TIMING_HOST */


	while (read_trace(input_file, &control_data, &measurement_data, &actual_state, &dt) == 0)
	{
#ifdef TIMING_HOST
		struct timespec t1,t2;
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

		fill_array32(sfmt_array, sfmt_num_random);

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s1 = time_diff_nano(t2,t1);
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

#ifdef PRNG_SFMT
		box_muller_sse(sfmt_array, rand_normal, rand_uniform, num_normal_random, num_uniform_random);
#endif /* PRNG_SFMT */

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s2 = time_diff_nano(t2,t1);
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

		for (int i=0; i < num_particles; ++i)
		{
			sampling_importance(&particle_data[i], control_data, measurement_data, &rand_normal[i * NUM_STATE_VARIABLES], dt);
		}

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s3 = time_diff_nano(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		printf("%i: SAMPLING\n", sample_count);
		print_particles(particle_data, num_particles);
#endif /* DEBUG */

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */


#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s4 = time_diff_nano(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		printf("%i: BLOCK SORT\n", sample_count);
		print_particles(particle_data, num_particles);
#endif /* DEBUG */

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

		const int lpwi = get_max_particle(particle_data, num_particles);

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s5 = time_diff_nano(t2,t1);
#endif /* TIMING_HOST */

		/* do not optimise out get_max_particle() */
		acc += lpwi;

#ifdef DEBUG_ESTIMATE
		{
			particle_state lpw = particle_data[lpwi];

			float est_error = estimate_error(lpw, actual_state);
			error_sum += est_error;
#ifndef GLOBAL_ONLY
			printf("%i: ESTIMATE ::%i,%f\n", sample_count, lpwi, lpw.weight);
			print_particle(lpw);
#endif /* GLOBAL_ONLY */
		}
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s6 = time_diff_nano(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		printf("%i: EXCHANGE\n", sample_count);
		print_particles(particle_data, num_particles);
#endif /* DEBUG */

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
#endif /* TIMING_HOST */

#ifdef VALIAS_RESAMPLING
		resampling_vose(
			tmp_particle_data,
			particle_data,
			rand_uniform,
			num_particles
			,resampling_alias
			,resampling_prob
			,resampling_small
			,resampling_large
		);
#else /* !VALIAS_RESAMPLING */
		resampling(
			tmp_particle_data,
			particle_data,
			rand_uniform,
			num_particles);
#endif /* VALIAS_RESAMPLING */

		/* swap pointers */
		{
			particle_state* swap_tmp = tmp_particle_data;
			tmp_particle_data = particle_data;
			particle_data = swap_tmp;
		}

#ifdef TIMING_HOST
		clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
		const int64_t s7 = time_diff_nano(t2,t1);
#endif /* TIMING_HOST */

#ifdef DEBUG
		printf("%i: RESAMPLING\n", sample_count);
		print_particles(particle_data, num_particles);
#endif /* DEBUG */

#ifdef TIMING_HOST
#ifndef GLOBAL_ONLY
		printf("%2ld %4ld %4ld "
				"%3ld %4ld %4ld %4ld | %ld\n",
				s1,//time_diff_nano(s2,s1),//((double)time_diff_nano(s2,s1)*100)/time_diff_nano(s6,s1),
				s2,//time_diff_nano(s3,s2),//((double)time_diff_nano(s3,s2)*100)/time_diff_nano(s6,s1),
				s3,//time_diff_nano(s4,s3),//((double)time_diff_nano(s4,s3)*100)/time_diff_nano(s6,s1),
				s4,//time_diff_nano(s5,s4),//((double)time_diff_nano(s5,s4)*100)/time_diff_nano(s6,s1),
				s5,//time_diff_nano(s6,s5),//((double)time_diff_nano(s6,s5)*100)/time_diff_nano(s6,s1),
				s6,
				s7,
				s1+s2+s3+s4+s5+s6+s7);//time_diff_nano(s6,s1));
#endif /* GLOBAL_ONLY */

		s1_total += s1;
		s2_total += s2;
		s3_total += s3;
		s4_total += s4;
		s5_total += s5;
		s6_total += s6;
		s7_total += s7;
#endif /* TIMING_HOST */

		sample_count++;
	}

	fclose(input_file);

#ifdef DEBUG_ESTIMATE
	printf("%d %.16f\n", num_particles, error_sum/sample_count);
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
	printf("%d %ld %ld %ld %ld %ld %ld %ld %ld\n",
			num_particles,
			s1_total/sample_count,
			s2_total/sample_count,
			s3_total/sample_count,
			s4_total/sample_count,
			s5_total/sample_count,
			s6_total/sample_count,
			s7_total/sample_count,
			(s1_total+s2_total+s3_total+s4_total+s5_total+s6_total+s7_total)/sample_count);
#endif /* TIMING_HOST */

	free(particle_data);
	free(tmp_particle_data);

#ifdef PRNG_SFMT
	free(sfmt_array);
#endif /* PRNG_SFMT */

	free(rand_normal);
	free(rand_uniform);

#ifdef VALIAS_RESAMPLING
	free(resampling_alias);
	free(resampling_prob);
	free(resampling_small);
	free(resampling_large);
#endif /* VALIAS_RESAMPLING */

	return 0;
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

int main(int argc, char* argv[])
{
	int num_particles_start = 256;
	int num_particles_end   = 256;

	int loop_count          = 1;

	int opt;

	while (-1 != (opt = getopt(argc, argv, "m:l:"))) {
		switch (opt)
		{
			case 'm':
				parse_range(optarg, &num_particles_start, &num_particles_end);
				break;
			case 'l':
				loop_count = atoi(optarg);
				break;
			default: /* '?' */
				fprintf(stderr, "Usage: %s [-m #particles] [-l loop_count] input_file\n", argv[0]);
				exit(EXIT_FAILURE);
		}
	}

	if (optind >= argc)
	{
		fprintf(stderr, "no input file given\n");
		exit(EXIT_FAILURE);
	}

#ifdef DEBUG_ESTIMATE
	printf("m e\n");
#endif /* DEBUG_ESTIMATE */

#ifdef TIMING_HOST
	printf("m t1 t2 t3 t4 t5 t6 t7 total\n");
#endif /* TIMING_HOST */

	for (int num_particles=num_particles_start; num_particles <= num_particles_end; num_particles*=2)
	{
		for (int i=0; i < loop_count; ++i)
		{
			particle_filter(num_particles, argv[optind]);
		}
	}

	return EXIT_SUCCESS;
}

