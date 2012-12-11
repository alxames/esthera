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

/* resampling-valias.cl
 * Particle filter resampling using Vose's Alias method to sample from a discrete distribution - OpenCL kernel routines.
 * Authors: Alexander S. van Amesfoort and Mehdi Chitchian
 * License: GNU Public License version 3 (GPLv3)
 * Date:    April 2012
 * Refs:    Michael D. Vose, "A Linear Algorithm For Generating Random Numbers With a Given Distribution",
 *          In "IEEE Transactions on Software Engineering", Vol. 17, No. 9, pp. 972--975, Sep 1991.
 *          http://web.eecs.utk.edu/~vose/Publications/random.pdf
 *
 *          Keith Schwarz, "Darts, Dice, and Coins: Sampling from a Discrete Distribution", "Last Major Update: December 29, 2011".
 *          http://www.keithschwarz.com/darts-dice-coins/
 */

// AMD APP OpenCL kernel debugging: pass -g, then break after compiled (e.g. when creating a Kernel obj),
// Type into DDD/GDB: 'info func __OpenCL_<__kernel-func-name>_kernel' and/or 'break __OpenCL_<__kernel-func-name>_kernel' to set a breakpoint.
// Use CPU_MAX_COMPUTE_UNITS to restrict the number of threads (for scal or dbg).
#ifdef DEBUG

#ifdef cl_amd_printf
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#endif /* DEBUG */

// Needed for test-valias.cpp only; the real program compiles with the right particle_set() etc all at once.
#ifdef TEST_VALIAS
// from robotarm/pf_model.cl -----------------------------
inline void particle_set(
		__global float4* p1_angles1,
		__global float*  p1_angles2,
		__global float4* p1_pos,
		const int p1,
		__global const float4* p2_angles1,
		__global const float*  p2_angles2,
		__global const float4* p2_pos,
		const int p2,
		const int num_particles,
		const int num_blocks)
{
	p1_angles1[p1] = p2_angles1[p2];
	p1_angles2[p1] = p2_angles2[p2];
	p1_pos[p1]     = p2_pos[p2];
}
// ---------------------------------------------------
#endif /* TEST_VALIAS */

/*
 * Return the sum of all elements of arr of size 'get_local_size(0)'.
 * Notes: Clobbers the array arr. And the work group size 'get_local_size(0)' must be a power of 2.
 */
float sum_reduce(__local float *restrict arr) {
	unsigned int ltid = get_local_id(0);

	unsigned int offset = get_local_size(0);
	while (offset > 1) {
		offset >>= 1;
		if (ltid < offset) {
			arr[ltid] += arr[ltid + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	return arr[0];
}

// This variant is slightly faster on CPUs and GPUs.
float sum_reduce4(__local float *restrict arr) {
	unsigned int ltid = get_local_id(0);

	unsigned int offset = get_local_size(0);
	while (offset > 2) {
		offset >>= 2;
		if (ltid < offset) {
			arr[ltid] += arr[ltid + offset] + arr[ltid + offset+offset] + arr[ltid + offset+offset+offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (offset > 1) { // offset==2 tail case
		if (ltid == 0) {
			arr[0] += arr[1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	return arr[0];
}

/*
 * Create the alias table (may adapt prob too). The small_large array is just __local scratch space.
 * NOTE: sh_small_size must have been initialized (and visible for all threads!) to 0 and sh_large_size to get_local_size(0)-1
 * The Steps refer to the final algorithm on Keith Schwarz' page; the URL is at the top of this file.
 */
void vose_alias_init(__local float *restrict prob, __local unsigned int *restrict alias, __local unsigned int *restrict small_large,
                     __local unsigned int *restrict sh_small_size, __local unsigned int *restrict sh_large_size) {
	// Step 1 and 2 are to create arrays prob[] and alias[], and worklists (also arrays) small[] and large[].
	// Together, small[] and large[] have a length of size, we just don't know how long each is.
	// So pack small[] and large[] into small_large[].

	unsigned int ltid = get_local_id(0);
	prob[ltid] *= (float)get_local_size(0); // scale all samples, such that our small vs large "average" is 1.0f
	alias[ltid] = ltid; // instead of Step 6 and 7

	// Step 4: divide in small and large ( probs are [0.0f, (float)get_local_size(0)) )
	// In this impl, all probs must become small, so use <= instead of <.
	unsigned old_idx;
	if (prob[ltid] <= 1.0f) old_idx = atomic_inc(sh_small_size);
	else                    old_idx = atomic_dec(sh_large_size); // populate large vals backwards
	small_large[old_idx] = ltid;
//barrier(CLK_LOCAL_MEM_FENCE); // tmp
//printf("prob [%u]=%f\n", ltid, prob[ltid]); // tmp
//printf("alias[%u]=%u\n", ltid, alias[ltid]); // tmp

	// Step 5
	__local unsigned int *restrict sh_small_end  = sh_small_size; // __local var rename only
	__local unsigned int *restrict sh_large_rend = sh_large_size; // idem
	unsigned int small_begin, small_end;
	unsigned int nactive;
	for (small_begin = 0;

             // Conceptually, the "small and large both non-empty" cond, with some re-init.
             barrier(CLK_LOCAL_MEM_FENCE), // *sh_small_end, *sh_large_rend, small_large[] and if returning, alias[], prob[]
             small_end = *sh_small_end,
             (nactive = min(small_end - small_begin, (unsigned int)get_local_size(0) - small_end)) > 0;

             small_begin += nactive)
	{
		/*
		 * Work units 0 till nactive-1 can process in parallel for sure. Skim large vals to fill small vals to 1.0f. A large val may become small.
		 *
		 * Multiple small could speculatively steal from the same large and only commit if large remains >=1.0f,
		 * but OpenCL only has float atomic_xchg(), while we need float atomic_sub(). Try it with CUDA.
		 */
		if (ltid == 0) {
			*sh_large_rend += nactive; // probably faster than unconditional atomic_inc() under ltid < nactive
		}

		unsigned int small_idx, large_idx;
		float probS, probL;
		if (ltid < nactive) {
			// Step 5.1 and 5.2. Scan small in ascending direction, and remove/add large also in ascending direction to add new small elements correctly.
			small_idx = small_large[small_begin + ltid];
			large_idx = small_large[small_end   + ltid];

			// Step (5.3 and) 5.4
			probS = prob[small_idx];
			//prob[small_idx] = probS; // Was already probS (in-place) unless it was a large val last iter, which was set below.
			alias[small_idx] = large_idx;

			// Step 5.5, 5.6 and 5.7
			probL = prob[large_idx];
		}
		barrier(CLK_LOCAL_MEM_FENCE); // *sh_large_rend, and to make sure everyone loaded from small_large[] and prob[] before we write back there.
		if (ltid < nactive) { // recheck to keep prev barrier() outside of diverging branch
			unsigned old_idx;
			probL = (probL + probS) - 1.0f; // somewhat more num stable than 'probL = probS - (1.0f - probL);'
			if (probL <= 1.0f) {
				old_idx = atomic_inc(sh_small_end); // large val became small
			} else { // Large val stays large, but it may still change idx if another large val with a higher idx becomes small.
				old_idx = atomic_dec(sh_large_rend);
			}
			small_large[old_idx] = large_idx;
			prob[large_idx] = probL;
		}
	}

	/*
	 * Step 6 and 7 are not needed, because:
	 * - We don't care if a prob ends up e.g. 1.0001f, because the generate_next func only does r < prob[col] with r < 1.0f.
	 * - If a prob ends up e.g. 0.9999f, we also don't care, because we have set its alias in the beginning to itself.
	 * And in both cases the loop above terminates properly.
	 */

	//barrier(CLK_LOCAL_MEM_FENCE); // tmp
	//printf("-prob [%u]=%f\n", ltid, prob[ltid]); // tmp
	//printf("-alias[%u]=%u\n", ltid, alias[ltid]); // tmp
	//barrier(CLK_LOCAL_MEM_FENCE); // tmp
}

// Vose's alias method needs 2 rand nrs per sample.
unsigned int vose_alias_generate_next(const __local float *restrict prob, const __local unsigned int *restrict alias, float2 rand_nrs) {
	unsigned int column = floor(rand_nrs.x * get_local_size(0));
	bool coin_toss = rand_nrs.y < prob[column];
	return coin_toss ? column : alias[column];
}

/*
 * Resample particle weights (probabilities). Their input weights do not have to be normalized.
 * Each work unit deals with one particle.
 * All buffers pointed to have the same size as the work group size 'get_local_size(0)',
 * except that per particle, we need two random numbers, so d_random must have twice the work group size.
 * Currently, the local (work group) size must be 1-dim and a power of 2.
 *
 * Opt: Could use ushort instead of uint to save 33% local memory; not >16383 particles per work group. (Maybe idem with float to fp16.)
 */
//__attribute__((reqd_work_group_size(NUM_PARTICLES, 1, 1)))
__kernel void resampling_valias(
	__global float4* angles1,
	__global float*  angles2,
	__global float4* pos,
	__global const float4* tmp_angles1,
	__global const float*  tmp_angles2,
	__global const float4* tmp_pos,

	__global const float* const d_particle_weights, // 'probabilities' in a generic resampler
	__global const float* d_random,

	const int num_particles,
	const float resampling_threshold,
#ifdef PRNG_MTGP
	const int rand_offset,
#endif /* PRNG_MTGP */

	// size of __local decl has to be nparticles (=get_local_size(0))
	__local float* prob,
	__local unsigned int* alias,
	__local unsigned int* scratch
)
{
#ifdef PRNG_MTGP
	d_random = &d_random[rand_offset];
#endif

#ifndef ADAPTIVE_RESAMPLING
#ifdef PRNG_MTGP
	if ((d_random[get_local_size(0)*get_num_groups(0) + get_group_id(0)]-1.0f) > resampling_threshold)
		return;
#else /* !PRNG_MTGP */
	if (d_random[get_local_size(0)*get_num_groups(0) + get_group_id(0)] > resampling_threshold)
		return;
#endif /* PRNG_MTGP */
#endif

	const unsigned int idx  = get_global_id(0);
	const unsigned int ltid = get_local_id(0);

	// Init and piggy-back on a barrier() before vose_alias_init().
	__local unsigned int tmp[2];
	if (ltid == 0) {
		tmp[0] = 0;
		tmp[1] = get_local_size(0)-1;
	}

	__local float *restrict scratchf = (__local float *)scratch; // assumes it fits ( sz(float)<=sz(uint) )
	float p = d_particle_weights[idx];

	scratchf[ltid] = p; // copy for sum_reduce4() to clobber
//printf("%u: p=%f\n", ltid, p); // tmp; need barriers to print properly
	barrier(CLK_LOCAL_MEM_FENCE);
	float sum = sum_reduce4(scratchf);
	if (sum > 0.0f) p /= sum; // normalize
	else            p = 1.0f / (float)get_local_size(0);
	prob[ltid] = p;
//printf("%u: sum=%f norm p=%f\n", ltid, sum, p); // tmp; idem

#ifdef ADAPTIVE_RESAMPLING
	// Compute reciprocal effective sample size = SUM{p^2}.
	scratchf[ltid] = p * p; // copy for sum_reduce4() to clobber
#endif
	barrier(CLK_LOCAL_MEM_FENCE);
#ifdef ADAPTIVE_RESAMPLING
	float recip_eff_sampl_sz = sum_reduce4(scratchf);

	// Resample this sub-filter iff the effective sample size is < some threshold.
	// We do reciprocal, i.e. resample iff 1.0/ess > 1.0/c. resampling_threshold: 0.0f: always; 1.0f: never (but fp precision)
	if (recip_eff_sampl_sz > resampling_threshold)
#endif
	{
		vose_alias_init(prob, alias, scratch, &tmp[0], &tmp[1]);

		__global const float2* d_random2 = (__global const float2*)d_random;
		float2 r2 = d_random2[idx];
#ifdef PRNG_MTGP
		// MTGP generates [1.0f, 2.0f), so correct that. (hack, MTGP kernel should do this)
		r2.x -= 1.0f;
		r2.y -= 1.0f;
#endif
		unsigned int selection_idx = vose_alias_generate_next(prob, alias, r2);

		// Copy the survivors. For pf, the weights will be recomp next round, so only copy the (other) state.
		//d_particle_weights[get_global_id(0)] = prob[selection_idx]; // different from Mehdi's commented-out code(?)
		particle_set(angles1, angles2, pos, idx, tmp_angles1, tmp_angles2, tmp_pos, get_group_id(0)*get_local_size(0)+selection_idx, num_particles, get_num_groups(0));
	}

	// TODO: if not resampled, keep the weights and in the next round, mult wcalc() with these weights to get the new weights.
	// If resampled, all weights become 1/N. We may want to set the weights here to 1.0 and always do the mult the next round (then normalize before the global est).
}

