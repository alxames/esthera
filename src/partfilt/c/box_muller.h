#ifndef _BOX_MULLER_H
#define _BOX_MULLER_H

#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>

#include <xmmintrin.h>

#include <simdmath.h>
#include <logf4.h>
#include <sincosf4.h>

#include "sse_aux.h"

#define D_2POW32_INV (2.3283064365386962890625e-10f)
#define D_2POW33_INV (1.16415321826934814453125e-10f)
#define D_2PI (6.28318530717958647692f)

static void box_muller(
	const uint32_t* const uniform_uint,
	float* const normal,
	float* const uniform,
	const int num_normal,
	const int num_uniform)
{
	for (int i=0; i < num_normal; i+=2)
	{
		float u1 = uniform_uint[i] * D_2POW32_INV + D_2POW33_INV;
		float u2 = (uniform_uint[i+1] * D_2POW32_INV + D_2POW33_INV) * D_2PI;
		float r = sqrtf(-2.0f * logf(u1));
		float s, c;
		sincosf(u2, &s, &c);
		normal[i] = r * s;
		normal[i+1] = r * c;
	}

	for (int i=0; i < num_uniform; ++i)
	{
		uniform[i] = uniform_uint[num_normal+i] * D_2POW32_INV + D_2POW33_INV;
	}
}

static void box_muller_sse(
	const uint32_t* const uniform_uint,
	float* const normal,
	float* const uniform,
	const int num_normal,
	const int num_uniform)
{
	const __m128 d_2pow32_inv = _mm_set1_ps(D_2POW32_INV);
	const __m128 d_2pow33_inv = _mm_set1_ps(D_2POW33_INV);
	const __m128 d_2pi        = _mm_set1_ps(D_2PI);

	const __m128i* m_uniform_uint = (__m128i*) uniform_uint;
	__m128* m_normal              = (__m128*)  normal;
	__m128* m_uniform             = (__m128*)  uniform;

	for (int i=0; i < (num_normal/4); i+=2)
	{
		__m128 u1 = _mm_add_ps(_mm_mul_ps(_mm_ctf_epu32(m_uniform_uint[i]), d_2pow32_inv), d_2pow33_inv);
		__m128 u2 = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_ctf_epu32(m_uniform_uint[i+1]), d_2pow32_inv), d_2pow33_inv), d_2pi);
		__m128 r = _mm_sqrt_ps(_mm_mul_ps(_mm_set1_ps(-2.0f), _logf4(u1)));
		__m128 s, c;
		_sincosf4(u2, &s, &c);
		m_normal[i]   = _mm_mul_ps(r, s);
		m_normal[i+1] = _mm_mul_ps(r, c);
	}

	for (int i=0; i < (num_uniform/4); ++i)
	{
		m_uniform[i] = _mm_add_ps(_mm_mul_ps(_mm_ctf_epu32(m_uniform_uint[(num_normal/4)+i]), _mm_set1_ps(D_2POW32_INV)), _mm_set1_ps(D_2POW33_INV));
	}
}

#endif /* _BOX_MULLER_H */
