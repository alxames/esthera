#ifndef _SSE_AUX_H
#define _SSE_AUX_H

/*
 * https://developer.apple.com/hardwaredrivers/ve/sse.html
 */

typedef __m128 vFloat;
typedef __m128i vUInt32;
typedef __m128i vSInt32;

const vFloat two16 = (const vFloat) {0x1.0p16f,0x1.0p16f,0x1.0p16f,0x1.0p16f};

//Convert vUInt32 to vFloat according to the current rounding mode
static inline vFloat _mm_ctf_epu32( vUInt32 v )
{

	// Avoid double rounding by doing two exact conversions
	//of high and low 16-bit segments
	vSInt32 hi = _mm_srli_epi32( (vSInt32) v, 16 );
	vSInt32 lo = _mm_srli_epi32( _mm_slli_epi32( (vSInt32) v, 16 ), 16 );
	vFloat fHi = _mm_mul_ps( _mm_cvtepi32_ps( hi ), two16);
	vFloat fLo = _mm_cvtepi32_ps( lo );

	// do single rounding according to current rounding mode
	// note that AltiVec always uses round to nearest. We use current
	// rounding mode here, which is round to nearest by default.
	return _mm_add_ps( fHi, fLo );

}

#endif /* _SSE_AUX_H */
