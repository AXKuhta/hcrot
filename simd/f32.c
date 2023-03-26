#include <stddef.h>
#include <assert.h>

#include "simd.h"

// K_STRIDE_F32 = 16:
// SSE2 	2 loops (-O2 -march=core2)
// AVX 		4 ops	(-O2 -march=sandybridge)
// AVX2 	2 ops	(-O2 -march=skylake)
// AVX512F 	1 op 	(-O2 -march=icelake-server)
#define K_STRIDE_F32 (K_STRIDE / sizeof(f32))

static int strides_cleanly(const size_t size) {
	return size % K_STRIDE == 0;
}

void f32_add_f32(f32* restrict a, f32* restrict b, const size_t size) {
	assert(strides_cleanly(size));

	for (size_t i = 0; i < size; i += K_STRIDE_F32) {
		for (size_t j = 0; j < K_STRIDE_F32; j++) {
			a[i + j] += b[i + j];
		}
	}
}

f32 dot_f32_f32(f32* restrict a, f32* restrict b, const size_t size) {
	f32 acc = 0.0;

	for (size_t i = 0; i < size; i += K_STRIDE_F32) {
		for (size_t j = 0; j < K_STRIDE_F32; j++) {
			acc += a[i + j] * b[i + j];
		}
	}

	return acc;
}

/*
f32 dot_f32_f32(f32* restrict a, f32* restrict b, const size_t size) {
	f32 accs[K_STRIDE_F32] = {0.0};
	f32 acc = 0.0;

	for (size_t i = 0; i < size; i += K_STRIDE_F32) {
		for (size_t j = 0; j < K_STRIDE_F32; j++) {
			accs[j] += a[i + j] * b[i + j];
		}
	}

	for (size_t j = 0; j < K_STRIDE_F32; j++)
		acc += accs[j];

	return acc;
}
*/
