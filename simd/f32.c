#include <stddef.h>
#include <assert.h>

#include "simd.h"

// SSE2 	2 loops (-O2 -march=core2)
// AVX 		4 ops	(-O2 -march=sandybridge)
// AVX2 	2 ops	(-O2 -march=skylake)
// AVX512F 	1 op 	(-O2 -march=icelake-server)
#ifndef K_STRIDE
#define K_STRIDE 16
#endif

static int strides_cleanly(const size_t size) {
	return size % K_STRIDE == 0;
}

void f32_add_f32(f32* restrict a, f32* restrict b, const size_t size) {
	assert(strides_cleanly(size));

	for (size_t i = 0; i < size; i += K_STRIDE) {
		for (size_t j = 0; j < K_STRIDE; j++) {
			a[i + j] += b[i + j];
		}
	}
}
