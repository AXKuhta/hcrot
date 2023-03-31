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

// A[i] += B[i]
static void elementwise_inplace(f32* restrict a, f32* restrict b, const size_t size, void fn(f32* a, f32* b)) {
	assert(strides_cleanly(size));

	size_t elements = size / sizeof(f32);

	for (size_t i = 0; i < elements; i += K_STRIDE_F32) {
		for (size_t j = 0; j < K_STRIDE_F32; j++) {
			fn(a + i + j, b + i + j);
		}
	}
}

static void f32_add_f32_inplace(f32* a, f32* b) { *a += *b; }
static void f32_sub_f32_inplace(f32* a, f32* b) { *a -= *b; }
static void f32_mul_f32_inplace(f32* a, f32* b) { *a *= *b; }
static void f32_div_f32_inplace(f32* a, f32* b) { *a /= *b; }

void f32_add_f32(f32* a, f32* b, const size_t size) { elementwise_inplace(a, b, size, f32_add_f32_inplace); }
void f32_sub_f32(f32* a, f32* b, const size_t size) { elementwise_inplace(a, b, size, f32_sub_f32_inplace); }
void f32_mul_f32(f32* a, f32* b, const size_t size) { elementwise_inplace(a, b, size, f32_mul_f32_inplace); }
void f32_div_f32(f32* a, f32* b, const size_t size) { elementwise_inplace(a, b, size, f32_div_f32_inplace); }

f32 dot_f32_f32(f32* restrict a, f32* restrict b, const size_t size) {
	assert(strides_cleanly(size));

	size_t elements = size / sizeof(f32);
	f32 accs[K_STRIDE_F32] = {0.0};
	f32 acc = 0.0;

	for (size_t i = 0; i < elements; i += K_STRIDE_F32) {
		for (size_t j = 0; j < K_STRIDE_F32; j++) {
			accs[j] += a[i + j] * b[i + j];
		}
	}

	for (size_t j = 0; j < K_STRIDE_F32; j++)
		acc += accs[j];

	return acc;
}
