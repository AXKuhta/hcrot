#include <stddef.h>
#include <assert.h>

#include "simd.h"

static f32 add(f32 a, f32 b) { return a + b; }
static f32 sub(f32 a, f32 b) { return a - b; }
static f32 mul(f32 a, f32 b) { return a * b; }
static f32 div(f32 a, f32 b) { return a / b; }

#define flat __attribute__((flatten))

// K_STRIDE_F32 = 16:
// SSE2 	2 loops (-O2 -march=core2)
// AVX 		4 ops	(-O2 -march=sandybridge)
// AVX2 	2 ops	(-O2 -march=skylake)
// AVX512F 	1 op 	(-O2 -march=icelake-server)
#define K_STRIDE_F32 (K_STRIDE / sizeof(f32))

#define FOR_HEAD(expr) for (size_t i = 0; i < head; i += K_STRIDE_F32) for (size_t j = 0; j < K_STRIDE_F32; j++) expr
#define FOR_TAIL(expr) for (size_t i = head, j = 0; j < tail; j++) expr

// A[i] += B[i]
static void elementwise_inplace_ab(f32* restrict a, f32* restrict b, const size_t size, f32 fn(f32 a, f32 b)) {
	size_t elements = size / sizeof(f32);
	size_t head = elements & ~(K_STRIDE_F32 - 1);
	size_t tail = elements & (K_STRIDE_F32 - 1);

	FOR_HEAD( a[i + j] = fn(a[i + j], b[i + j]) );
	FOR_TAIL( a[i + j] = fn(a[i + j], b[i + j]) );
}

flat void f32_add_f32(f32* a, f32* b, const size_t size) { elementwise_inplace_ab(a, b, size, add); }
flat void f32_sub_f32(f32* a, f32* b, const size_t size) { elementwise_inplace_ab(a, b, size, sub); }
flat void f32_mul_f32(f32* a, f32* b, const size_t size) { elementwise_inplace_ab(a, b, size, mul); }
flat void f32_div_f32(f32* a, f32* b, const size_t size) { elementwise_inplace_ab(a, b, size, div); }

// acc = fn(acc, A[i])
static f32 reduce(f32* restrict x, const size_t size, f32 fn(f32 a, f32 b), f32 acc_init) {
	size_t elements = size / sizeof(f32);
	size_t head = elements & ~(K_STRIDE_F32 - 1);
	size_t tail = elements & (K_STRIDE_F32 - 1);

	f32 accs[K_STRIDE_F32];
	f32 acc = acc_init;

	for (size_t i = 0; i < K_STRIDE_F32; i++)
		accs[i] = acc_init;

	FOR_HEAD( accs[j] = fn(x[i + j], accs[j]) );

	for (size_t i = 0; i < K_STRIDE_F32; i++)
		acc = fn(accs[i], acc);

	FOR_TAIL( acc = fn(x[i + j], acc) );

	return acc;
}

static f32 min(f32 a, f32 b) { return a < b ? a : b; }
static f32 max(f32 a, f32 b) { return a > b ? a : b; }
static f32 sum(f32 a, f32 b) { return a + b; }

flat f32 f32_min(f32* x, const size_t size) { return reduce(x, size, min, x[0]); }
flat f32 f32_max(f32* x, const size_t size) { return reduce(x, size, max, x[0]); }
flat f32 f32_sum(f32* x, const size_t size) { return reduce(x, size, sum, 0); }

// acc += A[i] * B[i]
f32 dot_f32_f32(f32* restrict a, f32* restrict b, const size_t size) {
	size_t elements = size / sizeof(f32);
	size_t head = elements & ~(K_STRIDE_F32 - 1);
	size_t tail = elements & (K_STRIDE_F32 - 1);

	f32 accs[K_STRIDE_F32] = {0.0};
	f32 acc = 0.0;

	FOR_HEAD( accs[j] += a[i + j] * b[i + j] );

	for (size_t j = 0; j < K_STRIDE_F32; j++)
		acc += accs[j];

	FOR_TAIL( acc += a[i + j] * b[i + j] );

	return acc;
}
