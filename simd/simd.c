#include <stddef.h>

static T add(T a, T b) { return a + b; }
static T sub(T a, T b) { return a - b; }
static T mul(T a, T b) { return a * b; }
static T div(T a, T b) { return a / b; }

#define flat __attribute__((flatten))

#define HEAD (count & ~(K_STRIDE - 1))
#define TAIL (count & (K_STRIDE - 1))
#define FOR_HEAD(expr) for (size_t i = 0; i < HEAD; i += K_STRIDE) for (size_t j = 0; j < K_STRIDE; j++) expr
#define FOR_TAIL(expr) for (size_t i = HEAD, j = 0; j < TAIL; j++) expr

// A[i] += B[i]
static void elementwise_inplace_ab(T* restrict a, T* restrict b, const size_t count, T fn(T a, T b)) {
	FOR_HEAD( a[i + j] = fn(a[i + j], b[i + j]) );
	FOR_TAIL( a[i + j] = fn(a[i + j], b[i + j]) );
}

// A[i] += x
static void elementwise_inplace_ax(T* a, T x, const size_t count, T fn(T a, T b)) {
	FOR_HEAD( a[i + j] = fn(a[i + j], x) );
	FOR_TAIL( a[i + j] = fn(a[i + j], x) );
}

// https://stackoverflow.com/questions/1253934/c-pre-processor-defining-for-generated-function-names
#define FN_(a, x, b) a ## _ ## x ## _ ## b
#define FN(a, x, b) FN_(a, x, b)

flat void FN(T, add, T) (T* a, T* b, const size_t count) { elementwise_inplace_ab(a, b, count, add); }
flat void FN(T, sub, T) (T* a, T* b, const size_t count) { elementwise_inplace_ab(a, b, count, sub); }
flat void FN(T, mul, T) (T* a, T* b, const size_t count) { elementwise_inplace_ab(a, b, count, mul); }
flat void FN(T, div, T) (T* a, T* b, const size_t count) { elementwise_inplace_ab(a, b, count, div); }

flat void FN(T, add, x) (T* a, T x, const size_t count) { elementwise_inplace_ax(a, x, count, add); }
flat void FN(T, sub, x) (T* a, T x, const size_t count) { elementwise_inplace_ax(a, x, count, sub); }
flat void FN(T, mul, x) (T* a, T x, const size_t count) { elementwise_inplace_ax(a, x, count, mul); }
flat void FN(T, div, x) (T* a, T x, const size_t count) { elementwise_inplace_ax(a, x, count, div); }

// acc += A[i] * B[i]
T FN(dot, T, T)(T* restrict a, T* restrict b, const size_t count) {
	T accs[K_STRIDE] = { (T)0 };
	T acc = 0.0;

	FOR_HEAD( accs[j] += a[i + j] * b[i + j] );

	for (size_t j = 0; j < K_STRIDE; j++)
		acc += accs[j];

	FOR_TAIL( acc += a[i + j] * b[i + j] );

	return acc;
}

// acc = fn(acc, A[i])
static T reduce(T* restrict x, const size_t count, T fn(T a, T b), T acc_init) {
	T accs[K_STRIDE];
	T acc = acc_init;

	for (size_t i = 0; i < K_STRIDE; i++)
		accs[i] = acc_init;

	FOR_HEAD( accs[j] = fn(x[i + j], accs[j]) );

	for (size_t i = 0; i < K_STRIDE; i++)
		acc = fn(accs[i], acc);

	FOR_TAIL( acc = fn(x[i + j], acc) );

	return acc;
}

static T sum(T a, T b) { return a + b; }

#undef FN
#undef FN_
#define FN_(a, x) a ## _ ## x
#define FN(a, x) FN_(a, x)

flat T FN(T, sum) (T* x, const size_t count) { return reduce(x, count, sum, 0); }

#ifndef NO_LT_GT
static T min(T a, T b) { return a < b ? a : b; }
static T max(T a, T b) { return a > b ? a : b; }

flat T FN(T, min) (T* x, const size_t count) { return reduce(x, count, min, x[0]); }
flat T FN(T, max) (T* x, const size_t count) { return reduce(x, count, max, x[0]); }
#endif
