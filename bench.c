#include <stddef.h>
#include <assert.h>
#include <stdio.h>
#include <sys/time.h>

#include "api.h"

static uint64_t microseconds() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec*(uint64_t)1000000 + tv.tv_usec;
}

static double bench_inplace_add(size_t trials) {
	tensor_t* a = rand_tensor(Shape(50277), "f32");
	tensor_t* b = rand_tensor(Shape(50277), "f32");

	uint64_t start = microseconds();

	for (size_t i = 0; i < trials; i++)
		add_inplace(a, b);

	uint64_t elapsed = microseconds() - start;

	return (double)elapsed / (double)trials;
}

static double bench_dot(size_t trials) {
	tensor_t* a = rand_tensor(Shape(50277), "f32");
	tensor_t* b = rand_tensor(Shape(50277), "f32");

	f32 acc = 0.0;

	uint64_t start = microseconds();

	for (size_t i = 0; i < trials; i++)
		acc += dot_f32(a, b);

	uint64_t elapsed = microseconds() - start;

	return (double)elapsed / (double)trials;
}

static void run_bench(const char* identifier, double fn(size_t)) {
	printf("%s: %.1lf us/iter\n", identifier, fn(100000));
}

#define bench(X) run_bench(#X, X)

int main() {
	bench(bench_inplace_add);
	bench(bench_dot);
}
