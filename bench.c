#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "api.h"

static uint64_t nanoseconds() {
	struct timespec ts;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
	return ts.tv_sec*1000000000ull + ts.tv_nsec;
}

static double bench_inplace_add(size_t trials) {
	tensor_t* a = rand_tensor(Shape(50277), "f32");
	tensor_t* b = rand_tensor(Shape(50277), "f32");

	uint64_t start = nanoseconds();

	for (size_t i = 0; i < trials; i++)
		add_inplace(a, b);

	uint64_t elapsed = nanoseconds() - start;

	return (double)elapsed / (double)trials;
}

static double bench_dot(size_t trials) {
	tensor_t* a = rand_tensor(Shape(50277), "f32");
	tensor_t* b = rand_tensor(Shape(50277), "f32");

	f32 acc = 0.0;

	uint64_t start = nanoseconds();

	for (size_t i = 0; i < trials; i++)
		acc += dot_f32(a, b);

	uint64_t elapsed = nanoseconds() - start;

	return (double)elapsed / (double)trials;
}

static void run_bench(const char* identifier, double fn(size_t)) {
	printf("%s: %.0lf us/iter\n", identifier, fn(100000)/1000.0);
}

#define bench(X) run_bench(#X, X)

int main() {
	bench(bench_inplace_add);
	bench(bench_dot);
}