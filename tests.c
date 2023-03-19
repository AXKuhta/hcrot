#include <stddef.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "api.h"

void test_set_get() {
	tensor_t* x = init_tensor(Shape(4), "f32");

	for (int i = 0; i < 4; i++)
		set_f32(x, Index(i), i);

	for (int i = 0; i < 4; i++)
		assert(get_f32(x, Index(i)) == i);
}

void set_get() {
	tensor_t* x = init_tensor(Shape(16, 3, 320, 240), "f32");

	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 320; k++)
				for (int l = 0; l < 240; l++)
					set_f32(x, Index(i, j, k, l), i+j+k+l);


	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 320; k++)
				for (int l = 0; l < 240; l++)
					assert(get_f32(x, Index(i, j, k, l)) == i+j+k+l);

	free_tensor(x);
}

void benchmark(const char* identifier, void fn(void)) {
	double start = clock();

	for (int i = 0; i < 100; i++) {
		fn();
	}
	
	double end = clock();
	double clock_per_iter = (end - start) / 100.0;

	printf("%s: %.1lf\n", identifier, clock_per_iter);
}

#define bench(X) benchmark(#X, X)

void run_tests() {
	test_set_get();
}

void run_benchmarks() {
	bench(set_get);
}

int main() {
	run_tests();

	printf("Self-testing OK\n");

	run_benchmarks();

	return 0;
}
