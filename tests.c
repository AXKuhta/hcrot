#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <stdio.h>

#include "api.h"

void test_set_get_1() {
	tensor_t* x = zeros_tensor(f32, 16, 3, 320, 240);

	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 320; k++)
				for (int l = 0; l < 240; l++)
					rw(f32, x, i, j, k, l) = i+j+k+l;


	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 320; k++)
				for (int l = 0; l < 240; l++)
					assert(rw(f32, x, i, j, k, l) == i+j+k+l);

	free_tensor(x);
}

void test_set_get_2() {
	tensor_t* x = array_tensor(f32, Shape(3, 3), Array_f32(
		8, 1, 5,
		2, 9, 7,
		2, 4, 6
	));

	f32 det = 0.0;

	for (size_t i = 0; i < 3; i++) {
		f32 pri_diag = 1.0;
		f32 sec_diag = 1.0;

		for (size_t j = 0; j < 3; j++) {
			size_t c = j;
			size_t r = (j + i) % 3;
			
			pri_diag *= rw(f32, x, r, c);
			sec_diag *= rw(f32, x, r, 2 - c);
		}

		det += pri_diag - sec_diag;
	}

	assert(det == 160.0);

	free_tensor(x);
}

void test_inplace_add_1() {
	tensor_t* a = array_tensor(f32, Shape(2, 2), Array_f32(
		1, 2,
		3, 4
	));

	tensor_t* b = array_tensor(f32, Shape(2, 2), Array_f32(
		5, 6,
		7, 8
	));

	tensor_f32_add_f32(a, b);

	assert(rw(f32, a, 0, 0) == 6 && rw(f32, a, 0, 1) == 8 && rw(f32, a, 1, 0) == 10 && rw(f32, a, 1, 1) == 12);

	free_tensor(a);
	free_tensor(b);
}

void test_dot_1() {
	tensor_t* a = array_tensor(f32, Shape(5), Array_f32(
		6, 9, 9, 8, 6
	));

	tensor_t* b = array_tensor(f32, Shape(5), Array_f32(
		2, 2, 3, 4, 5
	));

	assert(tensor_dot_f32(a, b) == 119.0);

	free_tensor(a);
	free_tensor(b);
}

void run_tests() {
	test_set_get_1();
	test_set_get_2();
	test_inplace_add_1();
	test_dot_1();

	printf("Self-testing OK\n");
}

int main() {
	run_tests();
}
