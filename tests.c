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
	tensor_t* x = array_tensor(Shape(3, 3), Array_f32(
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
	tensor_t* a = array_tensor(Shape(2, 2), Array_f32(
		1, 2,
		3, 4
	));

	tensor_t* b = array_tensor(Shape(2, 2), Array_f32(
		5, 6,
		7, 8
	));

	tensor_f32_add_f32(a, b);

	assert(rw(f32, a, 0, 0) == 6 && rw(f32, a, 0, 1) == 8 && rw(f32, a, 1, 0) == 10 && rw(f32, a, 1, 1) == 12);

	free_tensor(a);
	free_tensor(b);
}

void test_inplace_sub_1() {
	tensor_t* a = array_tensor(Shape(2, 2), Array_f32(
		1, 2,
		3, 4
	));

	tensor_t* b = array_tensor(Shape(2, 2), Array_f32(
		5, 6,
		7, 8
	));

	tensor_f32_sub_f32(a, b);

	assert(rw(f32, a, 0, 0) == -4 && rw(f32, a, 0, 1) == -4 && rw(f32, a, 1, 0) == -4 && rw(f32, a, 1, 1) == -4);

	free_tensor(a);
	free_tensor(b);
}

void test_inplace_mul_1() {
	tensor_t* a = array_tensor(Shape(2, 2), Array_f32(
		1, 2,
		3, 4
	));

	tensor_t* b = array_tensor(Shape(2, 2), Array_f32(
		5, 6,
		7, 8
	));

	tensor_f32_mul_f32(a, b);

	assert(rw(f32, a, 0, 0) == 5 && rw(f32, a, 0, 1) == 12 && rw(f32, a, 1, 0) == 21 && rw(f32, a, 1, 1) == 32);

	free_tensor(a);
	free_tensor(b);
}

void test_inplace_div_1() {
	tensor_t* a = array_tensor(Shape(2, 2), Array_f32(
		1, 2,
		3, 4
	));

	tensor_t* b = array_tensor(Shape(2, 2), Array_f32(
		5, 6,
		7, 8
	));

	tensor_f32_div_f32(a, b);

	assert(rw(f32, a, 0, 0) == 1.0f/5.0f && rw(f32, a, 0, 1) == 1.0f/3.0f && rw(f32, a, 1, 0) == 3.0f/7.0f && rw(f32, a, 1, 1) == 4.0f/8.0f);

	free_tensor(a);
	free_tensor(b);
}

void test_inplace_add_x_1() {
	tensor_t* a = array_tensor(Shape(2, 2), Array_f32(
		1, 2,
		3, 4
	));

	tensor_f32_add_x(a, 3);

	assert(rw(f32, a, 0, 0) == 4 && rw(f32, a, 0, 1) == 5 && rw(f32, a, 1, 0) == 6 && rw(f32, a, 1, 1) == 7);

	free_tensor(a);
}

void test_inplace_sub_x_1() {
	tensor_t* a = array_tensor(Shape(2, 2), Array_f32(
		1, 2,
		3, 4
	));

	tensor_f32_sub_x(a, 3);

	assert(rw(f32, a, 0, 0) == -2 && rw(f32, a, 0, 1) == -1 && rw(f32, a, 1, 0) == 0 && rw(f32, a, 1, 1) == 1);

	free_tensor(a);
}

void test_inplace_mul_x_1() {
	tensor_t* a = array_tensor(Shape(2, 2), Array_f32(
		1, 2,
		3, 4
	));

	tensor_f32_mul_x(a, 3);

	assert(rw(f32, a, 0, 0) == 3 && rw(f32, a, 0, 1) == 6 && rw(f32, a, 1, 0) == 9 && rw(f32, a, 1, 1) == 12);

	free_tensor(a);
}

void test_inplace_div_x_1() {
	tensor_t* a = array_tensor(Shape(2, 2), Array_f32(
		1, 2,
		3, 4
	));

	tensor_f32_div_x(a, 2);

	assert(rw(f32, a, 0, 0) == 0.5 && rw(f32, a, 0, 1) == 1 && rw(f32, a, 1, 0) == 1.5 && rw(f32, a, 1, 1) == 2);

	free_tensor(a);
}

void test_min_max_1() {
	tensor_t* x = array_tensor(Shape(3, 3), Array_f32(
		8, 1, 5,
		2, 9, 7,
		2, 4, 6
	));

	assert(tensor_min_f32(x) == 1.0f);
	assert(tensor_max_f32(x) == 9.0f);

	free_tensor(x);
}

void test_sum_1() {
	tensor_t* x = array_tensor(Shape(3, 3), Array_f32(
		8, 1, 5,
		2, 9, 7,
		2, 4, 6
	));

	assert(tensor_sum_f32(x) == 44.0f);

	free_tensor(x);
}

void test_mean_1() {
	tensor_t* x = array_tensor(Shape(3, 3), Array_f32(
		8, 1, 5,
		2, 9, 7,
		2, 4, 6
	));

	assert(tensor_mean_f32(x) == 44.0f/9.0f);

	free_tensor(x);
}

void test_dot_1() {
	tensor_t* a = array_tensor(Shape(5), Array_f32(
		6, 9, 9, 8, 6
	));

	tensor_t* b = array_tensor(Shape(5), Array_f32(
		2, 2, 3, 4, 5
	));

	assert(tensor_dot_f32(a, b) == 119.0);

	free_tensor(a);
	free_tensor(b);
}

void test_transpose_1() {
	tensor_t* a = array_tensor(Shape(3, 3), Array_f32(
		1, 2, 3,
		4, 5, 6,
		7, 8, 9
	));

	tensor_t* b = array_tensor(Shape(3, 3), Array_f32(
		1, 4, 7,
		2, 5, 8,
		3, 6, 9
	));

	tensor_t* t = transpose_tensor(a);

	for (size_t i = 0; i < 3; i++)
		for (size_t j = 0; j < 3; j++)
			assert(rw(f32, t, i, j) == rw(f32, b, i, j));

	free_tensor(a);
	free_tensor(b);
	free_tensor(t);
}

void test_transpose_2() {
	tensor_t* a = array_tensor(Shape(2, 5), Array_f32(
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10
	));

	tensor_t* b = array_tensor(Shape(5, 2), Array_f32(
		1, 6,
		2, 7,
		3, 8,
		4, 9,
		5, 10
	));

	tensor_t* t_b = transpose_tensor(a);
	tensor_t* t_a = transpose_tensor(t_b);

	for (size_t i = 0; i < 5; i++)
		for (size_t j = 0; j < 2; j++)
			assert(rw(f32, t_b, i, j) == rw(f32, b, i, j));

	for (size_t i = 0; i < 2; i++)
		for (size_t j = 0; j < 5; j++)
			assert(rw(f32, t_a, i, j) == rw(f32, a, i, j));

	free_tensor(a);
	free_tensor(b);
	free_tensor(t_b);
	free_tensor(t_a);
}

void run_tests() {
	test_set_get_1();
	test_set_get_2();
	test_inplace_add_1();
	test_inplace_sub_1();
	test_inplace_mul_1();
	test_inplace_div_1();
	test_inplace_add_x_1();
	test_inplace_sub_x_1();
	test_inplace_mul_x_1();
	test_inplace_div_x_1();
	test_min_max_1();
	test_sum_1();
	test_mean_1();
	test_dot_1();
	test_transpose_1();
	test_transpose_2();

	printf("Self-testing OK\n");
}

int main() {
	run_tests();
}
