#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <stdio.h>

#include "api.h"

void test_set_get_1() {
	f32_tensor* x = zeros_tensor(f32, 16, 3, 320, 240);

	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 320; k++)
				for (int l = 0; l < 240; l++)
					idx(x, i, j, k, l) = i+j+k+l;


	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 320; k++)
				for (int l = 0; l < 240; l++)
					assert(idx(x, i, j, k, l) == i+j+k+l);

	free_tensor(x);
}

void test_set_get_2() {
	f32_tensor* x = zeros_tensor(f32, 3, 3);

	idx(x, 0, 0) = 8;
	idx(x, 0, 1) = 1;
	idx(x, 0, 2) = 5;

	idx(x, 1, 0) = 2;
	idx(x, 1, 1) = 9;
	idx(x, 1, 2) = 7;

	idx(x, 2, 0) = 2;
	idx(x, 2, 1) = 4;
	idx(x, 2, 2) = 6;

	f32 det = 0.0;

	for (size_t i = 0; i < 3; i++) {
		f32 pri_diag = 1.0;
		f32 sec_diag = 1.0;

		for (size_t j = 0; j < 3; j++) {
			size_t c = j;
			size_t r = (j + i) % 3;
			
			pri_diag *= idx(x, r, c);
			sec_diag *= idx(x, r, 2 - c);
		}

		det += pri_diag - sec_diag;
	}

	assert(det == 160.0);

	free_tensor(x);
}
/*
void test_dot_1() {
	tensor_t* a = zeros_tensor(Shape(5), "f32");
	tensor_t* b = zeros_tensor(Shape(5), "f32");

	set_f32(a, Index(0), 6);
	set_f32(a, Index(1), 9);
	set_f32(a, Index(2), 9);
	set_f32(a, Index(3), 8);
	set_f32(a, Index(4), 6);

	set_f32(b, Index(0), 2);
	set_f32(b, Index(1), 2);
	set_f32(b, Index(2), 3);
	set_f32(b, Index(3), 4);
	set_f32(b, Index(4), 5);

	assert(dot_f32(a, b) == 119.0);
}
*/
void run_tests() {
	test_set_get_1();
	test_set_get_2();
	//test_dot_1();

	printf("Self-testing OK\n");
}

int main() {
	run_tests();
}
