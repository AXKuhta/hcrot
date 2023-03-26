#include <stddef.h>
#include <assert.h>
#include <stdio.h>

#include "api.h"

void test_set_get_1() {
	tensor_t* x = zeros_tensor(Shape(16, 3, 320, 240), "f32");

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

void test_set_get_2() {
	tensor_t* x = zeros_tensor(Shape(3, 3), "f32");
	
	set_f32(x, Index(0, 0), 8);
	set_f32(x, Index(0, 1), 1);
	set_f32(x, Index(0, 2), 5);

	set_f32(x, Index(1, 0), 2);
	set_f32(x, Index(1, 1), 9);
	set_f32(x, Index(1, 2), 7);

	set_f32(x, Index(2, 0), 2);
	set_f32(x, Index(2, 1), 4);
	set_f32(x, Index(2, 2), 6);

	f32 det = 0.0;

	for (size_t i = 0; i < 3; i++) {
		f32 pri_diag = 1.0;
		f32 sec_diag = 1.0;

		for (size_t j = 0; j < 3; j++) {
			size_t c = j;
			size_t r = (j + i) % 3;
			
			pri_diag *= get_f32(x, Index(r, c));
			sec_diag *= get_f32(x, Index(r, 2 - c));
		}

		det += pri_diag - sec_diag;
	}

	assert(det == 160.0);

	free_tensor(x);
}

void run_tests() {
	test_set_get_1();
	test_set_get_2();
}

int main() {
	run_tests();

	printf("Self-testing OK\n");

	return 0;
}
