#include <stddef.h>
#include <assert.h>
#include <stdio.h>

#include "api.h"

void test_set_get_1() {
	tensor_t* x = init_tensor(Shape(4), "f32");

	for (int i = 0; i < 4; i++)
		set_f32(x, Index(i), i);

	for (int i = 0; i < 4; i++)
		assert(get_f32(x, Index(i)) == i);
}

void test_set_get_2() {
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
}

int main() {
	test_set_get_1();
	test_set_get_2();

	printf("Self-testing OK\n");

	return 0;
}
