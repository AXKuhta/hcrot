#include <stddef.h>
#include <assert.h>
#include <stdio.h>

#include "api.h"

void test_set_get() {
	tensor_t* x = init_tensor(Shape(4), "f32");

	for (int i = 0; i < 4; i++)
		set_f32(x, Index(i), i);

	for (int i = 0; i < 4; i++)
		assert(get_f32(x, Index(i)) == i);
}

int main() {
	test_set_get();

	printf("Self-testing OK\n");

	return 0;
}
