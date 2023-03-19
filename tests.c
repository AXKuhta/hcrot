#include <stddef.h>
#include <stdio.h>

#include "api.h"

int main() {
	tensor_t* x = init_tensor(Shape(4, 16), "f32");
	tensor_t* y = init_tensor(Shape(4, 16), "f32");

	add_inplace(y, x);

	debug_tensor(y);

	printf("[3]: %zu\n", linear_index_dense(y, Shape(3)));
	printf("[1, 3]: %zu\n", linear_index_dense(y, Shape(1, 3)));
}
