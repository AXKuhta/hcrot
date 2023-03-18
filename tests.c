#include <stddef.h>

#include "tensor_t.h"
#include "ops.h"

int main() {
	tensor_t* x = init_tensor(Shape(4, 16), "f32");
	tensor_t* y = init_tensor(Shape(4, 16), "f32");

	add_inplace(y, x);

	debug_tensor(y);
}
