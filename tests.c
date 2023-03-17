#include <stddef.h>

#include "tensor_t.h"

int main() {
	tensor_t* x = init_tensor(Shape(4, 16), "f32");
	debug_tensor(x);
}
