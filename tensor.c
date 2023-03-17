#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

typedef struct tensor_t {
	char* datatype;
	void* storage;
	int dimensions;
	int shape[];
} tensor_t;

size_t tensor_size(const tensor_t* tensor) {
	size_t elements = 1;

	for (int i = 0; i < tensor->dimensions; i++)
		elements *= tensor->shape[i];

	if (strcmp(tensor->datatype, "f32") == 0) return elements*4;

	__builtin_unreachable();
}

tensor_t* init_tensor(int dimensions, int shape[], char* datatype) {
	tensor_t* tensor = malloc(sizeof(tensor_t) + sizeof(int)*dimensions);

	tensor->datatype = datatype;
	tensor->dimensions = dimensions;
	
	for (int i = 0; i < dimensions; i++) {
		tensor->shape[i] = shape[i];
		assert(shape[i] > 0);
	}

	tensor->storage = malloc(tensor_size(tensor));

	return tensor;
}

void debug_tensor(tensor_t* tensor) {
	printf("tensor(Shape(");

	for (int i = 0; i < tensor->dimensions; i++)
		printf("%d%s", tensor->shape[i], tensor->dimensions - i > 1 ? ", " : "");

	printf("), \"%s\")\n", tensor->datatype);
}

// Shape(4, 16) => 2, (int[]){4, 16}
#define Shape(...) (sizeof((int[]){__VA_ARGS__}) / sizeof(int)), ((int[]){__VA_ARGS__})

int main() {
	tensor_t* x = init_tensor(Shape(4, 16), "f32");
	debug_tensor(x);
}
