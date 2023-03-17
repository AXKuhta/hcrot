#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "tensor_t.h"

static size_t apply_padding(size_t size) {
	return size + size % (16 * 4);
}

size_t tensor_size(const tensor_t* tensor) {
	size_t element_size = 0;
	size_t elements = 1;

	for (int i = 0; i < tensor->dimensions; i++)
		elements *= tensor->shape[i];

	if (strcmp(tensor->datatype, "f32") == 0) { element_size = 4; }
	else if (strcmp(tensor->datatype, "i32") == 0) { element_size = 4; }
	else {
		printf("Unknown datatype: [%s]\n", tensor->datatype);
		exit(-1);
	}

	return apply_padding(elements * element_size);
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
