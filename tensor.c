#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "tensor_t.h"

static size_t apply_padding(size_t size) {
	return size + size % (16 * 4);
}

static size_t datatype_size(const char* datatype) {
	if (strcmp(datatype, "f32") == 0) { return 4; }
	else if (strcmp(datatype, "i32") == 0) { return 4; }
	else {
		printf("Unknown datatype: [%s]\n", datatype);
		exit(-1);
		return 0;
	}
}

static size_t tensor_size(const tensor_t* tensor) {
	size_t element_size = datatype_size(tensor->storage.datatype);
	size_t elements = 1;

	for (int i = 0; i < tensor->shape_dimensions; i++)
		elements *= tensor->shape[i].size;

	return apply_padding(elements * element_size);
}

static void default_stride(tensor_t* tensor) {
	size_t stride = 1;

	for (int i = tensor->shape_dimensions - 1; i >= 0; i--) {
		tensor->shape[i].stride = stride;
		stride *= tensor->shape[i].size;
	}
}

tensor_t* init_tensor(int shape_dimensions, size_t shape[], char* datatype) {
	tensor_t* tensor = malloc(sizeof(tensor_t) + sizeof(shape_dimension_t)*shape_dimensions);
	tensor->shape_dimensions = shape_dimensions;
	
	for (int i = 0; i < shape_dimensions; i++) {
		tensor->shape[i] = (shape_dimension_t){ .size = shape[i], .stride = 0 };
		assert(shape[i] > 0);
	}

	default_stride(tensor);

	tensor->storage.datatype = datatype;
	tensor->storage.size = tensor_size(tensor);
	tensor->storage.memory = malloc(tensor->storage.size);

	return tensor;
}

void debug_tensor(tensor_t* tensor) {
	printf("tensor(Shape(");

	for (int i = 0; i < tensor->shape_dimensions; i++)
		printf("%zu%s", tensor->shape[i].size, tensor->shape_dimensions - i > 1 ? ", " : "");

	printf("), \"%s\")\n", tensor->storage.datatype);
}
