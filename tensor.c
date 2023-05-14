#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "simd.h"
#include "tensor_t.h"

static void default_stride(tensor_t* tensor) {
	size_t stride = 1;

	for (int i = tensor->dimensions - 1; i >= 0; i--) {
		tensor->shape[i].stride = stride;
		stride *= tensor->shape[i].size;
	}
}

tensor_t* alloc_tensor(size_t shape_dimensions, size_t shape[]) {
	tensor_t* tensor = malloc(sizeof(tensor_t) + sizeof(size_t)*2*shape_dimensions);
	tensor->dimensions = shape_dimensions;
	size_t element_count = 1;

	for (size_t i = 0; i < shape_dimensions; i++) {
		assert(shape[i] > 0);
		element_count *= shape[i];
		tensor->shape[i].size = shape[i];
	}

	tensor->element_count = element_count;
	default_stride(tensor);

	return tensor;
}

void free_tensor(tensor_t* tensor) {
	free(tensor->storage.memory);
	free(tensor);
}
