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

static void contiguous_recursive_fn(tensor_t* dst, tensor_t* src, size_t base, size_t dim) {
	size_t element_size = src->element_size;
	size_t dimensions = src->dimensions;
	size_t stride = dst->shape[dim].stride;
	size_t size = src->shape[dim].size;

	static size_t src_idx;

	if (dim == 0)
		src_idx = 0;

	if (dim == dimensions) {
		memcpy(dst->storage.u8 + base*element_size, src->storage.u8 + src_idx*element_size, element_size);
		src_idx++;
	} else {
		for (size_t i = 0; i < size; i++) {
			contiguous_recursive_fn(dst, src, base + i*stride, dim + 1);
		}
	}
}

tensor_t* contiguous_tensor(tensor_t* tensor) {
	size_t struct_size = sizeof(tensor_t) + 2*sizeof(size_t) * tensor->dimensions;

	tensor_t* clone = malloc(struct_size);
	memcpy(clone, tensor, struct_size);

	clone->storage.memory = malloc(tensor->element_count * tensor->element_size);

	contiguous_recursive_fn(clone, tensor, 0, 0);

	return clone;
}
