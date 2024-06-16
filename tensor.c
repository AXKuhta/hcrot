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

tensor_t* alloc_tensor(const char* datatype, size_t shape_dimensions, size_t shape[]) {
	tensor_t* tensor = malloc(sizeof(tensor_t) + sizeof(size_t)*2*shape_dimensions);
	tensor->dimensions = shape_dimensions;
	tensor->datatype = datatype;
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

static tensor_t* clone_tensor_struct(tensor_t* tensor) {
	size_t struct_size = sizeof(tensor_t) + 2*sizeof(size_t) * tensor->dimensions;
	tensor_t* clone = malloc(struct_size);
	memcpy(clone, tensor, struct_size);

	return clone;
}

static void* contiguous_recursive_fn(tensor_t* tensor, size_t base, size_t dim) {
	size_t element_size = tensor->element_size;
	size_t dimensions = tensor->dimensions;
	size_t stride = tensor->shape[dim].stride;
	size_t size = tensor->shape[dim].size;

	static size_t src_idx;
	static u8* dst;

	if (dim == 0) {
		dst = malloc(tensor->element_count * tensor->element_size);
		src_idx = 0;
	}

	if (dim == dimensions) {
		memcpy(dst + base*element_size, tensor->storage.u8 + src_idx*element_size, element_size);
		src_idx++;
	} else {
		for (size_t i = 0; i < size; i++) {
			contiguous_recursive_fn(tensor, base + i*stride, dim + 1);
		}
	}

	return dst;
}

static void swap(size_t* a, size_t* b) {
	size_t t = *a; *a = *b; *b = t;
}

tensor_t* transpose_tensor(tensor_t* tensor) {
	assert(tensor->dimensions == 2);

	size_t rows = tensor->shape[0].size;

	tensor_t* transposed = clone_tensor_struct(tensor);

	transposed->shape[0].stride = 1;
	transposed->shape[1].stride = rows;

	transposed->storage.memory = contiguous_recursive_fn(transposed, 0, 0);;

	swap(&transposed->shape[0].stride, &transposed->shape[1].stride);
	swap(&transposed->shape[0].size, &transposed->shape[1].size);

	return transposed;
}
