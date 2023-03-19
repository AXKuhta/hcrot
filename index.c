#include <stddef.h>
#include <assert.h>

#include "tensor_t.h"

size_t linear_index_dense(const tensor_t* tensor, int index_dimensions, size_t index[]) {
	assert(tensor->shape_dimensions >= index_dimensions);

	size_t offset = tensor->shape_dimensions - index_dimensions;
	size_t linear_index = 0;

	for (int i = 0; i < index_dimensions; i++) {
		linear_index += index[i] * tensor->shape[i + offset].stride;
	}

	return linear_index;
}
