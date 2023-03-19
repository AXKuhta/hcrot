#include <stddef.h>
#include <assert.h>

#include "tensor_t.h"
#include "simd.h"

static size_t linear_index_dense(const tensor_t* tensor, int index_dimensions, size_t index[]) {
	assert(tensor->shape_dimensions >= index_dimensions);

	size_t offset = tensor->shape_dimensions - index_dimensions;
	size_t linear_index = 0;

	for (int i = 0; i < index_dimensions; i++) {
		linear_index += index[i] * tensor->shape[i + offset].stride;
	}

	return linear_index;
}

#define SET_GET_FN(TYPE) 	void set_##TYPE(const tensor_t* tensor, int index_dimensions, size_t index[], TYPE x) {	\
								((TYPE*)tensor->storage.memory)[ linear_index_dense(tensor, index_dimensions, index) ] = x; \
							} \
							TYPE get_##TYPE(const tensor_t* tensor, int index_dimensions, size_t index[]) { \
								return ((TYPE*)tensor->storage.memory)[ linear_index_dense(tensor, index_dimensions, index) ]; \
							}

SET_GET_FN(f32)
SET_GET_FN(i32)
