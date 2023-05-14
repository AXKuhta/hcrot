#include <stdlib.h>
#include <assert.h>

#include "simd.h"
#include "tensor_t.h"
#include "transpose.h"

static void swap(size_t* a, size_t* b) {
	size_t t = *a; *a = *b; *b = t;
}

tensor_t* transpose_tensor(tensor_t* tensor) {
	assert(tensor->dimensions == 2);

	size_t rows = tensor->shape[0].size;

	tensor_t* transposed = clone_tensor_struct(tensor);

	transposed->shape[0].stride = 1;
	transposed->shape[1].stride = rows;

	transposed->storage.memory = contiguous_storage(transposed);

	swap(&transposed->shape[0].stride, &transposed->shape[1].stride);
	swap(&transposed->shape[0].size, &transposed->shape[1].size);

	return transposed;
}

// Fast path for rectangular matrices
// if (rows == cols)
//	for (size_t i = 0; i < cols; i++)
//		for (size_t j = i + 1; j < rows; j++)
//			swap_fn(tensor, i*cols + j, j*cols + i);
