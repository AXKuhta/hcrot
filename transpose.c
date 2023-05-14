#include <stdlib.h>
#include <assert.h>

#include "simd.h"
#include "tensor_t.h"
#include "transpose.h"

tensor_t* transpose_tensor(tensor_t* tensor) {
	assert(tensor->dimensions == 2);

	size_t rows = tensor->shape[0].size;
	size_t cols = tensor->shape[1].size;

	tensor->shape[0].stride = 1;
	tensor->shape[1].stride = rows;

	tensor_t* transposed = contiguous_tensor(tensor);

	tensor->shape[0].stride = cols;
	tensor->shape[1].stride = 1;

	transposed->shape[0].stride = rows;
	transposed->shape[1].stride = 1;

	transposed->shape[0].size = cols;
	transposed->shape[1].size = rows;

	return transposed;
}

// Fast path for rectangular matrices
// if (rows == cols)
//	for (size_t i = 0; i < cols; i++)
//		for (size_t j = i + 1; j < rows; j++)
//			swap_fn(tensor, i*cols + j, j*cols + i);
