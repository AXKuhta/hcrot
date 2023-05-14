#include <stdlib.h>
#include <assert.h>

#include "simd.h"
#include "tensor_t.h"
#include "transpose.h"

static void transpose_tensor(tensor_t* tensor, void swap_fn(tensor_t* tensor, size_t a, size_t b)) {
	assert(tensor->dimensions == 2);

	size_t rows = tensor->shape[0].size;
	size_t cols = tensor->shape[1].size;

	for (size_t i = 0; i < cols; i++)
		for (size_t j = i + 1; j < rows; j++)
			swap_fn(tensor, i*cols + j, j*cols + i);

	tensor->shape[0].size = cols;
	tensor->shape[1].size = rows;

	tensor->shape[0].stride = rows;
	tensor->shape[1].stride = 1;
}

static void swap_f32(tensor_t* tensor, size_t a, size_t b) {
	f32* ptr = tensor->storage.f32;
	f32 tmp = ptr[a];
	ptr[a] = ptr[b];
	ptr[b] = tmp;
}

void transpose_f32_tensor(tensor_t* tensor) { transpose_tensor(tensor, swap_f32); }
