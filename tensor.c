#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "simd.h"
#include "tensor_t.h"

static void default_stride(tensor_shape_t* shape) {
	size_t stride = 1;

	for (int i = shape->dimensions - 1; i >= 0; i--) {
		shape->arr[i].stride = stride;
		stride *= shape->arr[i].size;
	}
}

void_tensor* _alloc_tensor(size_t shape_dimensions, size_t shape[]) {
	void_tensor* tensor = malloc(sizeof(void_tensor) + sizeof(shape_arr_t)*shape_dimensions);
	tensor->shape.dimensions = shape_dimensions;
	size_t elements = 1;

	for (size_t i = 0; i < shape_dimensions; i++) {
		tensor->shape.arr[i] = (shape_arr_t){
			.size = shape[i],
			.stride = 0
		};

		elements *= shape[i];
		assert(shape[i] > 0);
	}

	tensor->shape.elements = elements;
	default_stride(&tensor->shape);

	return tensor;
}

// ============================================================================
// f32_tensor
// ============================================================================

f32_tensor* zeros_init_f32_tensor(f32_tensor* tensor) {
	for (size_t i = 0, elements = tensor->shape.elements; i < elements; i++)
		tensor->storage[i] = 0.0;

	return tensor;
}

f32_tensor* ones_init_f32_tensor(f32_tensor* tensor) {
	for (size_t i = 0, elements = tensor->shape.elements; i < elements; i++)
		tensor->storage[i] = 1.0;

	return tensor;
}

f32_tensor* rand_init_f32_tensor(f32_tensor* tensor) {
	for (size_t i = 0, elements = tensor->shape.elements; i < elements; i++)
		tensor->storage[i] = (double)rand() / (double)RAND_MAX;

	return tensor;
}

// ============================================================================
// i32_tensor
// ============================================================================

i32_tensor* zeros_init_i32_tensor(i32_tensor* tensor) {
	for (size_t i = 0, elements = tensor->shape.elements; i < elements; i++)
		tensor->storage[i] = 0;

	return tensor;
}

i32_tensor* ones_init_i32_tensor(i32_tensor* tensor) {
	for (size_t i = 0, elements = tensor->shape.elements; i < elements; i++)
		tensor->storage[i] = 1;

	return tensor;
}

i32_tensor* rand_init_i32_tensor(i32_tensor* tensor) {
	for (size_t i = 0, elements = tensor->shape.elements; i < elements; i++)
		tensor->storage[i] = rand();

	return tensor;
}

void free_tensor(void* tensor) {
	free(((void_tensor*)tensor)->storage);
	free(tensor);
}
