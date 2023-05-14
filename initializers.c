#include <stdlib.h>
#include <assert.h>

#include "simd.h"
#include "tensor_t.h"
#include "initializers.h"

#define FOR_EVERY_ELEMENT(expr) for (size_t i = 0, element_count = tensor->element_count; i < element_count; i++) expr

// ============================================================================
// f32
// ============================================================================

tensor_t* zeros_init_f32_tensor(tensor_t* tensor) {
	FOR_EVERY_ELEMENT( tensor->storage.f32[i] = 0.0 );

	return tensor;
}

tensor_t* ones_init_f32_tensor(tensor_t* tensor) {
	FOR_EVERY_ELEMENT( tensor->storage.f32[i] = 1.0 );

	return tensor;
}

tensor_t* rand_init_f32_tensor(tensor_t* tensor) {
	FOR_EVERY_ELEMENT( tensor->storage.f32[i] = (double)rand() / (double)RAND_MAX );

	return tensor;
}

tensor_t* array_init_f32_tensor(tensor_t* tensor, size_t array_size, f32 array[]) {
	assert(tensor->element_count == array_size);

	FOR_EVERY_ELEMENT( tensor->storage.f32[i] = array[i] );

	return tensor;
}

// ============================================================================
// i32
// ============================================================================

tensor_t* zeros_init_i32_tensor(tensor_t* tensor) {
	FOR_EVERY_ELEMENT( tensor->storage.i32[i] = 0 );

	return tensor;
}

tensor_t* ones_init_i32_tensor(tensor_t* tensor) {
	FOR_EVERY_ELEMENT( tensor->storage.i32[i] = 1 );

	return tensor;
}

tensor_t* rand_init_i32_tensor(tensor_t* tensor) {
	FOR_EVERY_ELEMENT( tensor->storage.i32[i] = rand() );

	return tensor;
}

tensor_t* array_init_i32_tensor(tensor_t* tensor, size_t array_size, i32 array[]) {
	assert(tensor->element_count == array_size);

	FOR_EVERY_ELEMENT( tensor->storage.i32[i] = array[i] );

	return tensor;
}
