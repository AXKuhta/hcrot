#include <stddef.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "simd.h"
#include "tensor_t.h"
#include "ops.h"

// Beware of pointer lt/gt comparison undefined behavior 
static int no_overlap(tensor_t* a, tensor_t* b) {
	return a->storage.memory != b->storage.memory;
}

static int same_size(tensor_t* a, tensor_t* b) {
	return a->storage_size == b->storage_size;
}

void tensor_f32_add_f32(tensor_t* a, tensor_t* b) {
	assert(no_overlap(a, b));
	assert(same_size(a, b));

	f32_add_f32(a->storage.f32, b->storage.f32, a->storage_size);
}

f32 tensor_dot_f32(tensor_t* a, tensor_t* b) {
	assert(a->dimensions == 1 && b->dimensions == 1);
	assert(no_overlap(a, b));
	assert(same_size(a, b));

	return dot_f32_f32(a->storage.f32, b->storage.f32, a->storage_size);
}
