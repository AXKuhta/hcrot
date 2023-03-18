#include <stddef.h>
#include <assert.h>
#include <string.h>

#include "tensor_t.h"
#include "simd.h"
#include "ops.h"

// Beware of pointer lt/gt comparison undefined behavior 
static int no_overlap(tensor_t* a, tensor_t* b) {
	return a->storage.memory != b->storage.memory;
}

static int same_size(tensor_t* a, tensor_t* b) {
	return a->storage.size == b->storage.size;
}

static int f32_tensor(tensor_t* a) {
	return 0 == strcmp(a->storage.datatype, "f32");
}

static int i32_tensor(tensor_t* a) {
	return 0 == strcmp(a->storage.datatype, "i32");
}

void add_inplace(tensor_t* a, tensor_t* b) {
	assert(no_overlap(a, b));
	assert(same_size(a, b));

	if (f32_tensor(a) && f32_tensor(b)) {
		f32_add_f32((f32*)a->storage.memory, (f32*)b->storage.memory, a->storage.size);
		return;
	}

	if (i32_tensor(a) && i32_tensor(b)) {
		return;
	}
}

