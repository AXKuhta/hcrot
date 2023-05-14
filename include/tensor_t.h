
// ============================================================================
// SHAPING + INITIALIZERS
// ============================================================================

// Shape(4, 16) => 2, (size_t[]){4, 16}
#define Array(datatype, ...) (sizeof((datatype[]){__VA_ARGS__}) / sizeof(datatype)), ((datatype[]){__VA_ARGS__})
#define Shape(...) Array(size_t, __VA_ARGS__)
#define Index(...) Array(size_t, __VA_ARGS__)

#define Array_f32(...) Array(f32, __VA_ARGS__)
#define Array_i32(...) Array(i32, __VA_ARGS__)

// ============================================================================
// TENSOR STRUCTURE
// ============================================================================

typedef struct tensor_t {
	union {
		void* memory;
		f32* f32;
		i32* i32;
		u8* u8;
	} storage;

	size_t element_count;
	size_t element_size;
	size_t dimensions;

	struct {
		size_t size;
		size_t stride;
	} shape[];
} tensor_t;

// ============================================================================
// TENSOR CREATION
// ============================================================================

// Size annotations for compiler to warn you if you go out of bounds
// At least on static indices
static __attribute__((unused)) tensor_t* alloc_storage(size_t element_size, tensor_t* tensor) {
	tensor->storage.memory = malloc(element_size * tensor->element_count);
	tensor->element_size = element_size;

	return tensor;
}

// ============================================================================
// PROTOTYPES
// ============================================================================

tensor_t* alloc_tensor(size_t shape_dimensions, size_t shape[]);
void free_tensor(tensor_t* tensor);
tensor_t* clone_tensor_struct(tensor_t* tensor);
void* contiguous_storage(tensor_t* tensor);

// ============================================================================
// INDEXING
// ============================================================================

static __attribute__((unused)) size_t linear_index(const tensor_t* tensor, size_t index_dimensions, size_t index[]) {
	assert(tensor->dimensions >= index_dimensions);

	size_t offset = tensor->dimensions - index_dimensions;
	size_t linear_index = 0;

	for (size_t i = 0; i < index_dimensions; i++) {
		linear_index += index[i] * tensor->shape[i + offset].stride;
	}

	return linear_index;
}

#define rw(datatype, x, ...) *(x->storage.datatype + linear_index(x, Index(__VA_ARGS__)))
