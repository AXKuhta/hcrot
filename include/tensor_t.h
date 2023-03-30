
// ============================================================================
// SHAPING
// ============================================================================

// Shape(4, 16) => 2, (size_t[]){4, 16}
#define Shape(...) (sizeof((size_t[]){__VA_ARGS__}) / sizeof(size_t)), ((size_t[]){__VA_ARGS__})
#define Index(...) Shape(__VA_ARGS__)

// ============================================================================
// TENSOR STRUCTURE
// ============================================================================

typedef struct shape_arr_t {
	size_t size;
	size_t stride;
} shape_arr_t;

typedef struct tensor_shape_t {
	size_t elements;
	size_t dimensions;
	shape_arr_t arr[];
} tensor_shape_t;

#define DEFINE_TENSOR(T) \
typedef struct T##_tensor { \
	T* storage; \
	size_t storage_size; \
	tensor_shape_t shape; \
} T##_tensor;

DEFINE_TENSOR(void)
DEFINE_TENSOR(f32)
DEFINE_TENSOR(i32)

// ============================================================================
// TENSOR CREATION
// ============================================================================

static size_t apply_padding(size_t size) {
	return size + (-size % K_STRIDE);
}

// Size annotations for compiler to warn you if you go out of bounds
// At least on static indices
static void* alloc_storage(size_t element_size, void_tensor* tensor) {
	tensor->storage_size = apply_padding(element_size * tensor->shape.elements);
	tensor->storage = malloc(tensor->storage_size);

	return tensor;
}

#define alloc_tensor(datatype, ...) alloc_storage(sizeof(datatype), _alloc_tensor(Shape(__VA_ARGS__)))
#define zeros_tensor(datatype, ...) zeros_init_##datatype##_tensor(alloc_tensor(datatype, __VA_ARGS__))
#define ones_tensor(datatype, ...) ones_init_##datatype##_tensor(alloc_tensor(datatype, __VA_ARGS__))
#define rand_tensor(datatype, ...) rand_init_##datatype##_tensor(alloc_tensor(datatype, __VA_ARGS__))

// ============================================================================
// PROTOTYPES
// ============================================================================

void_tensor* _alloc_tensor(size_t shape_dimensions, size_t shape[]);

f32_tensor* zeros_init_f32_tensor(f32_tensor* tensor);
f32_tensor* ones_init_f32_tensor(f32_tensor* tensor);
f32_tensor* rand_init_f32_tensor(f32_tensor* tensor);

i32_tensor* zeros_init_i32_tensor(i32_tensor* tensor);
i32_tensor* ones_init_i32_tensor(i32_tensor* tensor);
i32_tensor* rand_init_i32_tensor(i32_tensor* tensor);

void free_tensor(void* tensor);

// ============================================================================
// INDEXING
// ============================================================================

static size_t linear_index(const tensor_shape_t* shape, size_t index_dimensions, size_t index[]) {
	assert(shape->dimensions == index_dimensions);

	size_t offset = shape->dimensions - index_dimensions;
	size_t linear_index = 0;

	for (size_t i = 0; i < index_dimensions; i++) {
		linear_index += index[i] * shape->arr[i + offset].stride;
	}

	return linear_index;
}

#define idx(x, ...) *(x->storage + linear_index(&x->shape, Index(__VA_ARGS__)))
