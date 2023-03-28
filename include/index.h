
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

