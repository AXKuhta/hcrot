#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "tensor_t.h"
#include "simd.h"

static size_t tensor_element_count(const tensor_t* tensor) {
	size_t elements = 1;

	for (int i = 0; i < tensor->shape_dimensions; i++)
		elements *= tensor->shape[i].size;

	return elements;
}

static size_t datatype_size(const char* datatype) {
	if (strcmp(datatype, "f32") == 0) { return 4; }
	else if (strcmp(datatype, "i32") == 0) { return 4; }
	else {
		printf("Unknown datatype: [%s]\n", datatype);
		exit(-1);
		return 0;
	}
}

static size_t apply_padding(size_t size) {
	return size + size % (16 * 4);
}

static size_t tensor_storage_size(const tensor_t* tensor) {
	size_t element_size = datatype_size(tensor->storage.datatype);
	size_t elements = tensor_element_count(tensor);

	return apply_padding(elements * element_size);
}

static void default_stride(tensor_t* tensor) {
	size_t stride = 1;

	for (int i = tensor->shape_dimensions - 1; i >= 0; i--) {
		tensor->shape[i].stride = stride;
		stride *= tensor->shape[i].size;
	}
}

static tensor_t* alloc_tensor(int shape_dimensions, size_t shape[], char* datatype) {
	tensor_t* tensor = malloc(sizeof(tensor_t) + sizeof(shape_dimension_t)*shape_dimensions);
	tensor->shape_dimensions = shape_dimensions;
	
	for (int i = 0; i < shape_dimensions; i++) {
		tensor->shape[i] = (shape_dimension_t){ .size = shape[i], .stride = 0 };
		assert(shape[i] > 0);
	}

	default_stride(tensor);

	tensor->storage.datatype = datatype;
	tensor->storage.size = tensor_storage_size(tensor);
	tensor->storage.memory = malloc(tensor->storage.size);

	return tensor;
}

tensor_t* zeros_tensor(int shape_dimensions, size_t shape[], char* datatype) {
	tensor_t* tensor = alloc_tensor(shape_dimensions, shape, datatype);

	memset(tensor->storage.memory, 0, tensor->storage.size);

	return tensor;
}

tensor_t* ones_tensor(int shape_dimensions, size_t shape[], char* datatype) {
	tensor_t* tensor = alloc_tensor(shape_dimensions, shape, datatype);
	size_t elements = tensor_element_count(tensor);

	if (strcmp(datatype, "f32") == 0) {
		f32* data = (f32*)tensor->storage.memory;

		for (size_t i = 0; i < elements; i++)
			data[i] = 1.0;

	} else if (strcmp(datatype, "i32") == 0) {
		i32* data = (i32*)tensor->storage.memory;

		for (size_t i = 0; i < elements; i++)
			data[i] = 1;
	}

	return tensor;
}

tensor_t* rand_tensor(int shape_dimensions, size_t shape[], char* datatype) {
	tensor_t* tensor = alloc_tensor(shape_dimensions, shape, datatype);
	size_t elements = tensor_element_count(tensor);

	if (strcmp(datatype, "f32") == 0) {
		f32* data = (f32*)tensor->storage.memory;

		for (size_t i = 0; i < elements; i++)
			data[i] = (double)rand() / (double)RAND_MAX;

	} else if (strcmp(datatype, "i32") == 0) {
		i32* data = (i32*)tensor->storage.memory;

		for (size_t i = 0; i < elements; i++)
			data[i] = rand();
	}

	return tensor;
}

void free_tensor(tensor_t* tensor) {
	free(tensor->storage.memory);
	free(tensor);
}
