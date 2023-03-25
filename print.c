#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "tensor_t.h"
#include "print.h"
#include "simd.h"

#define H_ELEMENT_MAX 6

static size_t braces(size_t index, const tensor_t* tensor) {
	size_t stride = 1;
	size_t count = 0;

	for (int i = tensor->shape_dimensions - 1; i >= 0; i--) {
		stride *= tensor->shape[i].size;
		count += 0 == index % stride;
	}

	return count;
}

static void print_f32(const tensor_t* tensor) {
	const size_t shape_dimensions = tensor->shape_dimensions;

	printf("tensor(Shape(");

	for (size_t i = 0; i < shape_dimensions; i++)
		printf("%zu%s", tensor->shape[i].size, shape_dimensions - i > 1 ? ", " : "");

	printf("), \"%s\")\n", tensor->storage.datatype);

	size_t elements = 1;

	for (int i = shape_dimensions - 1; i >= 0; i--)
		elements *= tensor->shape[i].size;

	char* level_enter = malloc(shape_dimensions + 1);
	char* level_leave = malloc(shape_dimensions + 1);
	char* level_pad = malloc(shape_dimensions + 1);

	memset(level_enter, '[', shape_dimensions);
	memset(level_leave, ']', shape_dimensions);
	memset(level_pad, ' ', shape_dimensions);

	level_enter[shape_dimensions] = 0;
	level_leave[shape_dimensions] = 0;
	level_pad[shape_dimensions] = 0;

	f32* data = (f32*)tensor->storage.memory;

	size_t h_elements = 0;

	for (size_t i = 0; i < elements; i++) {
		const size_t braces_open = braces(i, tensor);
		const size_t braces_close = braces(i + 1, tensor);

		const char* pad = 0 == h_elements % H_ELEMENT_MAX ? level_pad + braces_open : "";
		const char* prefix = level_enter + shape_dimensions - braces_open;
		const char* separator = elements - i > 1 ? ", " : "";
		const char* postfix = level_leave + shape_dimensions - braces_close;

		printf("%s%s%.4f%s%s", pad, prefix, data[i], postfix, separator);

		h_elements++;

		if (0 == h_elements % H_ELEMENT_MAX || braces_close > 0) {
			h_elements = 0;
			printf("\n");
		}
	}

	free(level_enter);
	free(level_leave);
	free(level_pad);
}

void print_tensor(const tensor_t* tensor) {
	print_f32(tensor);
}
