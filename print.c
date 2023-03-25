#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "tensor_t.h"
#include "print.h"
#include "simd.h"

#define H_ELEMENT_MAX 6

static void print_f32(const tensor_t* tensor) {
	const size_t shape_dimensions = tensor->shape_dimensions;

	printf("tensor(Shape(");

	for (size_t i = 0; i < shape_dimensions; i++)
		printf("%zu%s", tensor->shape[i].size, shape_dimensions - i > 1 ? ", " : "");

	printf("), \"%s\")\n", tensor->storage.datatype);

	size_t* counters_reload = malloc(shape_dimensions*sizeof(size_t));
	size_t* counters = malloc(shape_dimensions*sizeof(size_t));
	size_t elements = 1;

	for (int i = shape_dimensions - 1; i >= 0; i--) {
		elements *= tensor->shape[i].size;
		counters_reload[i] = elements;
		counters[i] = 0;
	}

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
		size_t braces_open = 0;
		size_t braces_close = 0;

		for (size_t j = 0; j < shape_dimensions; j++) {
			braces_open += counters[j] == 0;
			braces_close += 1+counters[j] == counters_reload[j];
			counters[j] = 1+counters[j] == counters_reload[j] ? 0 : counters[j] + 1;
		}

		h_elements++;

		if (h_elements >= H_ELEMENT_MAX) {
			if (h_elements == H_ELEMENT_MAX)
				printf("..., ");

			if (braces_close == 0)
				continue;
		}

		const char* pad = 1 == h_elements ? level_pad + braces_open : "";
		const char* prefix = level_enter + shape_dimensions - braces_open;
		const char* separator = elements - i > 1 ? ", " : "";
		const char* postfix = level_leave + shape_dimensions - braces_close;

		printf("%s%s%.4f%s%s", pad, prefix, data[i], postfix, separator);

		if (braces_close > 0) {
			h_elements = 0;
			printf("\n");
		}
	}

	free(level_enter);
	free(level_leave);
	free(level_pad);
	free(counters);
	free(counters_reload);
}

void print_tensor(const tensor_t* tensor) {
	print_f32(tensor);
}
