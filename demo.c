#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <stdio.h>

#include "api.h"

void tensor_initialization() {
	tensor_t* a = zeros_tensor(f32, 4, 4);
	tensor_t* b = ones_tensor(f32, 4, 4);
	tensor_t* c = rand_tensor(f32, 4, 4);

	printf("===============================================\n");
	printf("Zeros/ones/random initialization\n");
	printf("===============================================\n");

	printf("Zeros tensor:\n");
	print_tensor(a);

	printf("\nOnes tensor:\n");
	print_tensor(b);

	printf("\nRand tensor:\n");
	print_tensor(c);

	printf("\n\n");

	free_tensor(a);
	free_tensor(b);
	free_tensor(c);
}

void inplace_math() {
	tensor_t* a = rand_tensor(f32, 4, 4);
	tensor_t* b = rand_tensor(f32, 4, 4);
	tensor_t* c = ones_tensor(f32, 4, 4);

	printf("===============================================\n");
	printf("Inplace math\n");
	printf("===============================================\n");

	printf("Inplace add:\n");
	print_tensor(a);
	printf("+\n");
	print_tensor(c);
	tensor_f32_add_f32(a, c);
	printf("=\n");
	print_tensor(a);
	printf("\n");

	printf("Inplace sub:\n");
	print_tensor(a);
	printf("-\n");
	print_tensor(b);
	tensor_f32_sub_f32(a, b);
	printf("=\n");
	print_tensor(a);
	printf("\n");

	printf("Inplace mul:\n");
	print_tensor(a);
	printf("*\n");
	print_tensor(b);
	tensor_f32_mul_f32(a, b);
	printf("=\n");
	print_tensor(a);
	printf("\n");

	printf("Inplace div:\n");
	print_tensor(c);
	printf("/\n");
	print_tensor(a);
	tensor_f32_div_f32(c, a);
	printf("=\n");
	print_tensor(c);
	printf("\n");
}

int main() {
	tensor_initialization();
	inplace_math();
}
