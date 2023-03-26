
typedef float f32;
typedef int i32;

void f32_add_f32(f32* restrict a, f32* restrict b, const size_t size);
f32 dot_f32_f32(f32* restrict a, f32* restrict b, const size_t size);

#define K_STRIDE 128
