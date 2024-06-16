
typedef float f32;
typedef int i32;
typedef unsigned char u8;

void f32_add_f32(f32* a, f32* b, const size_t count);
void f32_sub_f32(f32* a, f32* b, const size_t count);
void f32_mul_f32(f32* a, f32* b, const size_t count);
void f32_div_f32(f32* a, f32* b, const size_t count);

void f32_add_x(f32* a, f32 x, const size_t count);
void f32_sub_x(f32* a, f32 x, const size_t count);
void f32_mul_x(f32* a, f32 x, const size_t count);
void f32_div_x(f32* a, f32 x, const size_t count);

f32 f32_min(f32* x, const size_t count);
f32 f32_max(f32* x, const size_t count);
f32 f32_sum(f32* x, const size_t count);

f32 dot_f32_f32(f32* restrict a, f32* restrict b, const size_t count);
