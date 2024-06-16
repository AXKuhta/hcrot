
#define zeros_tensor(datatype, ...) zeros_init_##datatype##_tensor(alloc_storage(sizeof(datatype), alloc_tensor(#datatype, Shape(__VA_ARGS__))))
#define ones_tensor(datatype, ...) ones_init_##datatype##_tensor(alloc_storage(sizeof(datatype), alloc_tensor(#datatype, Shape(__VA_ARGS__))))
#define rand_tensor(datatype, ...) rand_init_##datatype##_tensor(alloc_storage(sizeof(datatype), alloc_tensor(#datatype, Shape(__VA_ARGS__))))
#define _array_tensor(datatype, shape_size, shape, array_size, array) array_init_##datatype##_tensor(alloc_storage(sizeof(datatype), alloc_tensor(#datatype, shape_size, shape)), array_size, array)
#define array_tensor(...) _array_tensor(__VA_ARGS__)

tensor_t* zeros_init_f32_tensor(tensor_t* tensor);
tensor_t* ones_init_f32_tensor(tensor_t* tensor);
tensor_t* rand_init_f32_tensor(tensor_t* tensor);
tensor_t* array_init_f32_tensor(tensor_t* tensor, size_t array_size, f32 array[]);

tensor_t* zeros_init_i32_tensor(tensor_t* tensor);
tensor_t* ones_init_i32_tensor(tensor_t* tensor);
tensor_t* rand_init_i32_tensor(tensor_t* tensor);
tensor_t* array_init_i32_tensor(tensor_t* tensor, size_t array_size, i32 array[]);

tensor_t* zeros_init_c64_tensor(tensor_t* tensor);
tensor_t* ones_init_c64_tensor(tensor_t* tensor);
tensor_t* rand_init_c64_tensor(tensor_t* tensor);
tensor_t* array_init_c64_tensor(tensor_t* tensor, size_t array_size, c64 array[]);
