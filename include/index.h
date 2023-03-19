
#define SET_GET_FN_PROTOTYPE(TYPE) 	void set_##TYPE(const tensor_t* tensor, int index_dimensions, size_t index[], TYPE x); \
									TYPE get_##TYPE(const tensor_t* tensor, int index_dimensions, size_t index[]);

SET_GET_FN_PROTOTYPE(f32)
SET_GET_FN_PROTOTYPE(i32)
