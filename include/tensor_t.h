
typedef struct tensor_storage_t {
	char* datatype;
	void* memory;
	size_t size;
} tensor_storage_t;

typedef struct shape_dimension_t {
	size_t size;
	size_t stride;
} shape_dimension_t;

typedef struct tensor_t {
	tensor_storage_t storage;
	int shape_dimensions;
	shape_dimension_t shape[];
} tensor_t;

// Shape(4, 16) => 2, (size_t[]){4, 16}
#define Shape(...) (sizeof((size_t[]){__VA_ARGS__}) / sizeof(size_t)), ((size_t[]){__VA_ARGS__})
#define Index(...) Shape(__VA_ARGS__)

tensor_t* init_tensor(int shape_dimensions, size_t shape[], char* datatype);
void free_tensor(tensor_t* tensor);
void debug_tensor(tensor_t* tensor);
