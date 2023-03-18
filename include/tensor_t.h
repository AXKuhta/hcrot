
typedef struct tensor_storage_t {
	char* datatype;
	void* memory;
	size_t size;
} tensor_storage_t;

typedef struct tensor_t {
	tensor_storage_t storage;
	int dimensions;
	int shape[];
} tensor_t;

// Shape(4, 16) => 2, (int[]){4, 16}
#define Shape(...) (sizeof((int[]){__VA_ARGS__}) / sizeof(int)), ((int[]){__VA_ARGS__})

tensor_t* init_tensor(int dimensions, int shape[], char* datatype);
void debug_tensor(tensor_t* tensor);
