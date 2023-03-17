
typedef struct tensor_t {
	char* datatype;
	void* storage;
	int dimensions;
	int shape[];
} tensor_t;

// Shape(4, 16) => 2, (int[]){4, 16}
#define Shape(...) (sizeof((int[]){__VA_ARGS__}) / sizeof(int)), ((int[]){__VA_ARGS__})

size_t tensor_size(const tensor_t* tensor);
tensor_t* init_tensor(int dimensions, int shape[], char* datatype);
void debug_tensor(tensor_t* tensor);
