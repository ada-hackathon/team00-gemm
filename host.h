//Data Type
#define TYPE double

//Algorithm Parameters
#define row_size 16
#define col_size 16
#define N row_size*col_size
#define block_size 8
#define NUMOFBLOCKS N/block_size/block_size

//Define the input range to operate over
#define MIN 0.
#define MAX 1.0

//Set number of iterations to execute
#define MAX_ITERATION 1

//void bbgemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  TYPE m1[N];
  TYPE m2[N];
  TYPE prod[N];
};
