#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define MAXVAL 100
#define DIST 10

// helpful checks for cuda functions and kernel
#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

/*
* The kernel function to accelerate receives in input a vector of positive integers, called A,
* together with its size, and a second empty vector of integers, B, of the same size.
* For each element i in A, the function saves in B[i] the value 1 if A[i] is greater than all the
* neighbor values with an index between (i-DIST) and (i+DIST), bounds included and if they exist;
* 0 otherwise. DIST is a constant value defined with a macro.
* The main function is a dummy program that receives as an argument the vector size, instantiates and
* populates randomly A, invokes the above function, and shows results.
*/

void printV(int *V, int num);
void compute(int *V, int *R, int num);

//display a vector of numbers on the screen
void printV(int *V, int num) {
    int i;
    for(i=0; i<num; i++)
        printf("%3d(%d) ", V[i], i);
    printf("\n");
}

//kernel function: identify peaks in the vector
void compute(int *V, int *R, int num) {
    int i, j, ok;
    for(i=0; i<num; i++){
        // each thread have access to DIST
        for(j=-DIST, ok=1; j<=DIST; j++){
            if(i+j>=0 && i+j<num && j!=0 && V[i]<=V[i+j])
                ok=0;
            R[i] = ok;
        }
    }
}

__global__ void compute_gpu(int *V, int *R, int num){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int j;
    int ok=1;
    if (tid < num)
    {
        for (j = -DIST; j <=DIST; j++) {
            if(tid+j >= 0 && tid+j < num && j!=0 && V[tid] < V[tid+j])
                ok = 0;
            R[tid] = ok;
        }
    }
}

__global__ void compute_gpu_shared(int *V, int *R, int num){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int j;
    int ok=1;
//    __shared__ int tmp[32+2*DIST+1];
    __shared__ int tmp_result[32];
//    tmp[DIST+threadIdx.x] = V[tid];
    tmp_result[threadId.x] = 1;
    __syncthreads();
    if (tid < num)
    {
        for (j = -DIST; j <=DIST; j++) {
            ok = 1;
            if(tid+j >= 0 && tid+j < num && j!=0 && V[tid] < V[tid+j])
                ok = 0;
            tmp_result[threadId.x] = ok;
        }
        __syncthreads();
        R[tid] = tmp_result[threadIdx.x];
        __syncthreads();
    }
}

int main(int argc, char **argv) {
    int *A;
    int *B;
    int dim;
    int i;

    //read arguments
    if(argc!=2){
        printf("Please specify sizes of the input vector\n");
        return 0;
    }
    dim=atoi(argv[1]);

    //allocate memory for the three vectors
    A = (int*) malloc(sizeof(int) * dim);
    if(!A){
        printf("Error: malloc failed\n");
        return 1;
    }
    B = (int*) malloc(sizeof(int) * dim);
    if(!B){
        printf("Error: malloc failed\n");
        return 1;
    }
    int *A_d;
    int *B_d;
    int *R_h;
    int *R_sh;

    R_h = (int*) malloc(sizeof(int) * dim);
    R_sh = (int*) malloc(sizeof(int) * dim);

    cudaMalloc(A_d, sizeof(int) * dim);
    cudaMalloc(B_d, sizeof(int) * dim);

    //initialize input vectors
    srand(0);
    for(i=0; i<dim; i++) {
        A[i] = rand() % MAXVAL + 1;

    }
    cudaMemcpy(A_d, A, sizeof(int) * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(int) * dim, cudaMemcpyHostToDevice);
    compute_gpu <<<ceil(dim/32), 32>>>(A_d, B_d, dim);
    cudaMemcpy(R_h, B_d, sizeof(int) * dim, cudaMemcpyDeviceToHost);

    compute_gpu_shared<<<ceil(dim/32), 32>>>(A_d, B_d, dim);
    cudaMemcpy(R_sh, B_d, sizeof(int) * dim, cudaMemcpyDeviceToHost);

    //execute on CPU
    compute(A, B, dim);

    //print results
    printV(A, dim);
    printV(B, dim);

    //print gpu results
    printV(R_h, dim);
    //print gpu shares results
    printV(R_sh, dim);
    free(A);
    free(B);
    free(R_h);
    free(R_sh);
    cudafree(A_d);
    cudafree(B_d);

    return 0;
}

//    // copy c code host part
//    // declare the needed device data
//    // send the data from host to device
//    // call the gpu kernel
//    // send data back from device to host
//    // free the allocated device memory