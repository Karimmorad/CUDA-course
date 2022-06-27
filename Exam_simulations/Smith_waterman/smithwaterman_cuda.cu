#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define S_LEN 512
#define N 1000

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


double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__device__ __host__ int max(int n1, int n2, int n3, int n4) {
    int tmp1, tmp2;

    tmp1 = n1 > n2 ? n1 : n2;
    tmp2 = n3 > n4 ? n3 : n4;
    tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;

    return tmp1;
}

__global__ void smithwaterman(char *query_d, char *string_d, int *mat_d, int *res_d, int ins, int del){

    char *query = &query_d[S_LEN*blockIdx.x];
    char *string = &string_d[S_LEN*blockIdx.x];
    int *mat = &mat_d[(S_LEN+1)*(S_LEN+1)*blockIdx.x];

    for(int i = 0; i < S_LEN+1; i++){
        for(int j = 0; j < S_LEN+1; j+=blockDim.x){//parallel initialization of the matrix
            if(j+threadIdx.x < S_LEN+1)
                mat[i*(S_LEN+1)+j+threadIdx.x]=0;
        }
    }
    __syncthreads();
    
    int A_LEN = 2;                                      //1 over the real antidiag length (it has to start from the first cell)
    int MIN_COL = 0;

    for (int i=1; i<S_LEN+S_LEN+1; i++){                //compute antidiags per matrix
        for(int j=1; j<A_LEN; j+=blockDim.x){           //compute the inner for in parallel
            if(j+threadIdx.x<A_LEN){
                int row, col;
                col = MIN_COL + j + threadIdx.x;
                row = i - col + 1;
                mat[row*(S_LEN+1)+col]= max(mat[(row-1)*(S_LEN+1)+col-1]+(query[row-1]==string[col-1]), mat[(row-1)*(S_LEN+1)+col]+del, mat[row*(S_LEN+1)+col-1]+ins, 0);
            }
        }
        __syncthreads();
        if(i < S_LEN)
            A_LEN++;                                    //increase antidiag len
        else{
            A_LEN--;                                    //decrease antidiag len
            MIN_COL++;
        }
    }
    __syncthreads();
    if(threadIdx.x == 0)
        res_d[blockIdx.x] = mat[S_LEN*(S_LEN+1)+S_LEN];
    __syncthreads();
}

int main(int argc, char *argv[]){

    if(argc < 2){
        printf("./exec BLOCKSIZE");
        return 0;
    }

    int numThreads = atoi(argv[1]);

    srand(time(NULL));
    int ins=-2, del=-2;

    char query[N][S_LEN];
    char string[N][S_LEN];

    // we need to flatten the sequences for the gpu

    char query_flat[N*S_LEN];
    char string_flat[N*S_LEN];

    //
    int res[N];

    char alphabet[5] = {'A','C','G','T','N'};
    int mat[S_LEN+1][S_LEN+1];
    for(int n = 0; n < N; n++){
        for(int i = 0; i < S_LEN; i++){
            query[n][i]=alphabet[rand()%5];
            string[n][i]=alphabet[rand()%5];
            query_flat[n*S_LEN+i]=query[n][i];
            string_flat[n*S_LEN+i]=string[n][i];
        } 
    }
    
    double start_cpu = get_time();
    for(int n = 0; n < N; n++){
        for(int i = 0; i < S_LEN+1; i++){
            for(int j = 0; j < S_LEN+1; j++){
                mat[i][j]=0;
            }
        }

        int A_LEN = 2;                                      //1 over the real antidiag length (it has to start from the first cell)
        int MIN_COL = 0;

        for (int i=1; i<S_LEN+S_LEN+1; i++){                //compute antidiags per matrix
            for(int j=1; j<A_LEN; j++){
                int row, col;
                
                col = MIN_COL + j;
                row = i - col + 1;
                mat[row][col]= max(mat[row-1][col-1]+(query[n][row-1]==string[n][col-1]), mat[row-1][col]+del, mat[row][col-1]+ins, 0);
            }
            if(i < S_LEN)
                A_LEN++;                                    //increase antidiag len
            else{
                A_LEN--;                                    //decrease antidiag len
                MIN_COL++;
            }
        }
        res[n] = mat[S_LEN][S_LEN];
    }
    double end_cpu = get_time();

    printf("CPU Time: %lf\n", end_cpu-start_cpu);


    /// GPU

    char *query_d, *string_d;
    int *mat_d, *res_d;

    int res_gpu[N];

    CHECK(cudaMalloc(&query_d, sizeof(char)*N*S_LEN));
    CHECK(cudaMalloc(&string_d, sizeof(char)*N*S_LEN));
    CHECK(cudaMalloc(&mat_d, sizeof(int)*N*(S_LEN+1)*(S_LEN+1)));
    CHECK(cudaMalloc(&res_d, sizeof(int)*N));

    CHECK(cudaMemcpy(query_d, query_flat, sizeof(char)*N*S_LEN,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(string_d, string_flat, sizeof(char)*N*S_LEN,cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(N,1,1);
    dim3 threadsPerBlock(numThreads,1,1);
    double start_gpu = get_time();
    smithwaterman<<<blocksPerGrid,threadsPerBlock>>>(query_d,string_d,mat_d,res_d,ins,del);
    CHECK_KERNELCALL();

    CHECK(cudaMemcpy(res_gpu, res_d, sizeof(int)*N,cudaMemcpyDeviceToHost));
    double end_gpu = get_time();

    int test = 1;
    for(int n = 0; n < N; n++){
        if(res[n]!=res_gpu[n]){
            test = 0;
            break;
        }
    }

    if(test)
        printf("GPU Time: %lf\n", end_gpu-start_gpu);

    CHECK(cudaFree(query_d));
    CHECK(cudaFree(string_d));
    CHECK(cudaFree(mat_d));
    CHECK(cudaFree(res_d));

    return 0;

}