#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 2048

#define CHECK(call) \
{ \
  const cudaError_t err = call; \
  if (err != cudaSuccess) { \
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} \


#define CHECK_KERNEL() \
{ \
  const cudaError_t err = cudaGetLastError();\
  if (err != cudaSuccess) {\
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);\
  }\
}\

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void reduction_serial(float* d_input, float* d_sum, int length){
    d_sum[0] = d_input[0];
    int i;
    for(i = 1; i < length; i++)
    {
        d_sum[i] = d_sum[i-1] + d_input[i];
    }
}

void sum_cpu(float *h_input, float* h_sum_cpu, int length){
    h_sum_cpu[0] = h_input[0];
    int i;
    for (i = 1 ; i < length; i++)
    {
        h_sum_cpu[i] += h_sum_cpu[i-1] + h_input[i];
    }
}

int main(int argc, char* argv[]){
    
    double start_cpu, end_cpu, start_gpu, end_gpu;
    srand(time(NULL));
    int length = N;
    
    float *d_input, *d_sum;

    float *h_input = (float*)malloc(sizeof(float)*length);
    float *h_sum_gpu = (float*)malloc(sizeof(float)*length);
    float *h_sum_cpu = (float*)malloc(sizeof(float)*length);

    dim3 blocksPerGrid(N/N, 1, 1);
    dim3 threadsPerBlock(1,1,1);

    //allocation of random numbers between 0-99
    for (int i = 0; i < length; i++)
        h_input[i] = rand()%100;

    start_cpu = get_time();
    sum_cpu(h_input, h_sum_cpu, length);
    end_cpu = get_time();


    //Allocating in gpu
    CHECK(cudaMalloc(&d_input, length*sizeof(float)));
    CHECK(cudaMalloc(&d_sum, length*sizeof(float)));

    //moving data to gpu
    CHECK(cudaMemcpy(d_input, h_input, length*sizeof(float),
                        cudaMemcpyHostToDevice));
   
    //gpu kernel execution
    start_gpu = get_time();
    reduction_serial<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_sum, length);
    CHECK_KERNEL()
    CHECK(cudaDeviceSynchronize());
    end_gpu = get_time();

    //moving data from gpu to cpu
    CHECK(cudaMemcpy(h_sum_gpu, d_sum, length*sizeof(float),
                        cudaMemcpyDeviceToHost));


    if(h_sum_cpu[length-1] != h_sum_gpu[length-1])
    {
        printf("Wrong result on GPU: %f, CPU: %f \n", h_sum_gpu[length-1], h_sum_cpu[length-1]);
        return 1;
    }
    printf("all results are correct\n");

    double gpu_time = 0, cpu_time = 0;

    cpu_time = end_cpu - start_cpu;
    gpu_time = end_gpu - start_gpu;

    printf("GPU time: %lf\n", gpu_time);
    printf("CPU_time: %lf\n", cpu_time);

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_sum));
    return 0;
}