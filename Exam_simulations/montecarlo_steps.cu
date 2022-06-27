#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#define N (1<<30)

__global__ void init(unsigned int seed, curand_t *states){
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

//this code is using the cuda random library
// using CudaStates to randomize numbers from a uniform distribution
// then finding using the numbers to calculate if they full inside the area of a circle
// if the condition is correct we will use atomic add to calculate the space over all points
// after that we need to send back the value
// calculate the number found over the total numbers sent
__global__ void compute_pi(cudaState_t *states, unsigned *pi){
    double x = curand_uniform(&state[blockIdx.x]);
    double y = curand_uniform(&state[blockIdx.x]);
    unsigned toAdd = (x*x + y*y <= 1);
    atomicAdd(pi, toAdd);
}

// warp reduction is a way to compute the last 32 threads with unrolling
// this way will achieve extra improvement by eliminating warp divergence
__device warpReduce(volatile unsigned *input, size_t threadId){
    input[threadId] += input[threadId+32];
    input[threadId] += input[threadId+16];
    input[threadId] += input[threadId+8];
    input[threadId] += input[threadId+4];
    input[threadId] += input[threadId+2];
    input[threadId] += input[threadId+1];
}
// this function will calculate the parallel reduction to the algorithm
// it will decrease the warp divergence and will decrease the amount of hits we use the global memory
// by utilizing the shared memory in adding the threads of each block
// then accessing the global memory just once to add the whole accumulation
__device__ unsigned accumulate(unsigned *input, size_t dim){
    size_t threadId = threadIdx.x;
    if(dim > 32){
        for(size_t i = dim/2; i > 32; i>>=1) {
            if (threadId < i)
                input[threadId] += input[threadId + i];
            __syncthreads();
        }
    }
    if(threadId < 32){
        warpReduce(input, threadId);
    }
    __syncthreads();
    return input[0]

}
// this is the second function, in tackling a parallel reduction technique and optimizing the CUDA program
// we gain more marks for using them.
__global__ void compute_pi_2(cudaState_t *states, unsigned *pi){
    extern __shared__ unsigned tmp[];
    double x = curand_uniform(&state[blockIdx.x]);
    double y = curand_uniform(&state[blockIdx.x]);
    unsigned tempval;
    tmp[threadIdx.x] = (x*x + y*y <= 1);
    __syncthreads();
    //parallel reduction
    tempval = accumulate(tmp, blockDim.x);
    if(threadId.x == 0)
        atomicAdd(pi, tempval);
}

int main(int argc, char *argv[]){

    int numThreads;
    if(argc < 2)
        return 0;

    numThreads = atoi(argv[1]);

    dim3 blocksPerGrid(N/numThreads,1,1);
    dim3 threadsPerBlocks(numThreads,1,1);

    unsigned int h_pi;
    unsigned int *d_pi;

    curandState_t *states;
    cudaMalloc(&states, N/numThreads*sizeof(curandState_t));
    cudaMalloc(&di_pi, sizeof(unsigned));

    init<<<blocksPerGrid,1>>>(time(NUll), states);
    // call for the simplest, passable grade
//    compute_pi<<<blocksPerGrid, threadsPerBlocks>>>(states, d_pi);

    compute_pi_2<<<blocksPerGrid, threadsPerBlocks, sizeof(unsigned)*numThreads>>>(states, d_pi);
    cudaMemcpy(&h_pi, d_pi, sizeof(unsigned), cudaMemcpyDeviceToHost);

    double pi = (double)h_pi/N*4;

    return 0;
}