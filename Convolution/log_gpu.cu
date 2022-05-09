//
// Created by Karim Morad on 5/9/22.
//
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

double get_time(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec*1e-6;
}

void load_image(char *fname, int Nx, int Ny, float *img){
    FILE *fp;
    fp = fopen(fname, "r");
    size_t i = 0,j = 0;
    for(i = 0; i < Ny; ++i){
        for (j = 0; j < Nx; ++j){
            fscanf(fp, "%f", &img[i*Nx + j]);
        }
        fscanf(fp, "\n");
    }
    fclose(fp);
}

void save_image(char *fname, int Nx, int Ny, float *img){
    FILE *fp;
    fp = fopen(fname, "w");
    size_t i,j;
    for(i = 0; i < Ny; ++i){
        for (j = 0; j < Nx; ++j){
            // we want to write 10 elements before the floating point and
            // 3 elements afterwards
            fprintf(fp, "%10.3f", &img[i*Nx + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}
// one size, since it's a square matrix kernel
void calculate_kernel(int kernel_size, float sigma, float *kernel){
    int n_sq = kernel_size*kernel_size;
    float x, y, center;

    center = (kernel_size - 1) / 2.0;
    int i;
    for (i = 0; i < n_sq; ++i){
        x = (float)(i % kernel_size) - center;
        y = (float)(i / kernel_size) - center;
        kernel[i] = -(1.0 / M_PI * pow(sigma, 4)) *
                    (1.0 - 0.5 * (x*x + y*y) /(sigma*sigma)) *
                    exp(-0.5 * (x*x + y*y)/ (sigma*sigma));
    }
}

__global__ conv_img_gpu(float *img, float *kernel, float *imgf, int Nx, int Ny, int kernel_size){
    int tid = threadIdx.x
    int iy = blockIdx.x + (kernel_size-1)/2;
    int ix = threadIdx.x + (kernel_size-1)/2;
    int idx = iy * Nx + ix
    int s_k = kernel_size*kernel_size;
    int center = (kernel_size - 1)/2
    int ii, jj;
    float sum = 0;
    extern __shared__ float sdata[];
    if(tid < s_k){
        sdata[tid] = kernel[tid];
    }
    __syncthreads();
    if(idx < NX * NY){
        for (int ki = 0; ki < kernel_size; ki++)
            for (int kj = 0; kj < kernel_size; kj++){
                ii = kj + ix - center;
                jj = ki + iy - center;
                sum += img[jj*Nx+ii]*sdata[ki*kernel_size+kj];
            }
        imgf[idx] = sum;
    }
    __syncthreads();
}

void conv_img_cpu(float* img, float *kernel, float* imgf, int Nx, int Ny, int kernel_size) {
    float sum = 0;
    int center = (kernel_size - 1) / 2;
    int ii, jj, i, j, ki, kj;

    for (i = center; i < Nx - center; ++i) {
        for (j = center; j < Ny - center; ++j) {
            sum = 0;
            for (ki = 0; ki < kernel_size; ++ki) {
                for (kj = 0; kj < kernel_size; ++kj) {
                    ii = kj + j - center;
                    jj = ki + i - center;
                    sum += img[jj * Nx + ii] * kernel[ki * kernel_size + kj];
                }
            }
            imgf[i * Nx + j] = sum;
        }
    }
}

int main(int argc, __attribute__((unused)) char* argv[]){

    int Ny, Nx, kernel_size;
    char finput[256], foutput[256];
    float sigma;
    double start_cpu, end_cpu, start_gpu, end_gpu;

    sprintf(finput, "lux_bw.dat");
    sprintf(foutput, "lux_output.dat");

    //Image dims
    Ny = 20;
    Nx = 20;
    kernel_size = 32;
    sigma = 0.8f;

    float *img, *imgf, *kernel;

    img = (float *) malloc(Nx*Ny*sizeof(float));
    imgf = (float *) malloc(Nx*Ny*sizeof(float));
    imgf_hw = (float *) malloc(Nx*Ny*sizeof(float));
    kernel = (float *) malloc(kernel_size*kernel_size*sizeof(float));
    calculate_kernel(kernel_size, sigma, kernel);
    load_image(finput, Nx, Ny, img);

    start_cpu = get_time();
    conv_img_cpu(img, kernel, imgf, Nx, Ny, kernel_size);
    end_cpu = get_time();

    //save_image(foutput, Nx, Ny, imgf);

    printf("CPU time: %lf\n", end_cpu - start_cpu);

    float *d_img, *d_img_res, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_img, Nx*Ny*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img_res, Nx*Ny*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size*kernel_size*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_img, img, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_img_res, imgf, Nx*Ny*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel, kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice));
    start_gpu = get_time();
    conv_img_gpu<<<(Nx - (kernel_size-1)), Ny - (kernel_size-1), kernel_size*kernel_size*sizeof(float)>>>(d_img, d_kernel, d_img_res, Nx, Ny, kernel_size);
    cudaDeviceSynchronize();
    end_gpu = get_time();

    cudaMemcpy(imgf_hw, d_img_res, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
    save_image(foutput, Nx, Ny, imgf_hw);

    free(img);
    free(imgf);
    free(kernel);

    return 0;
}