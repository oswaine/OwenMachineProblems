// Owen Swaine - 20386155
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TOL 1e-3f   // tolerance for CPU vs GPU result check

// RELATES TO PART B
// single-thread GPU version used for comparison
__global__ void matMulSingleThread(float* P, float* M, float* N, int width)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        for (int row = 0; row < width; row++)
        {
            for (int col = 0; col < width; col++)
            {
                float sum = 0.0f;
                for (int k = 0; k < width; k++)
                {
                    sum += M[row * width + k] * N[k * width + col];
                }
                P[row * width + col] = sum;
            }
        }
    }
}

// RELATES TO PART B AND PART C
// parallel kernel where each thread computes one output element
__global__ void matMulKernel(float* P, float* M, float* N, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // make sure thread is inside matrix bounds
    if (row < width && col < width)
    {
        float sum = 0.0f;
        for (int k = 0; k < width; k++)
        {
            sum += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = sum;
    }
}

// RELATES TO PART B AND PART C
// fills input matrices with random values
void fillRandom(float* A, int width)
{
    for (int i = 0; i < width * width; i++)
    {
        A[i] = (float)rand() / RAND_MAX;
    }
}

// RELATES TO PART B AND PART C
// CPU version used for timing and correctness check
void matMulCPU(float* P, float* M, float* N, int width)
{
    for (int row = 0; row < width; row++)
    {
        for (int col = 0; col < width; col++)
        {
            float sum = 0.0f;
            for (int k = 0; k < width; k++)
            {
                sum += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = sum;
        }
    }
}

// RELATES TO PART B AND PART C
// checks if GPU output matches CPU output
int compareResults(float* cpu, float* gpu, int width)
{
    for (int i = 0; i < width * width; i++)
    {
        if (fabs(cpu[i] - gpu[i]) > TOL)
        {
            printf("Mismatch at index %d CPU=%f GPU=%f\n", i, cpu[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

// RELATES TO PART A
// measures host to device transfer time for M and N
float getHtoDTime(float* d_M, float* d_N, float* h_M, float* h_N, int bytes)
{
    cudaEvent_t start, stop;
    float time_ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_ms;
}

// RELATES TO PART A
// measures device to host transfer time for P
float getDtoHTime(float* h_P, float* d_P, int bytes)
{
    cudaEvent_t start, stop;
    float time_ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_ms;
}

// RELATES TO PART B
// times the 1 block 1 thread GPU case
float getKernelTimeSingleThread(float* d_P, float* d_M, float* d_N, int width)
{
    cudaEvent_t start, stop;
    float time_ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    matMulSingleThread<<<1, 1>>>(d_P, d_M, d_N, width);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_ms;
}

// RELATES TO PART B AND PART C
// times the parallel kernel for a chosen block width
float getKernelTime2D(float* d_P, float* d_M, float* d_N, int width, int blockWidth)
{
    cudaEvent_t start, stop;
    float time_ms;

    // define a 2D block of threads (blockWidth × blockWidth threads per block)
    dim3 threads(blockWidth, blockWidth);
    // compute how many blocks are needed in each dimension to cover the matrix
    // using ceiling division so all elements are handled even if width is not divisible by blockWidth
    dim3 blocks((width + blockWidth - 1) / blockWidth,
                (width + blockWidth - 1) / blockWidth);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    matMulKernel<<<blocks, threads>>>(d_P, d_M, d_N, width);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time_ms;
}

// RELATES TO PART B
// measures CPU matrix multiplication time
double getCPUTime(float* h_P, float* h_M, float* h_N, int width)
{
    clock_t start, stop;
    start = clock();
    matMulCPU(h_P, h_M, h_N, width);
    stop = clock();

    return 1000.0 * (double)(stop - start) / CLOCKS_PER_SEC;
}

// Exectutes ALL OF PART A, to C
// runs one full test case for one matrix size
void runCase(int width)
{
    int elements = width * width;
    int bytes = elements * sizeof(float);

    float* h_M;
    float* h_N;
    float* h_P_cpu;
    float* h_P_gpu;

    float* d_M;
    float* d_N;
    float* d_P;

    // allocate host memory
    h_M = (float*)malloc(bytes);
    h_N = (float*)malloc(bytes);
    h_P_cpu = (float*)malloc(bytes);
    h_P_gpu = (float*)malloc(bytes);

    // fill input matrices with random values
    fillRandom(h_M, width);
    fillRandom(h_N, width);

    // allocate device memory
    cudaMalloc((void**)&d_M, bytes);
    cudaMalloc((void**)&d_N, bytes);
    cudaMalloc((void**)&d_P, bytes);

    printf("\n==============================\n");
    printf("Matrix size: %d x %d\n", width, width);
    printf("==============================\n");

    // PART A: host to device transfer time
    float h2d = getHtoDTime(d_M, d_N, h_M, h_N, bytes);
    printf("H->D time for M and N: %f ms\n", h2d);

    // PART A: device to host transfer time
    cudaMemset(d_P, 0, bytes);
    float d2h = getDtoHTime(h_P_gpu, d_P, bytes);
    printf("D->H time for P     : %f ms\n", d2h);

    // PART B: CPU timing
    double cpu_time = getCPUTime(h_P_cpu, h_M, h_N, width);
    printf("CPU time            : %f ms\n", cpu_time);

    // have to copy actual input matrices to device before kernel tests
    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

    // PART B: only needed for 300x300 and 750x750
    if (width == 300 || width == 750)
    {
        // 1 block 1 thread case
        cudaMemset(d_P, 0, bytes);
        float gpu1 = getKernelTimeSingleThread(d_P, d_M, d_N, width);
        cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost);

        printf("GPU time 1 block 1 thread: %f ms\n", gpu1);
        if (compareResults(h_P_cpu, h_P_gpu, width))
            printf("Test PASSED\n");
        else
            printf("Test FAILED\n");

        // block width 1 case
        cudaMemset(d_P, 0, bytes);
        float gpu_bw1 = getKernelTime2D(d_P, d_M, d_N, width, 1);
        cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost);

        printf("GPU time block width 1   : %f ms\n", gpu_bw1);
        if (compareResults(h_P_cpu, h_P_gpu, width))
            printf("Test PASSED\n");
        else
            printf("Test FAILED\n");

        // approximate full offload time
        printf("Approx total GPU offload time: %f ms\n", h2d + gpu_bw1 + d2h);
    }

    // PART C: block width experiments
    int blockWidths[5] = {2, 5, 10, 15, 25};

    for (int i = 0; i < 5; i++)
    {
        int bw = blockWidths[i];

        cudaMemset(d_P, 0, bytes);
        float gpu_time = getKernelTime2D(d_P, d_M, d_N, width, bw);
        cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost);

        printf("GPU time block width %2d: %f ms   ", bw, gpu_time);

        if (compareResults(h_P_cpu, h_P_gpu, width))
            printf("[PASSED]\n");
        else
            printf("[FAILED]\n");
    }

    // free host memory
    free(h_M);
    free(h_N);
    free(h_P_cpu);
    free(h_P_gpu);

    // free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main()
{
    srand((unsigned int)time(NULL));   // seed random numbers

    int sizes[5] = {300, 750, 1500, 3000, 4500};

    // run all required matrix sizes
    for (int i = 0; i < 5; i++)
    {
        runCase(sizes[i]);
    }

    cudaDeviceReset();
    return 0;
}