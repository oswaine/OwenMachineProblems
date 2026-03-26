// Owen Swaine - 20386155
// ELEC 374 Machine Problem 2 - Part 1
// Tiled matrix multiplication using shared memory

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TOL 1e-3f

// TILED KERNEL based off slides
__global__ void MatrixMulKernel(float* M, float* N, float* P, int width, int tileWidth)
{
    // Allocate shared memory dynamically
    // First half = M tile, second half = N tile
    extern __shared__ float shared[];
    float* Mds = shared;
    float* Nds = shared + tileWidth * tileWidth;

    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column this thread is responsible for
    int Row = blockIdx.y * tileWidth + ty;
    int Col = blockIdx.x * tileWidth + tx;

    float Pvalue = 0.0f; // accumulator for one output element

    // Loop over tiles (ph = phase)
    for (int ph = 0; ph < width / tileWidth; ph++)
    {
        // Each thread loads ONE element of M and N into shared memory
        Mds[ty * tileWidth + tx] = M[Row * width + ph * tileWidth + tx];
        Nds[ty * tileWidth + tx] = N[(ph * tileWidth + ty) * width + Col];

        // Wait until all threads finish loading
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < tileWidth; k++)
        {
            Pvalue += Mds[ty * tileWidth + k] * Nds[k * tileWidth + tx];
        }

        // Ensure all threads finish before loading next tile
        __syncthreads();
    }

    // Write final result to global memory
    P[Row * width + Col] = Pvalue;
}

// UTIL FUNCTIONS
// Fill matrix with random floats
void fillRandom(float* A, int width)
{
    for (int i = 0; i < width * width; i++)
        A[i] = (float)rand() / RAND_MAX;
}

// CPU version for correctness check
void matMulCPU(float* P, float* M, float* N, int width)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < width; k++)
                sum += M[i * width + k] * N[k * width + j];
            P[i * width + j] = sum;
        }
    }
}

// Compare CPU and GPU outputs
int compareResults(float* cpu, float* gpu, int width)
{
    for (int i = 0; i < width * width; i++)
    {
        if (fabs(cpu[i] - gpu[i]) > TOL)
        {
            printf("Mismatch at %d CPU=%f GPU=%f\n", i, cpu[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

// RUN ONE TILE WIDTH
void runTile(int width, int tileWidth)
{
    if (width % tileWidth != 0)
    {
        printf("ERROR TILE_WIDTH=%d (not divisible)\n", tileWidth);
        return;
    }

    int elements = width * width;
    int bytes = elements * sizeof(float);

    // Host memory
    float* h_M = (float*)malloc(bytes);
    float* h_N = (float*)malloc(bytes);
    float* h_P_cpu = (float*)malloc(bytes);
    float* h_P_gpu = (float*)malloc(bytes);

    // Device memory
    float *d_M, *d_N, *d_P;

    fillRandom(h_M, width);
    fillRandom(h_N, width);

    cudaMalloc((void**)&d_M, bytes);
    cudaMalloc((void**)&d_N, bytes);
    cudaMalloc((void**)&d_P, bytes);

    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

    // Define execution configuration
    dim3 threads(tileWidth, tileWidth);
    dim3 blocks(width / tileWidth, width / tileWidth);

    // Shared memory size (M tile + N tile)
    size_t sharedBytes = 2 * tileWidth * tileWidth * sizeof(float);

    printf("\nTILE_WIDTH = %d\n", tileWidth);

    // Timing setup
    cudaEvent_t start, stop;
    float time_ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start, 0);

    MatrixMulKernel<<<blocks, threads, sharedBytes>>>(d_M, d_N, d_P, width, tileWidth);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    // Copy result back
    cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost);

    // Compute CPU result
    matMulCPU(h_P_cpu, h_M, h_N, width);

    printf("Kernel time: %f ms\n", time_ms);

    // Validate correctness
    if (compareResults(h_P_cpu, h_P_gpu, width))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_M);
    free(h_N);
    free(h_P_cpu);
    free(h_P_gpu);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

// MAIN (full trial with all block widths
// Note to me - repeat this multiple times
int main(int argc, char** argv)
{
    srand((unsigned int)time(NULL));

    int width = 1500;

    if (argc >= 2)
        width = atoi(argv[1]);

    printf("\n==============================\n");
    printf("Matrix size: %d x %d\n", width, width);
    printf("==============================\n");

    // Required tile widths from assignment
    int tileWidths[5] = {2, 5, 10, 15, 25};

    for (int i = 0; i < 5; i++)
    {
        runTile(width, tileWidths[i]);
    }

    cudaDeviceReset();
    return 0;
}