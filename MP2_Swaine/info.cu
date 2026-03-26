// Owen Swaine - 20386155
// ELEC 374 Machine Problem 2
// Resource query for Part 1 tiled matrix multiplication kernel

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            printf("CUDA error at %s:%d -> %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                     \
    } while (0)

// This is the same runtime-tileWidth kernel style as your Part 1 code.
// Keep this matching your actual submission kernel if you want the most accurate numbers.
__global__ void MatrixMulKernel(float* M, float* N, float* P, int width, int tileWidth)
{
    extern __shared__ float shared[];
    float* Mds = shared;
    float* Nds = shared + tileWidth * tileWidth;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.y * tileWidth + ty;
    int Col = blockIdx.x * tileWidth + tx;

    float Pvalue = 0.0f;

    for (int ph = 0; ph < width / tileWidth; ph++)
    {
        Mds[ty * tileWidth + tx] = M[Row * width + ph * tileWidth + tx];
        Nds[ty * tileWidth + tx] = N[(ph * tileWidth + ty) * width + Col];

        __syncthreads();

        for (int k = 0; k < tileWidth; k++)
        {
            Pvalue += Mds[ty * tileWidth + k] * Nds[k * tileWidth + tx];
        }

        __syncthreads();
    }

    P[Row * width + Col] = Pvalue;
}

void printPart1Resources(int tileWidth)
{
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    cudaFuncAttributes attr;
    CHECK_CUDA(cudaFuncGetAttributes(&attr, MatrixMulKernel));

    int threadsPerBlock = tileWidth * tileWidth;
    size_t dynamicSharedBytes = 2ULL * tileWidth * tileWidth * sizeof(float);

    int blocksPerSM = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        MatrixMulKernel,
        threadsPerBlock,
        dynamicSharedBytes));

    int activeThreadsPerSM = blocksPerSM * threadsPerBlock;
    int maxTotalActiveThreadsLaunch = activeThreadsPerSM * prop.multiProcessorCount;
    int hardwareMaxThreadsDevice =
        prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;

    printf("\n=============================================\n");
    printf("Part 1 Kernel Resource Report\n");
    printf("=============================================\n");
    printf("CUDA device name                    : %s\n", prop.name);
    printf("Streaming multiprocessors (SMs)     : %d\n", prop.multiProcessorCount);
    printf("Tile width                          : %d\n", tileWidth);
    printf("Threads per block                   : %d\n", threadsPerBlock);
    printf("Max threads per SM (hardware)       : %d\n", prop.maxThreadsPerMultiProcessor);
    printf("\n");

    printf("Kernel registers per thread         : %d\n", attr.numRegs);
    printf("Static shared memory per block      : %zu bytes\n", attr.sharedSizeBytes);
    printf("Dynamic shared memory per block     : %zu bytes\n", dynamicSharedBytes);
    printf("Total shared memory per block       : %zu bytes\n",
           attr.sharedSizeBytes + dynamicSharedBytes);
    printf("\n");

    printf("Blocks per SM for this launch       : %d\n", blocksPerSM);
    printf("Active threads per SM for launch    : %d\n", activeThreadsPerSM);
    printf("Max total active threads for launch : %d\n", maxTotalActiveThreadsLaunch);
    printf("Hardware max threads on device      : %d\n", hardwareMaxThreadsDevice);
    printf("=============================================\n");

    // Helpful wording for the report
    printf("\nFor question (a):\n");
    printf("Using this kernel launch, the maximum simultaneously scheduled/executing\n");
    printf("threads on the device is %d.\n", maxTotalActiveThreadsLaunch);

    printf("\nFor question (b):\n");
    printf("Registers/thread = %d\n", attr.numRegs);
    printf("Shared memory/block = %zu bytes total (%zu static + %zu dynamic)\n",
           attr.sharedSizeBytes + dynamicSharedBytes,
           attr.sharedSizeBytes,
           dynamicSharedBytes);
    printf("Blocks/SM = %d\n", blocksPerSM);
    printf("Maximum total active threads = %d\n", maxTotalActiveThreadsLaunch);
}

int main(int argc, char** argv)
{
    int tileWidth = 15;

    if (argc >= 2)
        tileWidth = atoi(argv[1]);

    if (tileWidth <= 0)
    {
        printf("Usage: %s [tileWidth]\n", argv[0]);
        return 1;
    }

    if (tileWidth * tileWidth > 1024)
    {
        printf("Error: tileWidth * tileWidth must be <= 1024\n");
        return 1;
    }

    printPart1Resources(tileWidth);

    cudaDeviceReset();
    return 0;
}