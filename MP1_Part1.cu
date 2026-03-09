#include <stdio.h>
#include <cuda_runtime.h>

int main() {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for(int i = 0; i < deviceCount; i++) {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("Clock Rate: %d\n", prop.clockRate);
        printf("Streaming Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("Warp Size: %d\n", prop.warpSize);
        printf("Global Memory: %ld bytes\n", prop.totalGlobalMem);
        printf("Constant Memory: %ld bytes\n", prop.totalConstMem);
        printf("Shared Memory per Block: %ld bytes\n", prop.sharedMemPerBlock);
        printf("Registers per Block: %d\n", prop.regsPerBlock);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);

        printf("Max Block Dimensions: %d x %d x %d\n",
            prop.maxThreadsDim[0],
            prop.maxThreadsDim[1],
            prop.maxThreadsDim[2]);

        printf("Max Grid Dimensions: %d x %d x %d\n\n",
            prop.maxGridSize[0],
            prop.maxGridSize[1],
            prop.maxGridSize[2]);
    }

    return 0;
}