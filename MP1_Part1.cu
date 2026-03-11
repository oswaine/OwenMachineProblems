#include <stdio.h>
#include <cuda_runtime.h>

int getCoresPerSM(int major)
{
    if (major == 2) return 32;
    if (major == 3) return 192;
    if (major == 5) return 128;
    if (major == 6) return 64;
    if (major == 7) return 64;
    if (major == 8) return 128;
    return 0;
}

int main() {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for(int i = 0; i < deviceCount; i++) {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        int coresPerSM = getCoresPerSM(prop.major);
        int totalCores = coresPerSM * prop.multiProcessorCount;

        printf("Device %d: %s\n", i, prop.name);
        printf("Clock Rate: %d\n", prop.clockRate);
        printf("Streaming Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("CUDA Cores: %d\n", totalCores);
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