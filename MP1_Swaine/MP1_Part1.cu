// Owen Swaine - 20386155

#include <stdio.h>         
#include <cuda_runtime.h>   

// simple helper to estimate CUDA cores per SM based on architecture
// See report for where I got this info
int getCoresPerSM(int major)
{
    if (major == 2) return 32;    // Fermi
    if (major == 3) return 192;   // Kepler
    if (major == 5) return 128;   // Maxwell
    if (major == 6) return 64;    // Pascal
    if (major == 7) return 64;    // Volta
    if (major == 8) return 128;   // Ampere
    return 0;                     // unknown arch
}

int main() {

    int deviceCount;

    // get how many CUDA GPUs are available
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    // loop through each detected GPU (only 1 in the bain lab I had)
    for(int i = 0; i < deviceCount; i++) {

        cudaDeviceProp prop;

        // fill struct with device info
        cudaGetDeviceProperties(&prop, i);

        // total CUDA cores
        int coresPerSM = getCoresPerSM(prop.major);
        int totalCores = coresPerSM * prop.multiProcessorCount;

        printf("Device %d: %s\n", i, prop.name);
        printf("Clock Rate: %d\n", prop.clockRate);
        printf("Streaming Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("CUDA Cores: %d\n", totalCores);
        printf("Warp Size: %d\n", prop.warpSize);
        printf("Global Memory: %zu bytes\n", prop.totalGlobalMem);
        printf("Constant Memory: %zu bytes\n", prop.totalConstMem);
        printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("Registers per Block: %d\n", prop.regsPerBlock);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);

        // maximum size of a thread block
        printf("Max Block Dimensions: %d x %d x %d\n",
            prop.maxThreadsDim[0],
            prop.maxThreadsDim[1],
            prop.maxThreadsDim[2]);

        // maximum size of the grid
        printf("Max Grid Dimensions: %d x %d x %d\n\n",
            prop.maxGridSize[0],
            prop.maxGridSize[1],
            prop.maxGridSize[2]);
    }

    return 0;
}