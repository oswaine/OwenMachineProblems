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

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                         \
do {                                                                             \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                  << " -> " << cudaGetErrorString(err) << std::endl;             \
        std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
} while (0)

static const float EPSILON = 1e-3f;

// ---------------------------------------------
// Utility functions
// ---------------------------------------------
void fillRandom(float* A, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void zeroMatrix(float* A, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = 0.0f;
    }
}

bool compareMatrices(const float* A, const float* B, int n, float tol = EPSILON) {
    for (int i = 0; i < n * n; i++) {
        float diff = std::fabs(A[i] - B[i]);
        if (diff > tol) {
            std::cout << "Mismatch at index " << i
                      << " CPU=" << A[i]
                      << " GPU=" << B[i]
                      << " diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

// ---------------------------------------------
// CPU reference
// ---------------------------------------------
void matMulCPU(float* P, const float* M, const float* N, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = sum;
        }
    }
}

// ---------------------------------------------
// GPU kernels
// ---------------------------------------------

// One thread computes the whole matrix
__global__ void matMulSingleThread(float* P, const float* M, const float* N, int width) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < width; col++) {
                float sum = 0.0f;
                for (int k = 0; k < width; k++) {
                    sum += M[row * width + k] * N[k * width + col];
                }
                P[row * width + col] = sum;
            }
        }
    }
}

// General kernel: each thread computes one output element
__global__ void matMulElementPerThread(float* P, const float* M, const float* N, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = sum;
    }
}

// ---------------------------------------------
// Timing helpers
// ---------------------------------------------
float timeHtoD(float* d_M, float* d_N, const float* h_M, const float* h_N, int bytes, int repeats = 20) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples;
    samples.reserve(repeats);

    for (int i = 0; i < repeats; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float sum = 0.0f;
    for (float x : samples) sum += x;
    return sum / samples.size();
}

float timeDtoH(float* h_P, float* d_P, int bytes, int repeats = 20) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples;
    samples.reserve(repeats);

    for (int i = 0; i < repeats; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float sum = 0.0f;
    for (float x : samples) sum += x;
    return sum / samples.size();
}

float timeKernelSingleThread(float* d_P, const float* d_M, const float* d_N, int width, int repeats = 10) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples;
    samples.reserve(repeats);

    for (int i = 0; i < repeats; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        matMulSingleThread<<<1, 1>>>(d_P, d_M, d_N, width);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float sum = 0.0f;
    for (float x : samples) sum += x;
    return sum / samples.size();
}

float timeKernelElementPerThread(float* d_P, const float* d_M, const float* d_N,
                                 int width, int blockWidth, int repeats = 10) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(blockWidth, blockWidth);
    dim3 grid((width + block.x - 1) / block.x,
              (width + block.y - 1) / block.y);

    std::vector<float> samples;
    samples.reserve(repeats);

    for (int i = 0; i < repeats; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        matMulElementPerThread<<<grid, block>>>(d_P, d_M, d_N, width);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float sum = 0.0f;
    for (float x : samples) sum += x;
    return sum / samples.size();
}

double timeCPU(float* h_P, const float* h_M, const float* h_N, int width, int repeats = 3) {
    std::vector<double> samples;
    samples.reserve(repeats);

    for (int i = 0; i < repeats; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        matMulCPU(h_P, h_M, h_N, width);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = t2 - t1;
        samples.push_back(ms.count());
    }

    double sum = 0.0;
    for (double x : samples) sum += x;
    return sum / samples.size();
}

// ---------------------------------------------
// One full experiment for a given size
// ---------------------------------------------
void runExperiment(int width) {
    std::cout << "\n========================================\n";
    std::cout << "Matrix size: " << width << " x " << width << "\n";
    std::cout << "========================================\n";

    int elements = width * width;
    int bytes = elements * sizeof(float);

    float* h_M = new float[elements];
    float* h_N = new float[elements];
    float* h_P_cpu = new float[elements];
    float* h_P_gpu = new float[elements];

    srand(42);
    fillRandom(h_M, width);
    fillRandom(h_N, width);
    zeroMatrix(h_P_cpu, width);
    zeroMatrix(h_P_gpu, width);

    float *d_M = nullptr, *d_N = nullptr, *d_P = nullptr;
    CUDA_CHECK(cudaMalloc(&d_M, bytes));
    CUDA_CHECK(cudaMalloc(&d_N, bytes));
    CUDA_CHECK(cudaMalloc(&d_P, bytes));

    // Copy timing
    float h2d_ms = timeHtoD(d_M, d_N, h_M, h_N, bytes);
    CUDA_CHECK(cudaMemset(d_P, 0, bytes));
    float d2h_ms = timeDtoH(h_P_gpu, d_P, bytes);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Average H->D time for 2 input matrices: " << h2d_ms << " ms\n";
    std::cout << "Average D->H time for 1 output matrix : " << d2h_ms << " ms\n";

    // CPU timing and reference
    double cpu_ms = timeCPU(h_P_cpu, h_M, h_N, width);
    std::cout << "Average CPU matrix multiply time      : " << cpu_ms << " ms\n";

    // Load inputs once before kernel timing
    CUDA_CHECK(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));

    // Part (b): single block, one thread
    if (width == 300 || width == 750) {
        CUDA_CHECK(cudaMemset(d_P, 0, bytes));
        float gpu_single_ms = timeKernelSingleThread(d_P, d_M, d_N, width);
        CUDA_CHECK(cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost));

        bool ok1 = compareMatrices(h_P_cpu, h_P_gpu, width);
        std::cout << "GPU kernel time (1 block, 1 thread)  : " << gpu_single_ms << " ms\n";
        std::cout << (ok1 ? "Test PASSED\n" : "Test FAILED\n");

        // Part (b): block width = 1
        CUDA_CHECK(cudaMemset(d_P, 0, bytes));
        float gpu_bw1_ms = timeKernelElementPerThread(d_P, d_M, d_N, width, 1);
        CUDA_CHECK(cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost));

        bool ok2 = compareMatrices(h_P_cpu, h_P_gpu, width);
        std::cout << "GPU kernel time (block width = 1)    : " << gpu_bw1_ms << " ms\n";
        std::cout << (ok2 ? "Test PASSED\n" : "Test FAILED\n");

        std::cout << "Total offload time approx (bw=1)     : "
                  << (h2d_ms + gpu_bw1_ms + d2h_ms) << " ms\n";
    }

    // Part (c): block widths 2,5,10,15,25
    int blockWidths[] = {2, 5, 10, 15, 25};
    for (int bw : blockWidths) {
        CUDA_CHECK(cudaMemset(d_P, 0, bytes));
        float kernel_ms = timeKernelElementPerThread(d_P, d_M, d_N, width, bw);
        CUDA_CHECK(cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost));

        bool ok = compareMatrices(h_P_cpu, h_P_gpu, width);
        std::cout << "GPU kernel time (block width = " << std::setw(2) << bw
                  << ") : " << kernel_ms << " ms   "
                  << (ok ? "[PASSED]" : "[FAILED]") << "\n";
    }

    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));

    delete[] h_M;
    delete[] h_N;
    delete[] h_P_cpu;
    delete[] h_P_gpu;
}

int main() {
    int sizes[] = {300, 750, 1500, 3000, 4500};

    for (int n : sizes) {
        runExperiment(n);
    }

    return 0;
}
