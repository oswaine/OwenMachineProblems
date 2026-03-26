// Owen Swaine - 20386155
// ELEC 374 Machine Problem 2 - Part 2
// Tiled non-square matrix multiplication with boundary checks

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TOL 1e-3f

__global__ void MatrixMulKernel(float* M, float* N, float* P,
                                int m, int n, int k,
                                int tileWidthX, int tileWidthY)
{
    extern __shared__ float shared[];
    float* Mds = shared;
    float* Nds = shared + tileWidthX * tileWidthY;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.y * tileWidthY + ty;
    int Col = blockIdx.x * tileWidthX + tx;

    float Pvalue = 0.0f;

    int numPhases = (n + tileWidthX - 1) / tileWidthX;

    for (int ph = 0; ph < numPhases; ph++)
    {
        int tiledColM = ph * tileWidthX + tx;
        int tiledRowN = ph * tileWidthX + ty;

        if (Row < m && tiledColM < n)
            Mds[ty * tileWidthX + tx] = M[Row * n + tiledColM];
        else
            Mds[ty * tileWidthX + tx] = 0.0f;

        if (tiledRowN < n && Col < k)
            Nds[ty * tileWidthX + tx] = N[tiledRowN * k + Col];
        else
            Nds[ty * tileWidthX + tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < tileWidthX; i++)
        {
            Pvalue += Mds[ty * tileWidthX + i] * Nds[i * tileWidthX + tx];
        }

        __syncthreads();
    }

    if (Row < m && Col < k)
        P[Row * k + Col] = Pvalue;
}

void fillRandom(float* A, int count)
{
    for (int i = 0; i < count; i++)
        A[i] = (float)rand() / RAND_MAX;
}

void matMulCPU(float* P, float* M, float* N, int m, int n, int k)
{
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < k; col++)
        {
            float sum = 0.0f;
            for (int i = 0; i < n; i++)
            {
                sum += M[row * n + i] * N[i * k + col];
            }
            P[row * k + col] = sum;
        }
    }
}

int compareResults(float* cpu, float* gpu, int count)
{
    for (int i = 0; i < count; i++)
    {
        if (fabs(cpu[i] - gpu[i]) > TOL)
        {
            printf("Mismatch at index %d CPU=%f GPU=%f\n", i, cpu[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

void runCase(int m, int n, int k)
{
    int tileWidthX = 14;
    int tileWidthY = 17;

    int elemsM = m * n;
    int elemsN = n * k;
    int elemsP = m * k;

    int bytesM = elemsM * sizeof(float);
    int bytesN = elemsN * sizeof(float);
    int bytesP = elemsP * sizeof(float);

    float* h_M = (float*)malloc(bytesM);
    float* h_N = (float*)malloc(bytesN);
    float* h_P_cpu = (float*)malloc(bytesP);
    float* h_P_gpu = (float*)malloc(bytesP);

    float *d_M, *d_N, *d_P;

    fillRandom(h_M, elemsM);
    fillRandom(h_N, elemsN);

    // CPU reference runs once for this case
    printf("\n[CPU] Computing reference result...\n");
    matMulCPU(h_P_cpu, h_M, h_N, m, n, k);
    printf("[CPU] Done\n");

    cudaMalloc((void**)&d_M, bytesM);
    cudaMalloc((void**)&d_N, bytesN);
    cudaMalloc((void**)&d_P, bytesP);

    cudaMemcpy(d_M, h_M, bytesM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytesN, cudaMemcpyHostToDevice);

    dim3 threads(tileWidthX, tileWidthY);
    dim3 blocks((k + tileWidthX - 1) / tileWidthX,
                (m + tileWidthY - 1) / tileWidthY);

    size_t sharedBytes = 2 * tileWidthX * tileWidthY * sizeof(float);

    printf("\n=====================================\n");
    printf("Matrix M size: %d x %d\n", m, n);
    printf("Matrix N size: %d x %d\n", n, k);
    printf("Matrix P size: %d x %d\n", m, k);
    printf("Tile size     : %d x %d\n", tileWidthY, tileWidthX);
    printf("=====================================\n");

    cudaEvent_t start, stop;
    float time_ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    MatrixMulKernel<<<blocks, threads, sharedBytes>>>(d_M, d_N, d_P,
                                                      m, n, k,
                                                      tileWidthX, tileWidthY);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(h_P_gpu, d_P, bytesP, cudaMemcpyDeviceToHost);

    printf("Kernel time: %f ms\n", time_ms);

    if (compareResults(h_P_cpu, h_P_gpu, elemsP))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");

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

int main(int argc, char** argv)
{
    srand((unsigned int)time(NULL));

    int m = 600;
    int n = 650;
    int k = 700;

    if (argc >= 2) m = atoi(argv[1]);
    if (argc >= 3) n = atoi(argv[2]);
    if (argc >= 4) k = atoi(argv[3]);

    if (m <= 0 || n <= 0 || k <= 0)
    {
        printf("Usage: %s [m] [n] [k]\n", argv[0]);
        return 1;
    }

    runCase(m, n, k);

    cudaDeviceReset();
    return 0;
}