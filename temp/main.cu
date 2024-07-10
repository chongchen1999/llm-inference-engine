#include <iostream>
#include <cuda_fp16.h>
#include "vec_add.h"

// CUDA kernel for vector addition using half-precision
__global__ void vectorAdd(const half* A, const half* B, half* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = __hadd(A[i], B[i]);
    }
}

// Helper function to check CUDA errors
void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

int main() {
    int N = 1000;
    size_t size = N * sizeof(half);

    // Allocate host memory
    half *h_A = new half[N];
    half *h_B = new half[N];
    half *h_C = new half[N];

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = __float2half(static_cast<float>(i));
        h_B[i] = __float2half(static_cast<float>(i) * 2);
    }

    // Allocate device memory
    half *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, size));
    checkCudaErrors(cudaMalloc((void**)&d_B, size));
    checkCudaErrors(cudaMalloc((void**)&d_C, size));

    // Copy vectors from host memory to device memory
    checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch the vector addition kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Copy result vector from device memory to host memory
    checkCudaErrors(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < N; i++) {
        float expected = static_cast<float>(i) + static_cast<float>(i) * 2;
        float result = __half2float(h_C[i]);
        if (abs(result - expected) > 0.001) {
            std::cerr << "Result verification failed at element " << i << "!" << std::endl;
            exit(-1);
        }
    }

    std::cout << "Test PASSED" << std::endl;

    // Free device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}