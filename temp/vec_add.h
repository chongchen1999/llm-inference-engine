#pragma once
#include <cuda_fp16.h>

// CUDA kernel for vector addition using half-precision
__global__ void vectorAdd(const half* A, const half* B, half* C, int N);

// Helper function to check CUDA errors
void checkCudaErrors(cudaError_t err);
