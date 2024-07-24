#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void kernel() {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    static __shared__ int x = 0;
    __shared__ int y = 0;
    static int z = 0;
    x = gid;
    y = gid + x;

    z++;
    printf("thread %d, x = %d, y = %d, z = %d\n", gid, x, y, z);
}

int main() {
    const int thread_num = 512;
    printf("1st launch\n");
    kernel<<<1, thread_num>>>();
    cudaDeviceSynchronize();

    printf("2nd launch\n");
    kernel<<<1, thread_num>>>();
    cudaDeviceSynchronize();

    printf("3rd launch\n");
    kernel<<<1, thread_num>>>();
    cudaDeviceSynchronize();

    printf("over!\n");
    return 0;
}
