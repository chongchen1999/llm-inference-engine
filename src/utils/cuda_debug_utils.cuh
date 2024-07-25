#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

template<typename T>
__global__ void print_data(T *src1, bool is_target = false) {
    int tid = threadIdx.x;
    
    if (tid == 0) {
        printf("%dth = %f\n", tid, src1[tid]);
        printf("%dth = %f\n", tid + 1, src1[tid + 1]);
        
        // is_target is used to print the info for a specified function,
        // to avoid too much print info on the screen.
        if (is_target) {
            printf("%dth = %f\n", tid + 128, src1[tid + 128]);
            printf("%dth = %f\n", tid + 129, src1[tid + 129]);
            printf("%dth = %f\n", tid + 130, src1[tid + 130]);
            printf("%dth = %f\n", tid + 131, src1[tid + 131]);
            printf("%dth = %f\n", tid + 1024, src1[tid + 1024]);
        }
    }
}
