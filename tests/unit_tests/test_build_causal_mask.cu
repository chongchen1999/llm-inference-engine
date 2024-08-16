#include <algorithm>   // std::fill_n
#include <iostream>    // std::snprintf
#include <cmath>       // std::expf, std::log
#include <cstdlib>     // std::rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <cuda_runtime.h>

#include "../../src/kernels/includes/build_causal_mask.cuh"

// Implement LLMs inference on CPU, reusing the CPU kernel
// Compare kernel correctness by visual inspection and printed result info
void CPUbuildCausalMask(float *mask, 
                        const int *q_lens,  // input lens, shape=[batch size]
                        const int *k_lens,  // context lens, shape=[batch size]
                        int max_q_len, int max_k_len, int batch_size) {
    for (int b = 0; b < batch_size; ++b) {
        int start = b * max_q_len * max_k_len;
        int q_len = q_lens[b];
        int k_len = k_lens[b];
        for (int i = 0; i < max_q_len; ++i) {
            for (int j = 0; j < max_k_len; ++j) {
                if (j <= i + (k_len - q_len) && i < q_len && j < k_len) {
                    mask[start + i * max_k_len + j] = 1.0f;
                } else {
                    mask[start + i * max_k_len + j] = 0.0f;   
                }
            }
        }
    }
}

bool checkResult(const float * const &CPUres, const float * const &GPUres, int size) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(CPUres[i] - GPUres[i]) > 1e-3) {
            printf("The %dth result is wrong, CPU mask = %f, GPU mask = %f\n", i, CPUres[i], GPUres[i]);
            return false;
        }
    }
    return true;
}

// `./causalmask` to test fp32 GPU build causal mask kernel
int main() {
    const int batch_size = 64;
    const int max_q_len = 128;
    const int max_k_len = 512;

    // Debug info: better to retain
    // std::cout <<"batch_size=" << batch_size << " vocab_size=" << vocab_size << std::endl;

    const int mask_size = batch_size * max_q_len * max_k_len;
    
    int *h_q_lens, *d_q_lens;
    h_q_lens = (int *)malloc(sizeof(int) * batch_size);
    cudaMalloc((void **)&d_q_lens, sizeof(int) * batch_size);
    
    int *h_k_lens, *d_k_lens;
    h_k_lens = (int *)malloc(sizeof(int) * batch_size);
    cudaMalloc((void **)&d_k_lens, sizeof(int) * batch_size);
    
    float *d_mask, *h_mask;
    h_mask = (float *)malloc(sizeof(float) * mask_size);
    cudaMalloc((void **)&d_mask, sizeof(float) * mask_size);

    for (int i = 0; i < batch_size; ++i) {
        h_q_lens[i] = std::rand() % max_q_len + 1;
    }

    for (int i = 0; i < batch_size; ++i) {
        h_k_lens[i] = std::rand() % max_k_len + 1;
    }

    CHECK(cudaMemcpy(d_q_lens, h_q_lens, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_k_lens, h_k_lens, sizeof(int) * batch_size, cudaMemcpyHostToDevice));

    DataType type_float = getTensorType<float>();
    DataType type_int = getTensorType<int>();

    TensorWrapper<float> *mask = new TensorWrapper<float>(Device::GPU, type_float, {batch_size, max_q_len, max_k_len}, d_mask);
    TensorWrapper<int> *q_lens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_q_lens);
    TensorWrapper<int> *k_lens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_k_lens);

    launchBuildCausalMasks(mask, q_lens, k_lens);

    // Debug info: better to retain
    // std::cout << "after launch kernel" << std::endl;

    // Note: Remember to memcpy from device to host and define the correct copy size (multiply by sizeof(dtype)), or it will cause a segmentation fault
    CHECK(cudaMemcpy(h_mask, d_mask, sizeof(float) * mask_size, cudaMemcpyDeviceToHost));

    float *CPUmask = (float *)malloc(sizeof(float) * mask_size);
    CPUbuildCausalMask(CPUmask, h_q_lens, h_k_lens, max_q_len, max_k_len, batch_size);

    if (checkResult(CPUmask, h_mask, mask_size)) {
        printf("Test passed!\n");
    }

    // Debug info: better to retain
    // std::cout << "before free" << std::endl;

    free(h_q_lens);
    free(h_k_lens);
    free(h_mask);
    free(CPUmask);
    cudaFree(d_q_lens);
    cudaFree(d_k_lens);
    cudaFree(d_mask);

    return 0;
}