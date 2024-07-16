#include <algorithm> // std::fill_n
#include <iostream>  // snprintf
#include <math.h>    // expf, log, rsqrtf
#include <stdlib.h>  // rand, malloc, free
#include <string>    // std::string
#include <vector>    // std::vector
#include <cuda_runtime.h> // cudaMalloc, cudaMemcpy, cudaFree

#include "src/kernels/includes/attn_softmax.h"

// `./test_mask_softmax 1` to test half GPU kernel
// `./test_mask_softmax` to test fp32 GPU kernel

template <typename T>
void test_masked_softmax(int batch_size, int head_num, int q_length, int k_length, float scale) {
    const int qk_size = batch_size * head_num * q_length * k_length;

    T *h_qk = (T *)malloc(sizeof(T) * qk_size);
    T *d_qk;
    cudaMalloc((void **)&d_qk, sizeof(T) * qk_size);

    T *h_score = (T *)malloc(sizeof(T) * qk_size);
    T *d_score;
    cudaMalloc((void **)&d_score, sizeof(T) * qk_size);

    T *h_mask = (T *)malloc(sizeof(T) * batch_size * q_length * k_length);
    T *d_mask;
    cudaMalloc((void **)&d_mask, sizeof(T) * batch_size * q_length * k_length);

    for (int i = 0; i < qk_size; ++i) {
        h_qk[i] = i % 8;
    }

    for (int i = 0; i < batch_size * q_length * k_length; ++i) {
        h_mask[i] = (T)(1);
    }

    cudaMemcpy(d_qk, h_qk, sizeof(T) * qk_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, sizeof(T) * batch_size * q_length * k_length, cudaMemcpyHostToDevice);

    DataType type = getTensorType<T>();
    TensorWrapper<T> *qk = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, q_length, k_length}, d_qk);
    TensorWrapper<T> *mask = new TensorWrapper<T>(Device::GPU, type, {batch_size, q_length, k_length}, d_mask);
    TensorWrapper<T> *score = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, q_length, k_length}, d_score);

    std::cout << "before launch softmax kernel" << std::endl;
    launchScaleMaskAndSoftmax(qk, mask, score, scale);
    std::cout << "after launch softmax kernel" << std::endl;
    std::cout << "cuda memcpy device to host" << std::endl;

    cudaMemcpy(h_score, score->data, sizeof(T) * qk_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < qk_size; ++i) {
        printf("attn score[%d] = %f\n", i, (float)h_score[i]);
    }

    free(h_qk);
    free(h_score);
    free(h_mask);
    cudaFree(d_qk);
    cudaFree(d_score);
    cudaFree(d_mask);
}

int main(int argc, char *argv[]) {
    const int batch_size = 1;
    const int head_num = 2;
    const int q_length = 8;
    const int k_length = 8;
    const int head_size = 4;
    float scale = rsqrtf(static_cast<float>(head_size));

    if (argc > 1 && argv[1]) {
        test_masked_softmax<half>(batch_size, head_num, q_length, k_length, scale);
    } else {
        test_masked_softmax<float>(batch_size, head_num, q_length, k_length, scale);
    }

    return 0;
}