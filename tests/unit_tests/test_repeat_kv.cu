#include <algorithm>  // std::fill_n
#include <iostream>   // std::printf
#include <cmath>      // expf, log
#include <cstdlib>    // malloc, free
#include <string>     // std::string
#include <vector>     // std::vector

#include "../../src/kernels/includes/repeat_kv.cuh"

// Note:
// - There is no repeat KV CPU kernel implementation now.
// - We compare the kernel correctness by eye.
// - Run `./test_repeat_kv` to test the FP32 GPU kernel.

int main() {
    const int batch_size = 1;
    const int head_num = 2;
    const int kv_head_num = 2;
    const int max_seq_len = 4;
    const int max_k_len = 2;
    const int head_size = 2;
    const int num_layers = 2;

    const int k_size = num_layers * batch_size * kv_head_num * max_seq_len * head_size;
    const int out_k_size = batch_size * head_num * max_k_len * head_size;

    // Allocate host memory
    float *h_k = static_cast<float *>(malloc(sizeof(float) * k_size));
    float *h_v = static_cast<float *>(malloc(sizeof(float) * k_size));
    int *h_ctx_len = static_cast<int *>(malloc(sizeof(int) * batch_size));
    float *h_trans_k = static_cast<float *>(malloc(sizeof(float) * out_k_size));
    float *h_trans_v = static_cast<float *>(malloc(sizeof(float) * out_k_size));
    int *h_layer_id = static_cast<int *>(malloc(sizeof(int) * batch_size));

    // Allocate device memory
    float *d_k;
    float *d_v;
    int *d_ctx_len;
    float *d_trans_k;
    float *d_trans_v;

    cudaMalloc(reinterpret_cast<void **>(&d_k), sizeof(float) * k_size);
    cudaMalloc(reinterpret_cast<void **>(&d_v), sizeof(float) * k_size);
    cudaMalloc(reinterpret_cast<void **>(&d_ctx_len), sizeof(int) * batch_size);
    cudaMalloc(reinterpret_cast<void **>(&d_trans_k), sizeof(float) * out_k_size);
    cudaMalloc(reinterpret_cast<void **>(&d_trans_v), sizeof(float) * out_k_size);

    // Initialize host memory
    for (int i = 0; i < k_size; ++i) {
        h_v[i] = static_cast<float>(i);
        h_k[i] = static_cast<float>(i);
    }

    for (int i = 0; i < batch_size; ++i) {
        h_ctx_len[i] = 2;
        h_layer_id[i] = 0;
    }

    // Copy data from host to device
    cudaMemcpy(d_k, h_k, sizeof(float) * k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeof(float) * k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx_len, h_ctx_len, sizeof(int) * batch_size, cudaMemcpyHostToDevice);

    // Set tensor types
    DataType type = getTensorType<float>();
    DataType type_int = getTensorType<int>();

    // Create TensorWrapper instances
    auto *in_k = new TensorWrapper<float>(Device::GPU, type, {num_layers, batch_size, kv_head_num, max_seq_len, head_size}, d_k);
    auto *in_v = new TensorWrapper<float>(Device::GPU, type, {num_layers, batch_size, kv_head_num, max_seq_len, head_size}, d_v);
    auto *ctx_len = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_ctx_len);
    auto *out_k = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size}, d_trans_k);
    auto *out_v = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size}, d_trans_v);
    auto *layer_id = new TensorWrapper<int>(Device::CPU, type_int, {batch_size}, h_layer_id);

    std::cout << "Before launching repeat KV kernel" << std::endl;
    launchRepeatKVCache(in_k, in_v, ctx_len, layer_id, out_k, out_v);
    std::cout << "After launching repeat KV kernel" << std::endl;
    std::cout << "CUDA memcpy device to host" << std::endl;

    // Copy data from device to host
    cudaMemcpy(h_trans_k, out_k->data, sizeof(float) * out_k_size, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < out_k_size; ++i) {
        std::printf("k trans[%d] = %f\n", i, h_trans_k[i]);
    }

    // Clean up
    free(h_k);
    free(h_v);
    free(h_ctx_len);
    free(h_trans_k);
    free(h_trans_v);
    free(h_layer_id);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_ctx_len);
    cudaFree(d_trans_k);
    cudaFree(d_trans_v);

    return 0;
}