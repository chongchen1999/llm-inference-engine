#include <algorithm>   // std::fill_n
#include <iostream>    // std::cout, std::endl
#include <cmath>       // std::cos, std::sin, std::pow
#include <cstdlib>     // std::malloc, std::free
#include <string>      // std::string
#include <vector>      // std::vector
#include <ctime>       // std::time

#include "../../src/kernels/includes/qkv_bias_and_rope.cuh"
#include "../../src/weights/includes/attention_weights.h"
#include "../../src/utils/macro.h"

// CPU function for processing Q, K, V
void CPUfunc(
    float *q,
    float *k,
    float *v,
    float *QKV,
    const float *qkv_bias,
    const int *padding_offset,
    const int *history_length,
    const int *input_length,
    const int batch_size,
    const int seq_len,
    const int token_num,
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int rotary_embedding_dim,
    float rotary_embedding_base
) {
    int qbatchstride = seq_len * head_num * head_size;
    int kvbatchstride = seq_len * kv_head_num * head_size;
    int batchstride = seq_len * (head_num + 2 * kv_head_num) * head_size;

    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int timestep = history_length[b] + s;
            int base_offset = b * batchstride + s * (head_num + 2 * kv_head_num) * head_size;

            // Process Q
            for (int head = 0; head < head_num; ++head) {
                int base_offset_q = base_offset + head * head_size;
                for (int d = 0; d < head_size / 2; ++d) {
                    int transposed_offset = b * qbatchstride + head * seq_len * head_size + s * head_size + d;
                    float x0 = QKV[base_offset_q + d];
                    float x1 = QKV[base_offset_q + d + 64];
                    float inv_freq = timestep / std::pow(rotary_embedding_base, (float)(d * 2) / rotary_embedding_dim);

                    q[transposed_offset] = x0 * std::cos(inv_freq) - x1 * std::sin(inv_freq);
                    q[transposed_offset + 64] = x1 * std::cos(inv_freq) + x0 * std::sin(inv_freq);
                }
            }

            // Process K
            for (int head = 0; head < kv_head_num; ++head) {
                int base_offset_k = base_offset + (head + head_num) * head_size;
                // int base_offset_v = base_offset + (head + head_num + kv_head_num) * head_size;

                for (int d = 0; d < head_size / 2; ++d) {
                    int transposed_offset = b * kvbatchstride + head * seq_len * head_size + s * head_size + d;
                    float x0 = QKV[base_offset_k + d];
                    float x1 = QKV[base_offset_k + d + 64];
                    float inv_freq = timestep / std::pow(rotary_embedding_base, (float)(d * 2) / rotary_embedding_dim);

                    k[transposed_offset] = x0 * std::cos(inv_freq) - x1 * std::sin(inv_freq);
                    k[transposed_offset + 64] = x1 * std::cos(inv_freq) + x0 * std::sin(inv_freq);
                }
            }
        }
    }
}

// Function to check results
bool checkResult(float *q, float *k, float *hq, float *hk, 
                 const int q_size, const int k_size) {
    for (int i = 0; i < q_size; ++i) {
        if (std::fabs(q[i] - hq[i]) > 1e-3) {
            std::printf("The %dth q is wrong, q = %f, hq = %f\n", i, q[i], hq[i]);
            return false;
        }
    }
    for (int i = 0; i < k_size; ++i) {
        if (std::fabs(k[i] - hk[i]) > 1e-3) {
            std::printf("The %dth k is wrong, k = %f, hk = %f\n", i, k[i], hk[i]);
            return false;
        }
    }
    return true;
}

// Main function
int main() {
    std::srand(666233);
    const int batch_size = 1;
    const int seq_len = 32;
    const int token_num = batch_size * seq_len;
    const int head_num = 32;
    const int kv_head_num = 32;
    const int head_size = 128;
    const int rotary_embedding_dim = 128;
    const float rotary_embedding_base = 10000.0f;
    const int max_position_embeddings = 2048;

    // Allocate memory
    int *padding_offset = (int *)std::malloc(sizeof(int) * batch_size * seq_len);
    int *history_length = (int *)std::malloc(sizeof(int) * batch_size);
    int *input_length = (int *)std::malloc(sizeof(int) * batch_size);
    float *q = (float *)std::malloc(sizeof(float) * batch_size * seq_len * head_num * head_size);
    float *k = (float *)std::malloc(sizeof(float) * batch_size * seq_len * kv_head_num * head_size);
    float *v = (float *)std::malloc(sizeof(float) * batch_size * seq_len * kv_head_num * head_size);
    float *QKV = (float *)std::malloc(sizeof(float) * token_num * (head_num + 2 * kv_head_num) * head_size);
    float *qkv_bias = (float *)std::malloc(sizeof(float) * (head_num + 2 * kv_head_num) * head_size);

    // Initialize data
    std::fill_n(QKV, token_num * (head_num + 2 * kv_head_num) * head_size, 0.0f);
    std::fill_n(qkv_bias, (head_num + 2 * kv_head_num) * head_size, 2.0f);
    std::fill_n(padding_offset, batch_size * seq_len, 0);
    std::fill_n(history_length, batch_size, 0);
    std::fill_n(input_length, batch_size, 7);

    // Device memory allocation
    int *dpadding_offset, *dhistory_length, *dinput_length;
    float *dq, *dk, *dv, *dQKV, *dqkv_bias;
    cudaMalloc((void **)&dpadding_offset, sizeof(int) * batch_size * seq_len);
    cudaMalloc((void **)&dhistory_length, sizeof(int) * batch_size);
    cudaMalloc((void **)&dinput_length, sizeof(int) * batch_size);
    cudaMalloc((void **)&dq, sizeof(float) * batch_size * seq_len * head_num * head_size);
    cudaMalloc((void **)&dk, sizeof(float) * batch_size * seq_len * kv_head_num * head_size);
    cudaMalloc((void **)&dv, sizeof(float) * batch_size * seq_len * kv_head_num * head_size);
    cudaMalloc((void **)&dQKV, sizeof(float) * token_num * (head_num + 2 * kv_head_num) * head_size);
    cudaMalloc((void **)&dqkv_bias, sizeof(float) * (head_num + 2 * kv_head_num) * head_size);

    // Copy data to device
    cudaMemcpy(dinput_length, input_length, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dhistory_length, history_length, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dpadding_offset, padding_offset, sizeof(int) * batch_size * seq_len, cudaMemcpyHostToDevice);
    cudaMemcpy(dQKV, QKV, sizeof(float) * token_num * (head_num + 2 * kv_head_num) * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dqkv_bias, qkv_bias, sizeof(float) * (head_num + 2 * kv_head_num) * head_size, cudaMemcpyHostToDevice);

    // TensorWrapper initialization
    DataType type = getTensorType<float>(); 
    TensorWrapper<float> *q_buf = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, seq_len, head_size}, dq);
    TensorWrapper<float> *k_buf = new TensorWrapper<float>(Device::GPU, type, {batch_size, kv_head_num, seq_len, head_size}, dk);
    TensorWrapper<float> *v_buf = new TensorWrapper<float>(Device::GPU, type, {batch_size, kv_head_num, seq_len, head_size}, dv);
    TensorWrapper<float> *QKV_buf = new TensorWrapper<float>(Device::GPU, type, {token_num, head_num + 2 * kv_head_num, head_size}, dQKV);

    LlamaAttentionWeights<float> attn_weights;
    attn_weights.qkv.bias = dqkv_bias;

    DataType type_int = getTensorType<int>(); 
    TensorWrapper<int> *input_length_buf = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, dinput_length);
    TensorWrapper<int> *history_length_buf = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, dhistory_length);
    TensorWrapper<int> *padding_offset_buf = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, seq_len}, dpadding_offset);

    LlamaAttentionStaticParams params;
    params.rotary_embedding_dim = rotary_embedding_dim;
    params.rotary_embedding_base = rotary_embedding_base;
    params.max_position_embeddings = max_position_embeddings;
    params.use_dynamic_ntk = false;

    // Launch kernel
    std::cout << "Before launch kernel" << std::endl;
    launchFusedQKVAddBiasAndTransposeAndRope(q_buf, k_buf, v_buf, QKV_buf, &attn_weights.qkv, padding_offset_buf, history_length_buf, input_length_buf, &params);
    std::cout << "After launch kernel" << std::endl;

    // Copy results to host
    std::cout << "CUDA memcpy device to host" << std::endl;
    CHECK(cudaMemcpy(q, dq, sizeof(float) * batch_size * seq_len * head_num * head_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(k, dk, sizeof(float) * batch_size * seq_len * kv_head_num * head_size, cudaMemcpyDeviceToHost));

    // Verify results
    std::cout << "After memcpy d2h, dq[0] = " << q[0] << std::endl;
    std::cout << "Before CPU function" << std::endl;
    float *hq = (float *)std::malloc(sizeof(float) * batch_size * seq_len * head_num * head_size);
    float *hk = (float *)std::malloc(sizeof(float) * batch_size * seq_len * kv_head_num * head_size);

    CPUfunc(hq, hk, v, QKV, qkv_bias, padding_offset, history_length, input_length, batch_size, seq_len, token_num, head_num, kv_head_num, head_size, rotary_embedding_dim, rotary_embedding_base);

    std::cout << "After CPU function" << std::endl;
    bool is_right = checkResult(q, k, hq, hk, batch_size * seq_len * head_num * head_size, batch_size * seq_len * kv_head_num * head_size);
    if (is_right) {
        std::cout << "Passed" << std::endl;
    } else {
        std::cout << "Wrong Answer!" << std::endl;
    }

    // free memory
    std::free(q);
    std::free(k);
    std::free(v);
    std::free(QKV);
    std::free(qkv_bias);
    std::free(padding_offset);
    std::free(history_length);
    std::free(input_length);
    std::free(hq);
    std::free(hk);
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dQKV);
    cudaFree(dqkv_bias);
    cudaFree(dpadding_offset);
    cudaFree(dhistory_length);
    cudaFree(dinput_length);

    return 0;
}