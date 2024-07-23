#include <algorithm>      // std::fill_n
#include <cmath>         // expf, log
#include <cstdlib>       // rand
#include <iostream>      // snprintf
#include <string>        // std::string
#include <vector>        // std::vector

#include "src/kernels/includes/fused_decoder_self_attention.h"
#include "src/utils/macro.h"

// Bug1: MUST add CHECK to cudaMemcpy to see if it works well
// Note: This CPU implementation still has bugs.
// When implementing LLM inference on CPU, reuse the CPU kernel to test its correctness.
// Use `./test_fused_decoder_attention` to test fp32 kernel
template <typename T>
void CPUMaskedAttention(
    T *q,
    T *k,
    T *v,
    T *k_cache,
    T *v_cache,
    float *mha_output,
    const int batch_size,
    const int num_heads,
    const int head_size,
    const int step
) {
    const int batch_stride = num_heads * head_size;
    const int head_stride = head_size;
    const int cache_offset = batch_size * batch_stride;
    const int block_nums = batch_size * num_heads;
    const float scale = rsqrt(float(head_size));

    const T *q_mem = q;
    const T *k_mem = k;
    const T *v_mem = v;

    // Temp buffer
    float *shared_qk = static_cast<float *>(malloc(sizeof(float) * (block_nums * (3 * head_size + step))));
    float *sq = shared_qk;
    float *sk = sq + block_nums * head_size;
    float *logits = sk + block_nums * head_size;
    float *sv = logits + block_nums * step;

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        for (int head_id = 0; head_id < num_heads; ++head_id) {
            float row_max = 0.0f;

            for (int iter = 0; iter < step; ++iter) {
                float attn_score = 0.0f;

                for (int tid = 0; tid < head_size; ++tid) {
                    const int qkv_offset = batch_id * batch_stride + head_id * head_stride + tid;

                    // Note: sq and sk's offset should be qkv_offset, not tid
                    sk[qkv_offset] = static_cast<float>(k_cache[iter * cache_offset + qkv_offset]);

                    // When final step, update k cache
                    if (iter == step - 1) {
                        // TODO: update k cache with k with bias add
                        k_cache[iter * cache_offset + qkv_offset] = k_mem[qkv_offset];
                        sk[qkv_offset] = static_cast<float>(k_mem[qkv_offset]);
                    }

                    sq[qkv_offset] = static_cast<float>(q_mem[qkv_offset]);
                    const float qk = sq[qkv_offset] * sk[qkv_offset] * scale;
                    attn_score += qk;
                }

                logits[batch_id * num_heads * step + head_id * step + iter] = attn_score;
                row_max = std::max(attn_score, row_max);
            }

            float fenzi = 0.0f;
            float fenmu = 0.0f;

            for (int iter = 0; iter < step; ++iter) {
                fenzi = expf(logits[batch_id * num_heads * step + head_id * step + iter] - row_max);
                fenmu += fenzi;
            }

            for (int iter = 0; iter < step; ++iter) {
                logits[batch_id * num_heads * step + head_id * step + iter] = fenzi / fenmu;
            }

            for (int tid = 0; tid < head_size; ++tid) {
                float O = 0.0f;
                const int qkv_offset = batch_id * batch_stride + head_id * head_stride + tid;

                for (int iter = 0; iter < step; ++iter) {
                    sv[qkv_offset] = static_cast<float>(v_cache[iter * cache_offset + qkv_offset]);

                    if (iter == step - 1) {
                        v_cache[iter * cache_offset + qkv_offset] = v_mem[qkv_offset];
                        sv[qkv_offset] = static_cast<float>(v_mem[qkv_offset]);
                    }

                    O += sv[qkv_offset] * logits[batch_id * num_heads * step + head_id * step + iter];
                }

                mha_output[qkv_offset] = O;
            }
        }
    }

    free(shared_qk);
}

template <typename T>
bool checkResult(
    const float *CPUoutput,
    const T *GPUoutput,
    const int output_size
) {
    for (int i = 0; i < output_size; ++i) {
        const float GPUres = static_cast<float>(GPUoutput[i]);
        if (fabs(CPUoutput[i] - GPUres) > 1e-6) {
            printf("The %dth result is wrong: CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUres);
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    constexpr int batch_size = 1;
    constexpr int head_size = 4;
    constexpr int num_heads = 2;
    constexpr int kv_num_heads = 2;
    constexpr int max_seq_len = 4;
    const int h_step = 4;
    const int h_layer_id = 0;
    const int rotary_embedding_dim = 128;
    const float rotary_embedding_base = 10000;
    const int max_position_embeddings = 2048;
    const bool use_dynamic_ntk = false; // for dyn scaling rope

    float *h_qkv, *d_qkv;
    const int qkv_size = batch_size * (2 * kv_num_heads + num_heads) * head_size;
    h_qkv = static_cast<float *>(malloc(sizeof(float) * qkv_size));
    cudaMalloc(reinterpret_cast<void **>(&d_qkv), sizeof(float) * qkv_size);

    float *h_kcache, *d_kcache;
    const int kcache_size = max_seq_len * batch_size * kv_num_heads * head_size;
    h_kcache = static_cast<float *>(malloc(sizeof(float) * kcache_size));
    cudaMalloc(reinterpret_cast<void **>(&d_kcache), sizeof(float) * kcache_size);

    float *h_vcache, *d_vcache;
    const int vcache_size = max_seq_len * batch_size * kv_num_heads * head_size;
    h_vcache = static_cast<float *>(malloc(sizeof(float) * vcache_size));
    cudaMalloc(reinterpret_cast<void **>(&d_vcache), sizeof(float) * vcache_size);

    for (int i = 0; i < qkv_size; ++i) {
        if (i < batch_size * num_heads * head_size) {
            if (i < batch_size * num_heads * head_size / 2) {
                h_qkv[i] = static_cast<float>(i + 1);
            } else {
                h_qkv[i] = static_cast<float>(i - 3) / 10;
            }
        } else if (i < batch_size * (num_heads + kv_num_heads) * head_size) {
            if (i < batch_size * (num_heads + kv_num_heads / 2) * head_size) {
                h_qkv[i] = static_cast<float>(i + 5);
            } else {
                h_qkv[i] = static_cast<float>(i + 1) / 10;
            }
        } else if (i < batch_size * (num_heads + kv_num_heads * 2) * head_size) {
            if (i < batch_size * (num_heads + kv_num_heads + kv_num_heads / 2) * head_size) {
                h_qkv[i] = static_cast<float>(i - 3);
            } else {
                h_qkv[i] = static_cast<float>(i - 7) / 10;
            }
        }
        printf("h_qkv[%d] = %f\n", i, h_qkv[i]);
    }

    float *h_q = h_qkv;
    float *h_k = h_q + batch_size * num_heads * head_size;
    float *h_v = h_k + batch_size * (kv_num_heads + num_heads) * head_size;

    for (int i = 0; i < (kcache_size * h_step) / max_seq_len; ++i) {
        if (i < kcache_size / 2) {
            h_kcache[i] = static_cast<float>(i + 1);
            h_vcache[i] = static_cast<float>(i + 1);
        } else {
            h_kcache[i] = static_cast<float>(i - kcache_size / 2 + 1) / 10;
            h_vcache[i] = static_cast<float>(i - kcache_size / 2 + 1) / 10;
        }
        printf("h_kcache[%d] = %f\n", i, h_kcache[i]);
        printf("h_vcache[%d] = %f\n", i, h_vcache[i]);
    }

    float *h_o, *d_o;
    const int o_size = batch_size * num_heads * head_size;
    h_o = static_cast<float *>(malloc(sizeof(float) * o_size));
    cudaMalloc(reinterpret_cast<void **>(&d_o), sizeof(float) * o_size);

    bool *h_finished = static_cast<bool *>(malloc(sizeof(bool) * batch_size));
    bool *d_finished;
    cudaMalloc(reinterpret_cast<void **>(&d_finished), sizeof(bool) * batch_size);

    for (int i = 0; i < batch_size; ++i) {
        h_finished[i] = false;
    }

    float *h_qkv_bias = static_cast<float *>(malloc(sizeof(float) * (2 * kv_num_heads + num_heads) * head_size));
    float *d_qkv_bias;
    cudaMalloc(reinterpret_cast<void **>(&d_qkv_bias), sizeof(float) * (2 * kv_num_heads + num_heads) * head_size);

    for (int i = 0; i < (2 * kv_num_heads + num_heads) * head_size; ++i) {
        h_qkv_bias[i] = 0.0f;
    }

    cudaMemcpy(d_qkv, h_qkv, sizeof(float) * qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(float) * (2 * kv_num_heads + num_heads) * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_finished, h_finished, sizeof(bool) * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kcache, h_kcache, sizeof(float) * kcache_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vcache, h_vcache, sizeof(float) * vcache_size, cudaMemcpyHostToDevice);

    DataType type = getTensorType<float>();
    DataType type_bool = getTensorType<bool>();
    DataType type_int = getTensorType<int>();

    TensorWrapper<float> *qkv = new TensorWrapper<float>(
        Device::GPU, type, 
        {batch_size, num_heads + 2 * kv_num_heads, head_size}, 
        d_qkv
    );
    TensorWrapper<float> *kcache = new TensorWrapper<float>(
        Device::GPU, type, 
        {h_layer_id, batch_size, kv_num_heads, max_seq_len, head_size}, 
        d_kcache
    );
    TensorWrapper<float> *vcache = new TensorWrapper<float>(
        Device::GPU, type, 
        {h_layer_id, batch_size, kv_num_heads, max_seq_len, head_size}, 
        d_vcache
    );
    TensorWrapper<bool> *finished = new TensorWrapper<bool>(Device::GPU, type_bool, {batch_size}, d_finished);
    TensorWrapper<int> *step = new TensorWrapper<int>(Device::CPU, type_int, {1}, const_cast<int *>(&h_step));
    TensorWrapper<int> *layer_id = new TensorWrapper<int>(Device::CPU, type_int, {1}, const_cast<int *>(&h_layer_id));
    TensorWrapper<float> *mha_output = new TensorWrapper<float>(Device::GPU, type, {batch_size, num_heads, head_size}, d_o);

    BaseWeight<float> qkv_weight;
    qkv_weight.bias = d_qkv_bias;

    LlamaAttentionStaticParams params;
    params.rotary_embedding_dim = rotary_embedding_dim;
    params.rotary_embedding_base = rotary_embedding_base;
    params.max_position_embeddings = max_position_embeddings;
    params.use_dynamic_ntk = use_dynamic_ntk;

    launchDecoderMaskedMultiHeadAttention(
        qkv,
        &qkv_weight,
        layer_id,
        kcache,
        vcache,
        finished,
        step,
        mha_output,
        &params
    );

    CHECK(cudaMemcpy(h_o, d_o, sizeof(float) * o_size, cudaMemcpyDeviceToHost));

    float *CPU_output = static_cast<float *>(malloc(sizeof(float) * o_size));
    CPUMaskedAttention<float>(h_q, h_k, h_v, h_kcache, h_vcache, CPU_output, batch_size, num_heads, head_size, h_step);

    const bool is_true = checkResult<float>(CPU_output, h_o, o_size);
    if (is_true) {
        printf("Test passed\n");
    } else {
        printf("Test failed\n");
    }

    free(h_qkv);
    free(h_kcache);
    free(h_vcache);
    free(h_o);
    free(CPU_output);
    free(h_finished);
    cudaFree(d_finished);
    cudaFree(d_qkv);
    cudaFree(d_o);
    cudaFree(d_kcache);
    cudaFree(d_vcache);

    return 0;
}
