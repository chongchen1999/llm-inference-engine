#pragma once

#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/includes/linear.h"
#include "src/kernels/includes/fused_scale_mask_and_softmax.h"
#include "src/kernels/includes/qkv_bias_and_rope.h"
#include "src/kernels/includes/fused_transpose_and_remove_padding.h"
#include "src/kernels/includes/concat_past_kv.h"
#include "src/kernels/includes/repeat_kv.h"
#include "src/utils/tensor.h"
#include "src/kernels/includes/cublas_utils.h"
#include "src/models/llama/llama_params.h"

template <typename T>
class LlamaContextAttentionLayer {
private:
    // Parameters shared across all LLMs
    const int head_num;
    const int head_size;
    const int hidden_units;
    const int q_head_per_kv; // For GQA and MQA
    const int kv_head_num;
    float scale;

    // Parameters specific to LLaMA that are unchanged
    LlamaAttentionStaticParams attn_static_params;

    cudaStream_t stream;
    BaseAllocator *allocator;
    cublasWrapper *cublas_wrapper;

    // Buffers for linear and batch_gemm
    TensorWrapper<T> *qkv_buf_without_padding = nullptr;
    TensorWrapper<T> *q_buf_with_padding = nullptr;
    TensorWrapper<T> *k_buf_with_padding = nullptr;
    TensorWrapper<T> *v_buf_with_padding = nullptr;
    TensorWrapper<T> *k_cache_buf = nullptr;
    TensorWrapper<T> *v_cache_buf = nullptr;
    TensorWrapper<T> *qk_buf = nullptr;
    TensorWrapper<T> *qkv_buf_with_padding = nullptr;
    TensorWrapper<T> *qkv_buf_without_padding_output = nullptr;

public:
    LlamaContextAttentionLayer(
        int head_num,
        int kv_head_num,
        int head_size,
        LLaMAAttentionStaticParams attn_params,
        cudaStream_t stream,
        cublasWrapper *cublas_wrapper,
        BaseAllocator *allocator
    );

    LlamaAttentionStaticParams *getAttnStaticParams() {
        return &attn_static_params;
    }

    void allocateMemoryForForward(LlamaAttentionDynamicParams *params);
    void freeBuf();
    
    void forward(
        TensorMap *inputs,
        TensorMap *outputs,
        LlamaAttentionWeights<T> *weights,
        LlamaAttentionDynamicParams *params,
        LlamaAttentionStaticParams *static_params
    );

    // Explanation of max len parameters:
    // max_seq_len: Maximum KV length considering context (e.g., multiple epochs chat)
    // max_q_len: Current maximum Q length after padding in this batch
    // max_k_len: Maximum context length in the current batch; used to adapt KV cache buffer
    //            All KV cache should be broadcast to adapt Q as KV cache buffer
    //            whose shape is max K length
    // So max K length is the max context length in the current batch
    // void flashAttn();
};
