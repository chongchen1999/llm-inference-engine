#pragma once

#include "../../weights/llama/attention_weights.h"
#include "../../memory/allocator/cuda_allocator.h"
#include "../../kernels/includes/linear.h"
#include "../../kernels/includes/scale_and_mask_and_softmax.h"
#include "../../kernels/includes/qkv_bias_and_rope.h"
#include "../../kernels/includes/transpose_and_remove_padding.h"
#include "../../kernels/includes/concat_past_kv.h"
#include "../../kernels/includes/repeat_kv.h"
#include "../../utils/tensor.h"
#include "../../kernels/includes/cublas_utils.h"
#include "../../models/llama/llama_params.h"

template <typename T>
class LlamaContextAttentionLayer {
private:
    // Parameters shared across all LLMs
    const int head_num;
    const int head_size;
    const int hidden_units;
    const int repeats_per_kv; // For GQA and MQA
    const int kv_head_num;
    float scale;

    // Parameters specific to LLaMA that are unchanged
    LlamaAttentionStaticParams *attention_static_params;

    cudaStream_t stream;
    BaseAllocator *allocator;
    CublasWrapper *cublas_wrapper;

    // Buffers for linear and batch_gemm
    TensorWrapper<T> *lineared_qkv_buf = nullptr; // 
    TensorWrapper<T> *padded_q_buf = nullptr; // intermediate
    TensorWrapper<T> *padded_k_buf = nullptr;
    TensorWrapper<T> *padded_v_buf = nullptr;
    TensorWrapper<T> *k_cache_buf = nullptr;
    TensorWrapper<T> *v_cache_buf = nullptr;
    TensorWrapper<T> *qkT_buf = nullptr;
    TensorWrapper<T> *padded_qkTv_buf = nullptr;
    TensorWrapper<T> *transposed_unpadded_qkv_buf = nullptr;

public:
    LlamaContextAttentionLayer(
        int head_num,
        int kv_head_num,
        int head_size,
        LlamaAttentionStaticParams *attention_static_params,
        cudaStream_t stream,
        cublasWrapper *cublas_wrapper,
        BaseAllocator *allocator
    );

    LlamaAttentionStaticParams *getAttnStaticParams() {
        return attention_static_params;
    }

    void allocateMemory(LlamaAttentionDynamicParams *params);
    void freeBuf();
    
    void forward(
        TensorMap *attention_layer_input,
        TensorMap *attention_layer_output,
        LlamaAttentionWeights<T> *weights,
        LlamaAttentionDynamicParams *params,
        LlamaAttentionStaticParams *static_params
    );
};
