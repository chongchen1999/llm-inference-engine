#pragma once

#include "../../weights/includes/attention_weights.h"
#include "../../memory/allocator/cuda_allocator.h"
#include "../../kernels/includes/linear.cuh"
#include "../../kernels/includes/scale_and_mask_and_softmax.cuh"
#include "../../kernels/includes/qkv_bias_and_rope.cuh"
#include "../../kernels/includes/transpose_and_remove_padding.cuh"
#include "../../kernels/includes/concat_past_kv.cuh"
#include "../../kernels/includes/repeat_kv.cuh"
#include "../../utils/tensor.h"
#include "../../kernels/includes/cublas_utils.cuh"
#include "../../models/llama/llama_params.h"

template <typename T>
class LlamaContextAttentionLayer {
private:
    int head_num;
    int head_size;
    int hidden_units;
    int repeats_per_kv; // For GQA and MQA
    int kv_head_num;
    float scale;

    // Parameters specific to LLaMA that are unchanged
    LlamaAttentionStaticParams *attention_static_params;

    cudaStream_t stream;
    BaseAllocator *allocator;
    CublasWrapper *cublas_wrapper;

    // intermediate buffers for linear and batch_gemm
    TensorWrapper<T> *lineared_qkv = nullptr;
    TensorWrapper<T> *padded_q = nullptr;
    TensorWrapper<T> *padded_k = nullptr;
    TensorWrapper<T> *padded_v = nullptr;
    TensorWrapper<T> *k_cache = nullptr;
    TensorWrapper<T> *v_cache = nullptr;
    TensorWrapper<T> *qkT = nullptr;
    TensorWrapper<T> *padded_qkTv = nullptr;
    TensorWrapper<T> *transposed_unpadded_qkv = nullptr;

public:
    LlamaContextAttentionLayer(
        int head_num,
        int kv_head_num,
        int head_size,
        LlamaAttentionStaticParams *attention_static_params,
        cudaStream_t stream,
        CublasWrapper *cublas_wrapper,
        BaseAllocator *allocator
    );

    LlamaAttentionStaticParams *getAttentionStaticParams() {
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
