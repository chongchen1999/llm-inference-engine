#pragma once

#include <memory>
#include "../../weights/includes/attention_weights.h"
#include "../../memory/allocator/cuda_allocator.h"
#include "../../kernels/includes/linear.cuh"
#include "../../kernels/includes/decoder_self_attention.cuh"
#include "../../kernels/includes/rope.cuh"
#include "../../utils/tensor.h"
#include "../../kernels/includes/cublas_utils.cuh"
#include "../../models/llama/llama_params.h"
#include "../../utils/macro.h"

// Note: The data members here are only present in the attention layer,
// unlike finished and sequence lengths which span the entire process.

template<typename T>
class LlamaSelfAttentionLayer {
private:
    // Shared parameters across all LLMs
    int head_num;
    int head_size;
    int hidden_units;
    int repeats_per_kv; // For GQA and MQA
    int kv_head_num;
    float scale;

    // Parameters specific to llama and unchanged
    LlamaAttentionStaticParams *attention_static_params;
    
    cudaStream_t stream;
    BaseAllocator *allocator;
    CublasWrapper *cublas_wrapper;

    // Intermediate buffers
    std::unique_ptr<TensorWrapper<T>> qkv_buf;
    std::unique_ptr<TensorWrapper<T>> mha_output;

public:
    LlamaSelfAttentionLayer(
        int head_num,
        int kv_head_num,
        int head_size,
        LlamaAttentionStaticParams *attention_params,
        cudaStream_t stream,
        CublasWrapper *cublas_wrapper,
        BaseAllocator *allocator
    );

    // Note: Private data members can only be accessed by member functions
    LlamaAttentionStaticParams *getAttentionStaticParams() {
        return attention_static_params;
    }

    void allocateMemory(LlamaAttentionDynamicParams *params);

    void freeBuf();

    void forward(
        TensorMap *inputs,
        TensorMap *outputs,
        LlamaAttentionWeights<T> *weights,
        LlamaAttentionDynamicParams *params
    );
};
