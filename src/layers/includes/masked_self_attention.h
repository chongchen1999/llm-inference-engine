#pragma once

#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/includes/linear.h"
#include "src/kernels/includes/fused_decoder_self_attention.h"
#include "src/kernels/includes/qkv_bias_and_rope.h" // 2nd rope
#include "src/utils/tensor.h"
// #include "src/kernels/cublas_utils.h"
#include "src/models/llama/llama_params.h"
#include "src/utils/macro.h"

// (RussWong) Note: The data members here are only present in the attention layer,
// unlike finished and sequence lengths which span the entire process.

template<typename T>
class LlamaSelfAttentionLayer {
private:
    // Shared parameters across all LLMs
    const int head_num;
    const int head_size;
    const int hidden_units;
    const int repeats_per_kv; // For GQA and MQA
    const int kv_head_num;
    float scale;

    // Parameters specific to llama and unchanged
    LlamaAttentionStaticParams attention_static_params;
    
    cudaStream_t stream;
    BaseAllocator *allocator;
    cublasWrapper *cublas_wrapper;

    // Intermediate buffers
    TensorWrapper<T> *qkv_buf = nullptr;     // For qkv linear output and rope input/output
    TensorWrapper<T> *mha_output = nullptr;  // MHA output, then invoke a linear to attention output

public:
    LlamaSelfAttentionLayer(
        int head_num,
        int kv_head_num,
        int head_size,
        LlamaAttentionStaticParams *attn_params,
        cudaStream_t stream,
        cublasWrapper *cublas_wrapper,
        BaseAllocator *allocator
    );

    // (RussWong) Note: Private data members can only be accessed by member functions
    LlamaAttentionStaticParams *getAttnStaticParams() {
        return &attention_static_params;
    }

    void AllocateMemoryForForward(LlamaAttentionDynamicParams *params);
    void FreeBuf();
    void Forward(
        TensorMap *inputs,
        TensorMap *outputs,
        LlamaAttentionWeights<T> *weights,
        LlamaAttentionDynamicParams *params
    );
};
