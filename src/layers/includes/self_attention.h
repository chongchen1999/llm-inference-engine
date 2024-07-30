#pragma once

#include "../../weights/llama/attention_weights.h"
#include "../../memory/allocator/cuda_allocator.h"
#include "../../kernels/includes/linear.h"
#include "../../kernels/includes/decoder_self_attention.h"
#include "../../kernels/includes/qkv_bias_and_rope.h" // 2nd rope
#include "../../utils/tensor.h"
#include "../../kernels/includes/cublas_utils.h"
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
