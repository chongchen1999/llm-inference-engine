#pragma once

#include "src/weights/llama/attention_weights.h"
#include "src/weights/llama/ffn_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/includes/linear.h"
#include "src/utils/tensor.h"
#include "src/kernels/includes/cublas_utils.h"
#include "src/models/llama/llama_params.h"
#include "src/kernels/includes/silu_and_mul.h"
#include "src/utils/macro.h"

template<typename T>
class LlamaFFNLayer {
private:
    // Shared params across all LLMs
    const int head_num;
    const int head_size;
    const int inter_size;
    const int hidden_units;
    
    int count = -1; // Used to record layer index currently

    cudaStream_t stream;
    BaseAllocator *allocator;
    cublasWrapper *cublas_wrapper;

    // Buffers
    // [2, num tokens, intersize]
    TensorWrapper<T> *swiglu_input = nullptr;  // Gate proj and up proj output buf
    
    // [num tokens, intersize]
    TensorWrapper<T> *down_proj_input = nullptr;

public:
    LlamaFFNLayer(
        int head_num,
        int head_size,
        int inter_size,
        cudaStream_t stream,
        cublasWrapper *cublas_wrapper,
        BaseAllocator *allocator
    );

    void allocateMemoryForForward(LlamaAttentionDynamicParams *params);
    
    void allocateMemoryForForward(int batch_size);
    
    void freeBuf();

    void forward(
        TensorMap *inputs, 
        TensorMap *outputs, 
        LlamaFFNWeights<T> *weights, 
        LlamaAttentionDynamicParams *params
    );
};
