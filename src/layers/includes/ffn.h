#pragma once

#include <memory>
#include "../../weights/includes/attention_weights.h"
#include "../../weights/includes/ffn_weights.h"
#include "../../memory/allocator/cuda_allocator.h"
#include "../../kernels/includes/linear.cuh"
#include "../../utils/tensor.h"
#include "../../kernels/includes/cublas_utils.cuh"
#include "../../models/llama/llama_params.h"
#include "../../kernels/includes/silu_and_mul.cuh"
#include "../../utils/macro.h"

template<typename T>
class LlamaFFNLayer {
private:
    int head_num;
    int head_size;
    int intermediate_size;
    int hidden_units;
    
    int count = -1; // Used to record layer index currently

    cudaStream_t stream;
    BaseAllocator *allocator;
    CublasWrapper *cublas_wrapper;

    // Buffers
    // [2, num tokens, intersize]
    TensorWrapper<T> *swiglu_input; //gate proj and up proj output buf
    
    // [num tokens, intersize]
    TensorWrapper<T> *down_proj_input; // Down proj output buf

public:
    LlamaFFNLayer(
        int head_num,
        int head_size,
        int intermediate_size,
        cudaStream_t stream,
        CublasWrapper *cublas_wrapper,
        BaseAllocator *allocator
    );

    void allocateMemory(LlamaAttentionDynamicParams *dynamic_params);
    void allocateMemory(const int &batch_size);
    void freeBuf();
    void forward(
        TensorMap *inputs, 
        TensorMap *outputs, 
        LlamaFFNWeights<T> *weights, 
        LlamaAttentionDynamicParams *dynamic_params
    );
};
