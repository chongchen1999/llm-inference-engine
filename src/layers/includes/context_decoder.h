#pragma once

#include "src/kernels/includes/build_causal_mask.h"
#include "src/kernels/includes/cal_padding_offset.h"
#include "src/kernels/includes/fused_add_residual_and_rmsnorm.h"
#include "src/kernels/includes/add_residual.h"
#include "src/kernels/includes/rmsnorm.h"
#include "src/layers/includes/context_attention.h"
#include "src/layers/includes/ffn.h"
#include "src/weights/llama/llama_weights.h"
#include "src/utils/tensor.h"

// Layer weights are ready at model_utils.h
template <typename T>
class LlamaContextDecoder {
private:
    int head_num;
    int kv_head_num;
    int head_size;
    int intermediate_size;
    int num_layer;
    int hidden_units;
    float rmsnorm_eps;
    
    TensorWrapper<T> *attention_mask;
    TensorWrapper<int> *padding_offset;
    TensorWrapper<int> *cum_seqlens;
    TensorWrapper<T> *decoder_residual;

    cudaStream_t stream;
    cublasWrapper *cublas_wrapper;
    BaseAllocator *allocator;

    LlamaContextAttentionLayer<T> *context_attention;
    LlamaFFNLayer<T> *ffn;
    DataType data_type;

public:
    LlamaContextDecoder(
        int head_num,
        int kv_head_num,
        int head_size,
        int intermediate_size,
        int num_layer,
        const LlamaAttentionStaticParams &attn_params,
        float rmsnorm_eps,
        cudaStream_t stream,
        cublasWrapper *cublas_wrapper,
        BaseAllocator *allocator
    ) : 
        head_num(head_num), 
        kv_head_num(kv_head_num), 
        head_size(head_size), 
        intermediate_size(intermediate_size), 
        num_layer(num_layer), 
        hidden_units(head_num * head_size), 
        rmsnorm_eps(rmsnorm_eps), 
        data_type(getTensorType<T>()), 
        stream(stream), 
        cublas_wrapper(cublas_wrapper), 
        allocator(allocator) {
            context_attention = new LlamaContextAttentionLayer<T>(
                head_num,
                kv_head_num,
                head_size,
                attn_params,
                stream,
                cublas_wrapper,
                allocator
            );

            ffn = new LlamaFFNLayer<T>(
                head_num,
                head_size,
                intermediate_size,
                stream,
                cublas_wrapper,
                allocator
            );
        }

    void allocateMemory(LlamaAttentionDynamicParams *dyn_params);
    void freeBuf();
    
    void forward(
        TensorMap *input_tensors,
        const std::vector<LlamaLayerWeight<T> *> *layer_weights,
        TensorMap *output_tensors,
        LlamaAttentionDynamicParams *dyn_params
    );
};
