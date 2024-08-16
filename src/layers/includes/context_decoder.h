#pragma once

#include "../../kernels/includes/build_causal_mask.cuh"
#include "../../kernels/includes/cal_padding_offset.cuh"
#include "../../kernels/includes/add_residual_and_rmsnorm.cuh"
#include "../../kernels/includes/add_residual.cuh"
#include "../../kernels/includes/rmsnorm.cuh"
#include "../../layers/includes/context_attention.h"
#include "../../layers/includes/ffn.h"
#include "../../weights/includes/llama_weights.h"
#include "../../utils/tensor.h"
#include <memory>

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
    
    std::unique_ptr<TensorWrapper<T>> attention_mask;
    std::unique_ptr<TensorWrapper<int>> padding_offset;
    std::unique_ptr<TensorWrapper<int>> cum_seqlens;
    std::unique_ptr<TensorWrapper<T>> decoder_residual;

    cudaStream_t stream;
    std::shared_ptr<CublasWrapper> cublas_wrapper;
    std::shared_ptr<BaseAllocator> allocator;

    std::unique_ptr<LlamaContextAttentionLayer<T>> context_attention;
    std::unique_ptr<LlamaFFNLayer<T>> ffn;
    DataType data_type;

public:
    LlamaContextDecoder(
        const int &head_num,
        const int &kv_head_num,
        const int &head_size,
        const int &intermediate_size,
        const int &num_layer,
        LlamaAttentionStaticParams *const &attention_static_params,
        const float &rmsnorm_eps,
        const cudaStream_t &stream,
        const std::shared_ptr<CublasWrapper> &cublas_wrapper,
        const std::shared_ptr<BaseAllocator> &allocator
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
        
        context_attention = std::make_unique<LlamaContextAttentionLayer<T>>(
            head_num,
            kv_head_num,
            head_size,
            attention_static_params,
            stream,
            cublas_wrapper.get(),
            allocator.get()
        );

        ffn = std::make_unique<LlamaFFNLayer<T>>(
            head_num,
            head_size,
            intermediate_size,
            stream,
            cublas_wrapper.get(),
            allocator.get()
        );
    }

    void allocateMemory(LlamaAttentionDynamicParams *attention_dynamic_params);

    void freeBuf();

    void forward(
        TensorMap *input_tensors,
        std::vector<std::unique_ptr<LlamaLayerWeight<T>>> *const layer_weights,
        TensorMap *output_tensors,
        LlamaAttentionDynamicParams *attention_dynamic_params
    );
};
