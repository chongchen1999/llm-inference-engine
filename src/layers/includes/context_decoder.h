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

    TensorWrapper<T> *attention_mask = nullptr;
    TensorWrapper<int> *padding_offset = nullptr;
    TensorWrapper<int> *cum_seqlens = nullptr;
    TensorWrapper<T> *decoder_residual = nullptr;

    cudaStream_t stream;
    CublasWrapper *cublas_wrapper;
    BaseAllocator *allocator;

    LlamaContextAttentionLayer<T> *context_attention = nullptr;
    LlamaFFNLayer<T> *ffn = nullptr;
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
        CublasWrapper *const &cublas_wrapper,
        BaseAllocator *const &allocator
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
                attention_static_params,
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

    void allocateMemory(LlamaAttentionDynamicParams *attention_dynamic_params);
    void freeBuf();
    void forward(
        TensorMap *input_tensors,
        std::vector<LlamaLayerWeight<T> *> *layer_weights,
        TensorMap *output_tensors,
        LlamaAttentionDynamicParams *attention_dynamic_params
    );
};
