#pragma once

#include <memory>
#include "../../kernels/includes/decoder_self_attention.h"
#include "../../kernels/includes/add_residual_and_rmsnorm.h"
#include "../../kernels/includes/rmsnorm.h"
#include "../../kernels/includes/add_residual.h"
#include "../../layers/includes/self_attention.h"
#include "../../layers/includes/ffn.h"
#include "../../weights/includes/llama_weights.h"
#include "../../utils/tensor.h"

// Layer weights are prepared in model_utils.h
// via loadweights in onellm.cpp, outside of the decoder
template <typename T>
class LlamaSelfDecoder {
private:
    int head_num;
    int kv_head_num;
    int head_size;
    int intermediate_size;
    int num_layer;
    int hidden_units;
    float rmsnorm_eps;

    cudaStream_t stream;
    std::shared_ptr<CublasWrapper> cublas_wrapper;
    std::shared_ptr<BaseAllocator> allocator;

    std::unique_ptr<TensorWrapper<T>> decoder_residual;

    std::unique_ptr<LlamaSelfAttentionLayer<T>> self_attention;
    std::unique_ptr<LlamaFFNLayer<T>> ffn;
    DataType data_type;

public:
    LlamaSelfDecoder(
        const int &head_num,
        const int &kv_head_num,
        const int &head_size,
        const int &intermediate_size,
        const int &num_layer,
        const LlamaAttentionStaticParams &attn_params,
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
        data_type(getTensorType<float>()),
        stream(stream),
        cublas_wrapper(cublas_wrapper),
        allocator(allocator) {
            self_attention = std::make_unique<LlamaSelfAttentionLayer<T>>(
                head_num,
                kv_head_num,
                head_size,
                const_cast<LlamaAttentionStaticParams *>(&attn_params),
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

    void allocateMemory(LlamaAttentionDynamicParams *dynamic_params);

    void freeBuf();

    void forward(
        TensorMap *const input_tensors,
        std::vector<std::unique_ptr<LlamaLayerWeight<T>>> *const layerWeights,
        TensorMap *const output_tensors,
        LlamaAttentionDynamicParams *const dynamic_params
    );
};
