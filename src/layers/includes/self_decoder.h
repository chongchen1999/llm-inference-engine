#pragma once

#include "../../kernels/includes/decoder_self_attention.cuh"
#include "../../kernels/includes/add_residual_and_rmsnorm.cuh"
#include "../../kernels/includes/rmsnorm.cuh"
#include "../../kernels/includes/add_residual.cuh"
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
    CublasWrapper *cublas_wrapper;
    BaseAllocator *allocator;

    TensorWrapper<T> *decoder_residual;
    LlamaSelfAttentionLayer<T> *self_attention;
    LlamaFFNLayer<T> *ffn;
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
        data_type(getTensorType<float>()),
        stream(stream),
        cublas_wrapper(cublas_wrapper),
        allocator(allocator) {
            self_attention = new LlamaSelfAttentionLayer<T>(
                head_num,
                kv_head_num,
                head_size,
                const_cast<LlamaAttentionStaticParams *>(&attn_params),
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

    void allocateMemory(LlamaAttentionDynamicParams *dynamic_params);
    void freeBuf();
    void forward(
        TensorMap *input_tensors,
        std::vector<LlamaLayerWeight<T> *> *layer_weights,
        TensorMap *output_tensors,
        LlamaAttentionDynamicParams *dynamic_params
    );
};
