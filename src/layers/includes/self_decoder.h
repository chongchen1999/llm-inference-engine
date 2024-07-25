#pragma once

#include "src/kernels/includes/fused_decoder_self_attention.h"
#include "src/kernels/includes/fused_add_residual_and_rmsnorm.h"
#include "src/kernels/includes/rmsnorm.h"
#include "src/kernels/includes/add_residual.h"
#include "src/layers/includes/masked_self_attention.h"
#include "src/layers/includes/ffn.h"
#include "src/weights/llama/llama_weights.h"
#include "src/utils/tensor.h"

// Layer weights are prepared in model_utils.h
// via loadweights in onellm.cpp, outside of the decoder
template <typename T>
class LlamaSelfDecoder {
private:
    int head_num;
    int kv_head_num;
    int head_size;
    int inter_size;
    int num_layer;
    int hidden_units;
    float rmsnorm_eps;

    cudaStream_t stream;
    cublasWrapper *cublas_wrapper;
    BaseAllocator *allocator;

    TensorWrapper<T> *decoder_residual;

    LlamaSelfAttentionLayer<T> *self_attention;
    LlamaFFNLayer<T> *ffn;
    DataType data_type;

public:
    LlamaSelfDecoder(
        int head_num,
        int kv_head_num,
        int head_size,
        int inter_size,
        int num_layer,
        const LLaMAAttentionStaticParams &attn_params,
        float rmsnorm_eps,
        cudaStream_t stream,
        cublasWrapper *cublas_wrapper,
        BaseAllocator *allocator
    ) : 
        head_num(head_num),
        kv_head_num(kv_head_num),
        head_size(head_size),
        inter_size(inter_size),
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
                attn_params,
                stream,
                cublas_wrapper,
                allocator
            );

            ffn = new LlamaFFNLayer<T>(
                head_num,
                head_size,
                inter_size,
                stream,
                cublas_wrapper,
                allocator
            );
        }

    void allocateMemoryForForward(
        LlamaAttentionDynamicParams *dyn_params
    );

    void freeBuf();

    void forward(
        TensorMap *input_tensors,
        const std::vector<LlamaLayerWeight<T> *> *layerWeights,
        TensorMap *output_tensors,
        LlamaAttentionDynamicParams *dyn_params
    );
};
