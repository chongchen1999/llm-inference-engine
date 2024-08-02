#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include "includes/llama_weights.h"

template<typename T>
LlamaWeight<T>::LlamaWeight(
    int head_num,
    int kv_head_num,
    int head_size,
    int intermediate_size,
    int vocab_size,
    int num_layer,
    bool attention_bias,
    WeightType weight_type
) :
    hidden_units(head_num * head_size),
    intermediate_size(intermediate_size),
    vocab_size(vocab_size),
    vocab_size_padded(vocab_size),
    num_layer(num_layer),
    weight_type(weight_type) {

    llama_layer_weight.reserve(num_layer);
    for (int l = 0; l < num_layer; ++l) {
        llama_layer_weight.push_back(
            std::make_unique<LlamaLayerWeight<T>>(
                head_num,
                kv_head_num,
                head_size,
                intermediate_size,
                weight_type,
                attention_bias
            )
        );
    }

    GPUMalloc(&out_rmsnorm_weight.gamma, hidden_units);
    GPUMalloc(&post_decoder_embedding_weight.data, vocab_size * hidden_units);
    GPUMalloc(&pre_decoder_embedding_weight.data, vocab_size * hidden_units);

    pre_decoder_embedding_weight.shape = { vocab_size, hidden_units };
    post_decoder_embedding_weight.shape = { vocab_size, hidden_units };
    pre_decoder_embedding_weight.type = weight_type;
    post_decoder_embedding_weight.type = weight_type;
}

template<typename T>
void LlamaWeight<T>::loadWeightsFromFile(const std::string &weight_path) {
    loadWeightFromBin<T, float>::loadFromFileToDevice(
        out_rmsnorm_weight.gamma,
        { hidden_units },
        weight_path + "model.norm.weight.bin"
    );

    loadWeightFromBin<T, float>::loadFromFileToDevice(
        post_decoder_embedding_weight.data,
        { vocab_size, hidden_units },
        weight_path + "lm_head.weight.bin"
    );

    loadWeightFromBin<T, float>::loadFromFileToDevice(
        pre_decoder_embedding_weight.data,
        { vocab_size, hidden_units },
        weight_path + "model.embed_tokens.weight.bin"
    );

    for (int layer = 0; layer < num_layer; ++layer) {
        llama_layer_weight[layer]->loadWeightsFromFile(
            weight_path + "model.layers." + std::to_string(layer),
            weight_type
        );
    }
}

template<typename T>
void LlamaWeight<T>::loadWeightsFromDummy() {
    T *d_dummy_out_rmsnorm_weight_gamma;
    T *d_dummy_post_decoder_embedding_weight;
    T *d_dummy_pre_decoder_embedding_weight;

    GPUMalloc(&d_dummy_out_rmsnorm_weight_gamma, sizeof(T) * hidden_units);
    GPUMalloc(&d_dummy_post_decoder_embedding_weight, sizeof(T) * hidden_units * vocab_size);
    GPUMalloc(&d_dummy_pre_decoder_embedding_weight, sizeof(T) * hidden_units * vocab_size);

    auto h_dummy_out_rmsnorm_weight_gamma = std::make_unique<T[]>(hidden_units);
    auto h_dummy_post_decoder_embedding_weight = std::make_unique<T[]>(hidden_units * vocab_size);
    auto h_dummy_pre_decoder_embedding_weight = std::make_unique<T[]>(hidden_units * vocab_size);

    auto fillArray = [](T *array, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            array[i] = 1.0f;
        }
    };
    fillArray(h_dummy_out_rmsnorm_weight_gamma.get(), hidden_units);
    fillArray(h_dummy_post_decoder_embedding_weight.get(), hidden_units * vocab_size);
    fillArray(h_dummy_pre_decoder_embedding_weight.get(), hidden_units * vocab_size);

    cudaMemcpy(
        d_dummy_out_rmsnorm_weight_gamma,
        h_dummy_out_rmsnorm_weight_gamma.get(),
        sizeof(T) * hidden_units,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_dummy_post_decoder_embedding_weight,
        h_dummy_post_decoder_embedding_weight.get(),
        sizeof(T) * hidden_units * vocab_size,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_dummy_pre_decoder_embedding_weight,
        h_dummy_pre_decoder_embedding_weight.get(),
        sizeof(T) * hidden_units * vocab_size,
        cudaMemcpyHostToDevice
    );

    out_rmsnorm_weight.gamma = d_dummy_out_rmsnorm_weight_gamma;
    post_decoder_embedding_weight.data = d_dummy_post_decoder_embedding_weight;
    pre_decoder_embedding_weight.data = d_dummy_pre_decoder_embedding_weight;

    for (int layer = 0; layer < num_layer; ++layer) {
        llama_layer_weight[layer]->loadWeightsFromFile();
    }
}

template<typename T>
LlamaWeight<T>::~LlamaWeight() {
    cudaFree(pre_decoder_embedding_weight.data);
    cudaFree(out_rmsnorm_weight.gamma);
    cudaFree(post_decoder_embedding_weight.data);

    for (const auto &layer : llama_layer_weight) {
        // No need to explicitly delete since using smart pointers
    }
}

// Template instantiation required at linking time
template struct LlamaWeight<float>;
template struct LlamaWeight<half>;
