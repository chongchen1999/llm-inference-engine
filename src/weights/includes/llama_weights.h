#pragma once

#include <string>
#include <vector>
#include "weight.h"
#include "base_weights.h"
#include "embedding_weights.h"
#include "layer_weights.h"

template<typename T>
class LlamaWeight : public Weight {
private:
    int hidden_units;
    int intermediate_size;
    int vocab_size;
    int vocab_size_padded;
    int num_layer;
    WeightType weight_type;

public:
    std::vector<std::unique_ptr<LlamaLayerWeight<T>>> llama_layer_weight;
    LayerNormWeight<T> out_rmsnorm_weight;
    EmbeddingWeight<T> post_decoder_embedding_weight;
    EmbeddingWeight<T> pre_decoder_embedding_weight;

    LlamaWeight() = default;

    LlamaWeight(
        int head_num,
        int kv_head_num,
        int head_size,
        int intermediate_size,
        int vocab_size,
        int num_layer,
        bool attention_bias,
        WeightType weight_type
    );

    ~LlamaWeight();

    void loadWeightsFromFile(const std::string &weight_path);
    void loadWeightsFromDummy();
};
