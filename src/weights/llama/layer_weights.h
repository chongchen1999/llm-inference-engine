#pragma once

#include "src/weights/llama/norm_weights.h"
#include "src/weights/llama/attention_weights.h"
#include "src/weights/llama/ffn_weights.h"
#include "src/utils/weight_utils.h"

template<typename T>
class LlamaLayerWeight {
private:
    int head_num;
    int kv_head_num;
    int head_size;
    int hidden_units;
    int inter_size;
    WeightType weight_type;
    int bit_size;
    bool attn_bias;

public:
    // Deleted default constructor
    LlamaLayerWeight() = delete;

    // Parameterized constructor
    LlamaLayerWeight(
        int head_num,
        int kv_head_num,
        int head_size,
        int inter_size,
        WeightType weight_type,
        bool attn_bias
    );

    // Destructor
    ~LlamaLayerWeight();

    // Load weights from the specified path with given weight type
    void loadWeights(const std::string *weight_path, WeightType weight_type);

    // Load weights with default parameters
    void loadWeights();

    LayerNormWeight<T> attn_norm_weight;
    LayerNormWeight<T> ffn_norm_weight;
    LlamaAttentionWeights<T> self_attn_weight;
    LlamaFFNWeights<T> ffn_weight;
};
