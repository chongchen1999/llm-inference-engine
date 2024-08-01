#pragma once

#include "norm_weights.h"
#include "attention_weights.h"
#include "ffn_weights.h"
#include "../../utils/weight_utils.h"

template<typename T>
class LlamaLayerWeight {
private:
    int head_num;
    int kv_head_num;
    int head_size;
    int hidden_units;
    int intermediate_size;
    WeightType weight_type;
    int bit_size;
    bool attention_bias;

public:
    LlamaLayerWeight() = delete;

    LlamaLayerWeight(
        int head_num,
        int kv_head_num,
        int head_size,
        int intermediate_size,
        WeightType weight_type,
        bool attention_bias
    );

    ~LlamaLayerWeight();

    void loadWeightsFromFile(const std::string &weight_path, WeightType weight_type);
    void loadWeightsFromFile();
    void freeWeights(BaseWeight<T> *weights);

    LayerNormWeight<T> attention_norm_weight;
    LayerNormWeight<T> ffn_norm_weight;
    LlamaAttentionWeights<T> self_attention_weight;
    LlamaFFNWeights<T> ffn_weight;
};
