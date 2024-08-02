#include <random>
#include <memory>
#include "includes/layer_weights.h"
#include "../utils/macro.h"

template<typename T>
LlamaLayerWeight<T>::LlamaLayerWeight(
    int head_num,
    int kv_head_num,
    int head_size,
    int intermediate_size,
    WeightType weight_type,
    bool attention_bias
) :
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    hidden_units(head_num * head_size),
    intermediate_size(intermediate_size),
    weight_type(weight_type),
    attention_bias(attention_bias) {
    
    // Initialize weights structure and cudaMalloc for weights
    CHECK(cudaMalloc(reinterpret_cast<void **>(&attention_norm_weight.gamma), sizeof(T) * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&ffn_norm_weight.gamma), sizeof(T) * hidden_units));
    
    self_attention_weight.qkv.type = weight_type;
    self_attention_weight.qkv.shape = { (head_num + 2 * kv_head_num) * head_size, hidden_units };
    CHECK(cudaMalloc(reinterpret_cast<void **>(&self_attention_weight.qkv.data), sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size));
    
    self_attention_weight.output.type = weight_type;
    self_attention_weight.output.shape = { hidden_units, hidden_units };
    CHECK(cudaMalloc(reinterpret_cast<void **>(&self_attention_weight.output.data), sizeof(T) * hidden_units * hidden_units));
    
    if (attention_bias) {
        CHECK(cudaMalloc(reinterpret_cast<void **>(&self_attention_weight.qkv.bias), sizeof(T) * (head_num + 2 * kv_head_num) * head_size));
        CHECK(cudaMalloc(reinterpret_cast<void **>(&self_attention_weight.output.bias), sizeof(T) * hidden_units));
    }
    
    // Concatenate gate linear weight and up linear weight into one weight tensor for performance improvement
    ffn_weight.gate_and_up.type = weight_type;
    ffn_weight.down.type = weight_type;
    ffn_weight.gate_and_up.shape = { 2 * intermediate_size, hidden_units };
    ffn_weight.down.shape = { hidden_units, intermediate_size };
    
    CHECK(cudaMalloc(reinterpret_cast<void **>(&ffn_weight.gate_and_up.data), sizeof(T) * hidden_units * 2 * intermediate_size));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&ffn_weight.down.data), sizeof(T) * hidden_units * intermediate_size));
}

template<typename T>
void LlamaLayerWeight<T>::loadWeightsFromFile(
    const std::string &weight_path,
    WeightType weight_type
) {
    auto loadWeight = [&](const std::string &suffix, const std::vector<int> &shape, auto *ptr) {
        loadWeightFromBin<T, float>::loadFromFileToDevice(
            ptr,
            shape,
            weight_path + suffix
        );
    };

    // Load weights
    loadWeight(".input_layernorm.weight.bin", { hidden_units }, attention_norm_weight.gamma);
    loadWeight(".post_attention_layernorm.weight.bin", { hidden_units }, ffn_norm_weight.gamma);
    loadWeight(".self_attn.qkv.weight.bin", { (head_num + 2 * kv_head_num) * head_size, hidden_units }, self_attention_weight.qkv.data);
    loadWeight(".self_attn.o_proj.weight.bin", { hidden_units, hidden_units }, self_attention_weight.output.data);
    loadWeight(".mlp.gate_up_proj.weight.bin", { 2 * intermediate_size, hidden_units }, ffn_weight.gate_and_up.data);
    loadWeight(".mlp.down_proj.weight.bin", { hidden_units, intermediate_size }, ffn_weight.down.data);

    // Load biases if applicable
    if (attention_bias) {
        loadWeight(".attention.wqkv.bias.bin", { (head_num + 2 * kv_head_num) * head_size }, self_attention_weight.qkv.bias);
        loadWeight(".attention.wo.bias.bin", { hidden_units }, self_attention_weight.output.bias);
    } else {
        self_attention_weight.qkv.bias = nullptr;
        self_attention_weight.output.bias = nullptr;
        ffn_weight.down.bias = nullptr;
    }
}

template<typename T>
void LlamaLayerWeight<T>::loadWeightsFromFile() {
    // Allocate device memory
    T *d_dummy_attn_norm_weight = nullptr;
    T *d_dummy_ffn_norm_weight = nullptr;
    T *d_dummy_qkv_weights = nullptr;
    T *d_dummy_output_weights = nullptr;
    T *d_dummy_output_bias = nullptr;
    T *d_dummy_ffn_down = nullptr;
    T *d_dummy_ffn_down_bias = nullptr;
    T *d_dummy_ffn_gate_up = nullptr;

    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dummy_attn_norm_weight), sizeof(T) * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dummy_ffn_norm_weight), sizeof(T) * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dummy_qkv_weights), sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dummy_output_weights), sizeof(T) * hidden_units * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dummy_output_bias), sizeof(T) * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dummy_ffn_down), sizeof(T) * hidden_units * intermediate_size));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dummy_ffn_down_bias), sizeof(T) * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dummy_ffn_gate_up), sizeof(T) * hidden_units * 2 * intermediate_size));

    // Allocate host memory
    auto h_dummy_attn_norm_weight = std::make_unique<T[]>(hidden_units);
    auto h_dummy_ffn_norm_weight = std::make_unique<T[]>(hidden_units);
    auto h_dummy_qkv_weights = std::make_unique<T[]>(hidden_units * (head_num + 2 * kv_head_num) * head_size);
    auto h_dummy_output_weights = std::make_unique<T[]>(hidden_units * hidden_units);
    auto h_dummy_output_bias = std::make_unique<T[]>(hidden_units);
    auto h_dummy_ffn_down = std::make_unique<T[]>(hidden_units * intermediate_size);
    auto h_dummy_ffn_down_bias = std::make_unique<T[]>(hidden_units);
    auto h_dummy_ffn_gate_up = std::make_unique<T[]>(hidden_units * 2 * intermediate_size);

    // Define the lambda for filling an array with random values
    auto fillArrayWithRandom = [](T *array, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            array[i] = static_cast<T>(rand() % 10000 / 100000.0f);
        }
    };

    // Initialize arrays with random values using the lambda
    fillArrayWithRandom(h_dummy_attn_norm_weight.get(), hidden_units);
    fillArrayWithRandom(h_dummy_ffn_norm_weight.get(), hidden_units);
    fillArrayWithRandom(h_dummy_output_bias.get(), hidden_units);
    fillArrayWithRandom(h_dummy_ffn_down_bias.get(), hidden_units);
    fillArrayWithRandom(h_dummy_ffn_down.get(), hidden_units * intermediate_size);
    fillArrayWithRandom(h_dummy_ffn_gate_up.get(), hidden_units * 2 * intermediate_size);
    fillArrayWithRandom(h_dummy_output_weights.get(), hidden_units * hidden_units);
    fillArrayWithRandom(h_dummy_qkv_weights.get(), hidden_units * (head_num + 2 * kv_head_num) * head_size);

    CHECK(cudaMemcpy(d_dummy_attn_norm_weight, h_dummy_attn_norm_weight.get(), sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_norm_weight, h_dummy_ffn_norm_weight.get(), sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_qkv_weights, h_dummy_qkv_weights.get(), sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_weights, h_dummy_output_weights.get(), sizeof(T) * hidden_units * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_bias, h_dummy_output_bias.get(), sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down, h_dummy_ffn_down.get(), sizeof(T) * hidden_units * intermediate_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down_bias, h_dummy_ffn_down_bias.get(), sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_gate_up, h_dummy_ffn_gate_up.get(), sizeof(T) * hidden_units * 2 * intermediate_size, cudaMemcpyHostToDevice));

    // Assign pointers
    attention_norm_weight.gamma = d_dummy_attn_norm_weight;
    ffn_norm_weight.gamma = d_dummy_ffn_norm_weight;
    self_attention_weight.qkv.data = d_dummy_qkv_weights;
    self_attention_weight.qkv.bias = nullptr;
    self_attention_weight.output.data = d_dummy_output_weights;
    self_attention_weight.output.bias = d_dummy_output_bias;
    ffn_weight.gate_and_up.data = d_dummy_ffn_gate_up;
    ffn_weight.down.data = d_dummy_ffn_down;
    ffn_weight.down.bias = d_dummy_ffn_down_bias;
}

template<typename T>
void LlamaLayerWeight<T>::freeWeights(BaseWeight<T> *weights) {
    cudaFree(weights->data);
    if (weights->bias != nullptr) {
        cudaFree(weights->bias);
    }

    weights->data = nullptr;
    weights->bias = nullptr;
}

template<typename T>
LlamaLayerWeight<T>::~LlamaLayerWeight() {
    // Free norm weights ptr
    cudaFree(attention_norm_weight.gamma);
    cudaFree(ffn_norm_weight.gamma);
    
    // Free other weights, including data and bias
    freeWeights(&self_attention_weight.qkv);
    freeWeights(&self_attention_weight.output);
    freeWeights(&ffn_weight.gate_and_up);
    freeWeights(&ffn_weight.down);
}

// Template instantiation required at linking time
template class LlamaLayerWeight<float>;
template class LlamaLayerWeight<half>;
