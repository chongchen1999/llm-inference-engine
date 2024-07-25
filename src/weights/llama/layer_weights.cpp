#include <random>
#include "src/weights/llama/layer_weights.h"
#include "src/utils/macro.h"

template<typename T>
LlamaLayerWeight<T>::LlamaLayerWeight(
    int head_num,
    int kv_head_num,
    int head_size,
    int inter_size,
    WeightType weight_type,
    bool attn_bias
) :
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    hidden_units(head_num * head_size),
    inter_size(inter_size),
    weight_type(weight_type),
    attn_bias(attn_bias) {
    // Initialize weights structure and cudaMalloc for weights
    CHECK(cudaMalloc(reinterpret_cast<void**>(&attn_norm_weight.gamma), sizeof(T) * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&ffn_norm_weight.gamma), sizeof(T) * hidden_units));
    
    self_attn_weight.qkv.type = weight_type;
    self_attn_weight.qkv.shape = {(head_num + 2 * kv_head_num) * head_size, hidden_units};
    CHECK(cudaMalloc(reinterpret_cast<void**>(&self_attn_weight.qkv.data), sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size));
    
    self_attn_weight.output.type = weight_type;
    self_attn_weight.output.shape = {hidden_units, hidden_units};
    CHECK(cudaMalloc(reinterpret_cast<void**>(&self_attn_weight.output.data), sizeof(T) * hidden_units * hidden_units));
    
    if (attn_bias) {
        CHECK(cudaMalloc(reinterpret_cast<void**>(&self_attn_weight.qkv.bias), sizeof(T) * (head_num + 2 * kv_head_num) * head_size));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&self_attn_weight.output.bias), sizeof(T) * hidden_units));
    }
    
    // Concatenate gate linear weight and up linear weight into one weight tensor for performance improvement
    ffn_weight.gateAndup.type = weight_type;
    ffn_weight.down.type = weight_type;
    ffn_weight.gateAndup.shape = {2 * inter_size, hidden_units};
    ffn_weight.down.shape = {hidden_units, inter_size};
    
    CHECK(cudaMalloc(reinterpret_cast<void**>(&ffn_weight.gateAndup.data), sizeof(T) * hidden_units * 2 * inter_size));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&ffn_weight.down.data), sizeof(T) * hidden_units * inter_size));
}

template<typename T>
void LlamaLayerWeight<T>::loadWeights(
    const std::string& weight_path,
    WeightType weight_type
) {
    loadWeightFromBin<T, float>::loadFromFileToDevice(
        attn_norm_weight.gamma,
        {hidden_units},
        weight_path + ".input_layernorm.weight.bin"
    );

    loadWeightFromBin<T, float>::loadFromFileToDevice(
        ffn_norm_weight.gamma,
        {hidden_units},
        weight_path + ".post_attention_layernorm.weight.bin"
    );

    loadWeightFromBin<T, float>::loadFromFileToDevice(
        self_attn_weight.qkv.data,
        {(head_num + 2 * kv_head_num) * head_size, hidden_units},
        weight_path + ".self_attn.qkv.weight.bin"
    );

    loadWeightFromBin<T, float>::loadFromFileToDevice(
        self_attn_weight.output.data,
        {hidden_units, hidden_units},
        weight_path + ".self_attn.o_proj.weight.bin"
    );

    loadWeightFromBin<T, float>::loadFromFileToDevice(
        ffn_weight.gateAndup.data,
        {2 * inter_size, hidden_units},
        weight_path + ".mlp.gate_up_proj.weight.bin"
    );

    loadWeightFromBin<T, float>::loadFromFileToDevice(
        ffn_weight.down.data,
        {hidden_units, inter_size},
        weight_path + ".mlp.down_proj.weight.bin"
    );

    if (attn_bias) {
        loadWeightFromBin<T, float>::loadFromFileToDevice(
            self_attn_weight.qkv.bias,
            {(head_num + 2 * kv_head_num) * head_size},
            weight_path + ".attention.wqkv.bias.bin"
        );

        loadWeightFromBin<T, float>::loadFromFileToDevice(
            self_attn_weight.output.bias,
            {head_num * head_size},
            weight_path + ".attention.wo.bias.bin"
        );
    } else {
        self_attn_weight.qkv.bias = nullptr;
        self_attn_weight.output.bias = nullptr;
        ffn_weight.down.bias = nullptr;
    }
}

template<typename T>
void LlamaLayerWeight<T>::loadWeights() {
    T* d_dummy_attn_norm_weight;
    T* d_dummy_ffn_norm_weight;
    T* d_dummy_qkv_weights;
    T* d_dummy_output_weights;
    T* d_dummy_output_bias;
    T* d_dummy_ffn_down;
    T* d_dummy_ffn_down_bias;
    T* d_dummy_ffn_gate_up;

    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dummy_attn_norm_weight), sizeof(T) * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dummy_ffn_norm_weight), sizeof(T) * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dummy_qkv_weights), sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dummy_output_weights), sizeof(T) * hidden_units * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dummy_output_bias), sizeof(T) * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dummy_ffn_down), sizeof(T) * hidden_units * inter_size));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dummy_ffn_down_bias), sizeof(T) * hidden_units));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dummy_ffn_gate_up), sizeof(T) * hidden_units * 2 * inter_size));

    // Allocate memory
    T* h_dummy_attn_norm_weight = static_cast<T*>(malloc(sizeof(T) * hidden_units));
    T* h_dummy_ffn_norm_weight = static_cast<T*>(malloc(sizeof(T) * hidden_units));
    T* h_dummy_qkv_weights = static_cast<T*>(malloc(sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size));
    T* h_dummy_output_weights = static_cast<T*>(malloc(sizeof(T) * hidden_units * hidden_units));
    T* h_dummy_output_bias = static_cast<T*>(malloc(sizeof(T) * hidden_units));
    T* h_dummy_ffn_down = static_cast<T*>(malloc(sizeof(T) * hidden_units * inter_size));
    T* h_dummy_ffn_down_bias = static_cast<T*>(malloc(sizeof(T) * hidden_units));
    T* h_dummy_ffn_gate_up = static_cast<T*>(malloc(sizeof(T) * hidden_units * 2 * inter_size));

    // Define the lambda for filling an array with random values
    auto fillArrayWithRandom = [](T* array, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            array[i] = static_cast<T>(rand() % 10000 / 100000.0f);
        }
    };

    // Initialize arrays with random values using the lambda
    fillArrayWithRandom(h_dummy_attn_norm_weight, hidden_units);
    fillArrayWithRandom(h_dummy_ffn_norm_weight, hidden_units);
    fillArrayWithRandom(h_dummy_output_bias, hidden_units);
    fillArrayWithRandom(h_dummy_ffn_down_bias, hidden_units);
    fillArrayWithRandom(h_dummy_ffn_down, hidden_units * inter_size);
    fillArrayWithRandom(h_dummy_ffn_gate_up, hidden_units * 2 * inter_size);
    fillArrayWithRandom(h_dummy_output_weights, hidden_units * hidden_units);
    fillArrayWithRandom(h_dummy_qkv_weights, hidden_units * (head_num + 2 * kv_head_num) * head_size);

    CHECK(cudaMemcpy(d_dummy_attn_norm_weight, h_dummy_attn_norm_weight, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_norm_weight, h_dummy_ffn_norm_weight, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_qkv_weights, h_dummy_qkv_weights, sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_weights, h_dummy_output_weights, sizeof(T) * hidden_units * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_bias, h_dummy_output_bias, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down, h_dummy_ffn_down, sizeof(T) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down_bias, h_dummy_ffn_down_bias, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_gate_up, h_dummy_ffn_gate_up, sizeof(T) * hidden_units * 2 * inter_size, cudaMemcpyHostToDevice));

    // Before kernel launch, the ptr is always void*, when launching kernel, ptr type will be cast to float* or T*
    attn_norm_weight.gamma = d_dummy_attn_norm_weight;
    ffn_norm_weight.gamma = d_dummy_ffn_norm_weight;
    self_attn_weight.qkv.data = d_dummy_qkv_weights;
    self_attn_weight.qkv.bias = nullptr;
    self_attn_weight.output.data = d_dummy_output_weights;
    self_attn_weight.output.bias = d_dummy_output_bias;
    ffn_weight.gateAndup.data = d_dummy_ffn_gate_up;
    ffn_weight.down.data = d_dummy_ffn_down;
    ffn_weight.down.bias = d_dummy_ffn_down_bias;
}

template<typename T>
void freeWeights(BaseWeight<T>* weights) {
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
    cudaFree(attn_norm_weight.gamma);
    cudaFree(ffn_norm_weight.gamma);
    
    // Free other weights, including data and bias
    freeWeights(&self_attn_weight.qkv);
    freeWeights(&self_attn_weight.output);
    freeWeights(&ffn_weight.gateAndup);
    freeWeights(&ffn_weight.down);
}

// Template instantiation required in linking time
template class LlamaLayerWeight<float>;
template class LlamaLayerWeight<half>;
