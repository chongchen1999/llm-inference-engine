#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../src/layers/includes/self_decoder.h"
#include "../../src/utils/macro.h"
#include "../../src/memory/memory_deleter.cuh"

// Function to allocate and initialize memory on the device
template<typename T>
T *mallocForDevice(size_t size, T init_value) {
    T *host_ptr = new T[size];
    std::fill(host_ptr, host_ptr + size, init_value);

    T *device_ptr;
    cudaMalloc(reinterpret_cast<void **>(&device_ptr), sizeof(T) * size);
    cudaMemcpy(device_ptr, host_ptr, sizeof(T) * size, cudaMemcpyHostToDevice);
    
    delete[] host_ptr;
    return device_ptr;
}

int main() {
    int h_step = 3;
    int head_num = 4;
    int kv_head_num = 2;
    int head_size = 8;
    int intermediate_size = 12;
    int num_layers = 32;
    int max_seq_len = 12;
    int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    int q_hidden_units = head_num * head_size;
    float rmsnorm_eps = 1e-6;
    int layer_id = 0;

    LlamaAttentionStaticParams attn_static_params {
        .rotary_embedding_dim = 128,
        .rotary_embedding_base = 10000,
        .max_position_embeddings = 2048,
        .use_dynamic_ntk = false
    };

    LlamaAttentionDynamicParams attn_dyn_params {
        .batch_size = 2
    };

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;

    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    auto cublas_wrapper = new CublasWrapper(cublas_handle, cublaslt_handle);
    auto allocator = new CudaAllocator();

    // Allocate and initialize memory
    float *d_decoder_input = mallocForDevice<float>(q_hidden_units * attn_dyn_params.batch_size, 0.0f);
    float *d_decoder_output = mallocForDevice<float>(q_hidden_units * attn_dyn_params.batch_size, 0.0f);
    float *d_all_k_cache = mallocForDevice<float>(num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, 1.0f);
    float *d_all_v_cache = mallocForDevice<float>(num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, 1.0f);
    bool *d_finished = mallocForDevice<bool>(attn_dyn_params.batch_size, false);

    float *d_output_norm_weight = mallocForDevice<float>(q_hidden_units, 2.0f);
    float *d_attn_norm_weight = mallocForDevice<float>(q_hidden_units, 1.0f);
    float *d_ffn_norm_weight = mallocForDevice<float>(q_hidden_units, 1.0f);
    float *d_qkv_weights = mallocForDevice<float>(hidden_units * q_hidden_units, 1.0f);
    float *d_qkv_bias = mallocForDevice<float>(hidden_units, 2.0f);
    float *d_output_weights = mallocForDevice<float>(q_hidden_units * q_hidden_units, 1.0f);
    float *d_out_bias = mallocForDevice<float>(head_num * head_size, 2.0f);

    float *d_ffn_gate = mallocForDevice<float>(hidden_units * 2 * intermediate_size, 2.0f);
    float *d_ffn_up = mallocForDevice<float>(hidden_units * intermediate_size, 2.0f);
    float *d_ffn_down_bias = mallocForDevice<float>(hidden_units, 0.0f);

    // Tensor Wrappers
    DataType type = getTensorType<float>();
    DataType type_int = getTensorType<int>();
    DataType type_bool = getTensorType<bool>();

    auto layer_weights = new std::vector<LlamaLayerWeight<float> *>;
    WeightType wtype = getWeightType<float>();
    layer_weights->reserve(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        auto weight = new LlamaLayerWeight<float>(head_num, kv_head_num, head_size, intermediate_size, wtype, true);
        weight->loadWeightsFromFile();
        layer_weights->push_back(std::move(weight));
    }

    TensorWrapper<float> *decoder_input = new TensorWrapper<float>(
        Device::GPU, type, 
        std::vector<int>{attn_dyn_params.batch_size, q_hidden_units}, d_decoder_input
    );

    TensorWrapper<int> *step = new TensorWrapper<int>(
        Device::CPU, type_int, 
        std::vector<int>{1}, &h_step
    );

    TensorWrapper<bool> *finished = new TensorWrapper<bool>(
        Device::GPU, type_bool, 
        std::vector<int>{attn_dyn_params.batch_size}, d_finished
    );

    TensorWrapper<int> *layer = new TensorWrapper<int>(
        Device::CPU, type_int, 
        std::vector<int>{1}, &layer_id
    );

    TensorWrapper<float> *output_norm_weight = new TensorWrapper<float>(
        Device::GPU, type, 
        std::vector<int>{q_hidden_units}, d_output_norm_weight
    );

    TensorWrapper<float> *decoder_output = new TensorWrapper<float>(
        Device::GPU, type, 
        std::vector<int>{attn_dyn_params.batch_size, q_hidden_units}, d_decoder_output
    );

    TensorWrapper<float> *key_cache = new TensorWrapper<float>(
        Device::GPU, type, 
        std::vector<int>{num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_k_cache
    );

    TensorWrapper<float> *value_cache = new TensorWrapper<float>(
        Device::GPU, type, 
        std::vector<int>{num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_v_cache
    );

    LLM_CHECK_WITH_INFO(
        decoder_input->data != nullptr, 
        "The data ptr of tensor inserted into TensorMap is nullptr!"
    );

    LLM_CHECK_WITH_INFO(
        step->data != nullptr, 
        "The data ptr of tensor inserted into TensorMap is nullptr!"
    );

    LLM_CHECK_WITH_INFO(
        finished->data != nullptr, 
        "The data ptr of tensor inserted into TensorMap is nullptr!"
    );

    LLM_CHECK_WITH_INFO(
        layer->data != nullptr, 
        "The data ptr of tensor inserted into TensorMap is nullptr!"
    );

    TensorMap decoder_inputs {
        {"decoder_input", decoder_input},
        {"step", step},
        {"finished", finished},
        {"layer_id", layer},
        {"output_norm_weight", output_norm_weight}
    };

    TensorMap decoder_outputs {
        {"decoder_output", decoder_output},
        {"all_k_cache", key_cache},
        {"all_v_cache", value_cache}
    };

    auto self_decoder = new LlamaSelfDecoder<float>(
        head_num, kv_head_num, head_size, intermediate_size, num_layers,
        attn_static_params, 
        rmsnorm_eps, 
        stream, 
        cublas_wrapper, 
        allocator
    );

    self_decoder->forward(
        &decoder_inputs, 
        layer_weights, 
        &decoder_outputs, 
        &attn_dyn_params
    );
    cudaDeviceSynchronize();

    // Deallocate device memory
    deallocate(d_decoder_input, "cudaMalloc");
    deallocate(d_decoder_output, "cudaMalloc");
    deallocate(d_all_k_cache, "cudaMalloc");
    deallocate(d_all_v_cache, "cudaMalloc");
    deallocate(d_finished, "cudaMalloc");
    deallocate(d_output_norm_weight, "cudaMalloc");
    deallocate(d_attn_norm_weight, "cudaMalloc");
    deallocate(d_ffn_norm_weight, "cudaMalloc");
    deallocate(d_qkv_weights, "cudaMalloc");
    deallocate(d_qkv_bias, "cudaMalloc");
    deallocate(d_output_weights, "cudaMalloc");
    deallocate(d_out_bias, "cudaMalloc");
    deallocate(d_ffn_gate, "cudaMalloc");
    deallocate(d_ffn_up, "cudaMalloc");
    deallocate(d_ffn_down_bias, "cudaMalloc");

    // Deallocate dynamically allocated objects
    deallocate(cublas_wrapper, "new");
    deallocate(allocator, "new");
    deallocate(decoder_input, "new");
    deallocate(step, "new");
    deallocate(finished, "new");
    deallocate(layer, "new");
    deallocate(output_norm_weight, "new");
    deallocate(decoder_output, "new");
    deallocate(key_cache, "new");
    deallocate(value_cache, "new");

    // Clean up layer_weights
    for (auto layer_weight : *layer_weights) {
        deallocate(layer_weight, "new");
    }
    deallocate(layer_weights, "new");

    return 0;
}
