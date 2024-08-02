#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../src/layers/includes/self_decoder.h"
#include "../../src/utils/macro.h"

// Function to allocate and initialize memory on the device
template<typename T>
T *mallocForDevice(size_t size, T init_value) {
    auto host_ptr = std::make_unique<T[]>(size);
    std::fill(host_ptr.get(), host_ptr.get() + size, init_value);

    T *device_ptr;
    cudaMalloc(reinterpret_cast<void **>(&device_ptr), sizeof(T) * size);
    cudaMemcpy(device_ptr, host_ptr.get(), sizeof(T) * size, cudaMemcpyHostToDevice);
    
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

    auto cublas_wrapper = std::make_shared<CublasWrapper>(cublas_handle, cublaslt_handle);
    auto allocator = std::make_shared<CudaAllocator>();

    // Allocate and initialize memory
    auto d_decoder_input = mallocForDevice<float>(q_hidden_units * attn_dyn_params.batch_size, 0.0f);
    auto d_decoder_output = mallocForDevice<float>(q_hidden_units * attn_dyn_params.batch_size, 0.0f);
    auto d_all_k_cache = mallocForDevice<float>(num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, 1.0f);
    auto d_all_v_cache = mallocForDevice<float>(num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, 1.0f);
    auto d_finished = mallocForDevice<bool>(attn_dyn_params.batch_size, false);

    auto d_output_norm_weight = mallocForDevice<float>(q_hidden_units, 2.0f);
    auto d_attn_norm_weight = mallocForDevice<float>(q_hidden_units, 1.0f);
    auto d_ffn_norm_weight = mallocForDevice<float>(q_hidden_units, 1.0f);
    auto d_qkv_weights = mallocForDevice<float>(hidden_units * q_hidden_units, 1.0f);
    auto d_qkv_bias = mallocForDevice<float>(hidden_units, 2.0f);
    auto d_output_weights = mallocForDevice<float>(q_hidden_units * q_hidden_units, 1.0f);
    auto d_out_bias = mallocForDevice<float>(head_num * head_size, 2.0f);

    auto d_ffn_gate = mallocForDevice<float>(hidden_units * 2 * intermediate_size, 2.0f);
    auto d_ffn_up = mallocForDevice<float>(hidden_units * intermediate_size, 2.0f);
    auto d_ffn_down_bias = mallocForDevice<float>(hidden_units, 0.0f);

    // Tensor Wrappers
    DataType type = getTensorType<float>();
    DataType type_int = getTensorType<int>();
    DataType type_bool = getTensorType<bool>();

    std::vector<std::unique_ptr<LlamaLayerWeight<float>>> layer_weights;
    WeightType wtype = getWeightType<float>();
    layer_weights.reserve(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        layer_weights.push_back(
            std::make_unique<LlamaLayerWeight<float>>(
                head_num, kv_head_num, head_size, intermediate_size, wtype, true
            )
        );
        layer_weights.back()->loadWeightsFromFile();
    }

    auto decoder_input = std::make_unique<TensorWrapper<float>>(
        Device::GPU, type, 
        std::vector<int>{attn_dyn_params.batch_size, q_hidden_units}, d_decoder_input
    );

    auto step = std::make_unique<TensorWrapper<int>>(
        Device::CPU, type_int, 
        std::vector<int>{1}, &h_step
    );

    auto finished = std::make_unique<TensorWrapper<bool>>(
        Device::GPU, type_bool, 
        std::vector<int>{attn_dyn_params.batch_size}, d_finished
    );

    auto layer = std::make_unique<TensorWrapper<int>>(
        Device::CPU, type_int, 
        std::vector<int>{1}, &layer_id
    );

    auto output_norm_weight = std::make_unique<TensorWrapper<float>>(
        Device::GPU, type, 
        std::vector<int>{q_hidden_units}, d_output_norm_weight
    );

    auto decoder_output = std::make_unique<TensorWrapper<float>>(
        Device::GPU, type, 
        std::vector<int>{attn_dyn_params.batch_size, q_hidden_units}, d_decoder_output
    );

    auto key_cache = std::make_unique<TensorWrapper<float>>(
        Device::GPU, type, 
        std::vector<int>{num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_k_cache
    );

    auto value_cache = std::make_unique<TensorWrapper<float>>(
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
        {"decoder_input", decoder_input.get()},
        {"step", step.get()},
        {"finished", finished.get()},
        {"layer_id", layer.get()},
        {"output_norm_weight", output_norm_weight.get()}
    };

    TensorMap decoder_outputs {
        {"decoder_output", decoder_output.get()},
        {"all_k_cache", key_cache.get()},
        {"all_v_cache", value_cache.get()}
    };

    auto self_decoder = std::make_unique<LlamaSelfDecoder<float>>(
        head_num, kv_head_num, head_size, intermediate_size, num_layers,
        attn_static_params, 
        rmsnorm_eps, 
        stream, 
        cublas_wrapper, 
        allocator
    );

    self_decoder->forward(
        &decoder_inputs, 
        &layer_weights, 
        &decoder_outputs, 
        &attn_dyn_params
    );
    cudaDeviceSynchronize();

    // Free allocated memory
    cudaFree(d_decoder_input);
    cudaFree(d_all_k_cache);
    cudaFree(d_all_v_cache);
    cudaFree(d_finished);
    cudaFree(d_qkv_weights);
    cudaFree(d_output_weights);
    cudaFree(d_qkv_bias);

    return 0;
}
