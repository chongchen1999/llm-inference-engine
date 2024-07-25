#include <iostream>
#include <vector>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>
#include "src/layers/includes/self_decoder.h"
#include "src/utils/macro.h"

// Current example doesn't consider layer_id in masked self-attention
// Now consider it
int main() {
    const int h_step = 3;
    const int head_num = 4;
    const int kv_head_num = 2;
    const int head_size = 8;
    const int inter_size = 12;
    const int num_layers = 32;
    const int max_seq_len = 12;
    const int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    const int q_hidden_units = head_num * head_size;
    const float rmsnorm_eps = 1e-6;

    LlamaAttentionStaticParams attn_static_params;
    attn_static_params.rotary_embedding_dim = 128;
    attn_static_params.rotary_embedding_base = 10000;
    attn_static_params.max_position_embeddings = 2048;
    attn_static_params.use_dynamic_ntk = false; // for dynamic scaling with rotary embeddings

    LlamaAttentionDynamicParams attn_dyn_params;
    attn_dyn_params.batch_size = 2;

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    auto cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    auto allocator = new CudaAllocator;

    // Prepare input, weight, and output data
    float *h_decoder_input = static_cast<float *>(malloc(sizeof(float) * q_hidden_units * attn_dyn_params.batch_size));
    float *d_decoder_input;
    cudaMalloc(reinterpret_cast<void **>(&d_decoder_input), sizeof(float) * q_hidden_units * attn_dyn_params.batch_size);

    for (int i = 0; i < q_hidden_units * attn_dyn_params.batch_size; ++i) {
        h_decoder_input[i] = rand() % 100 / static_cast<float>(1000);
    }

    float *d_decoder_output;
    cudaMalloc(reinterpret_cast<void **>(&d_decoder_output), sizeof(float) * q_hidden_units * attn_dyn_params.batch_size);

    // Max sequence length is the maximum kv cache length
    float *h_all_k_cache = static_cast<float *>(
        malloc(sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size)
    );
    float *d_all_k_cache;
    cudaMalloc(
        reinterpret_cast<void **>(&d_all_k_cache), 
        sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size
    );

    float *h_all_v_cache = static_cast<float *>(
        malloc(sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size)
    );
    float *d_all_v_cache;
    cudaMalloc(
        reinterpret_cast<void **>(&d_all_v_cache), 
        sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size
    );

    for (int i = 0; i < num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size; ++i) {
        h_all_k_cache[i] = 1.0f;
        h_all_v_cache[i] = 1.0f;
    }

    int layer_id = 0;
    bool *h_finished = static_cast<bool *>(malloc(sizeof(bool) * attn_dyn_params.batch_size));
    bool *d_finished;
    cudaMalloc(reinterpret_cast<void **>(&d_finished), sizeof(bool) * attn_dyn_params.batch_size);

    for (int i = 0; i < attn_dyn_params.batch_size; ++i) {
        h_finished[i] = false;
    }

    // Weights
    float *h_output_norm_weight = static_cast<float *>(malloc(sizeof(float) * q_hidden_units));
    float *d_output_norm_weight;
    cudaMalloc(reinterpret_cast<void **>(&d_output_norm_weight), sizeof(float) * q_hidden_units);

    for (int i = 0; i < q_hidden_units; ++i) {
        h_output_norm_weight[i] = 2.0f;
    }

    float *h_attn_norm_weight = static_cast<float *>(malloc(sizeof(float) * q_hidden_units));
    float *d_attn_norm_weight;
    cudaMalloc(reinterpret_cast<void **>(&d_attn_norm_weight), sizeof(float) * q_hidden_units);

    for (int i = 0; i < q_hidden_units; ++i) {
        h_attn_norm_weight[i] = 1.0f;
    }

    float *h_ffn_norm_weight = static_cast<float *>(malloc(sizeof(float) * q_hidden_units));
    float *d_ffn_norm_weight;
    cudaMalloc(reinterpret_cast<void **>(&d_ffn_norm_weight), sizeof(float) * q_hidden_units);

    for (int i = 0; i < q_hidden_units; ++i) {
        h_ffn_norm_weight[i] = 1.0f;
    }

    float *h_qkv_weights = static_cast<float *>(malloc(sizeof(float) * hidden_units * q_hidden_units));
    float *d_qkv_weights;
    cudaMalloc(reinterpret_cast<void **>(&d_qkv_weights), sizeof(float) * hidden_units * q_hidden_units);

    for (int i = 0; i < hidden_units * q_hidden_units; ++i) {
        h_qkv_weights[i] = 1.0f;
    }

    float *h_qkv_bias = static_cast<float *>(malloc(sizeof(float) * hidden_units));
    float *d_qkv_bias;
    cudaMalloc(reinterpret_cast<void **>(&d_qkv_bias), sizeof(float) * hidden_units);

    for (int i = 0; i < hidden_units; ++i) {
        h_qkv_bias[i] = 2.0f;
    }

    float *h_output_weights = static_cast<float *>(malloc(sizeof(float) * q_hidden_units * q_hidden_units));
    float *d_output_weights;
    cudaMalloc(reinterpret_cast<void **>(&d_output_weights), sizeof(float) * q_hidden_units * q_hidden_units);

    for (int i = 0; i < q_hidden_units * q_hidden_units; ++i) {
        h_output_weights[i] = 1.0f;
    }

    float *h_out_bias = static_cast<float *>(malloc(sizeof(float) * head_num * head_size));
    float *d_out_bias;
    cudaMalloc(reinterpret_cast<void **>(&d_out_bias), sizeof(float) * head_num * head_size);

    for (int i = 0; i < head_num * head_size; ++i) {
        h_out_bias[i] = 2.0f;
    }

    float *d_ffn_gate, *d_ffn_up, *d_ffn_down, *d_ffn_down_bias;
    float *h_ffn_gate_up = static_cast<float *>(malloc(sizeof(float) * hidden_units * 2 * inter_size));
    float *h_ffn_down = static_cast<float *>(malloc(sizeof(float) * hidden_units * inter_size));
    float *h_ffn_down_bias = static_cast<float *>(malloc(sizeof(float) * hidden_units));

    cudaMalloc(reinterpret_cast<void **>(&d_ffn_gate), sizeof(float) * hidden_units * 2 * inter_size);
    cudaMalloc(reinterpret_cast<void **>(&d_ffn_down), sizeof(float) * hidden_units * inter_size);
    cudaMalloc(reinterpret_cast<void **>(&d_ffn_down_bias), sizeof(float) * hidden_units);

    for (int i = 0; i < hidden_units * 2 * inter_size; ++i) {
        h_ffn_gate_up[i] = 2.0f;
    }

    for (int i = 0; i < hidden_units * inter_size; ++i) {
        h_ffn_down[i] = 2.0f;
        if (i < hidden_units) {
            h_ffn_down_bias[i] = 0.0f;
        }
    }

    // Host to Device memory copy
    cudaMemcpy(
        d_decoder_input, h_decoder_input, 
        sizeof(float) * q_hidden_units * attn_dyn_params.batch_size, 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_all_k_cache, h_all_k_cache, 
        sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_all_v_cache, h_all_v_cache, 
        sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_finished, h_finished, 
        sizeof(bool) * attn_dyn_params.batch_size, 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_output_norm_weight, h_output_norm_weight, 
        sizeof(float) * q_hidden_units, 
        cudaMemcpyHostToDevice
    );

    DataType type = getTensorType<float>(); // Note: The type should be a class data member!
    DataType type_int = getTensorType<int>();
    DataType type_bool = getTensorType<bool>();
    std::vector<LlamaLayerWeight<float> *> layer_weights;
    WeightType wtype = getWeightType<float>();
    layer_weights.reserve(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        layer_weights.push_back(
            new LlamaLayerWeight<float>(
                head_num, kv_head_num, head_size, inter_size, wtype, /*attn_bias*/ true
            )
        );
        layer_weights.back()->loadWeights();
    }

    auto decoder_input = new TensorWrapper<float>(
        Device::GPU, type, {attn_dyn_params.batch_size, q_hidden_units}, d_decoder_input
    );

    auto step = new TensorWrapper<int>(
        Device::CPU, type_int, {1}, &h_step
    );

    auto finished = new TensorWrapper<bool>(
        Device::GPU, type_bool, {attn_dyn_params.batch_size}, d_finished
    );

    auto layer = new TensorWrapper<int>(
        Device::CPU, type_int, {1}, &layer_id
    );

    auto output_norm_weight = new TensorWrapper<float>(
        Device::GPU, type, {q_hidden_units}, d_output_norm_weight
    );

    auto decoder_output = new TensorWrapper<float>(
        Device::GPU, type, {attn_dyn_params.batch_size, q_hidden_units}, d_decoder_output
    );

    auto key_cache = new TensorWrapper<float>(
        Device::GPU, type, {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_k_cache
    );

    auto value_cache = new TensorWrapper<float>(
        Device::GPU, type, {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_v_cache
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
        head_num, 
        kv_head_num, 
        head_size, 
        inter_size, 
        num_layers,
        attn_static_params, 
        rmsnorm_eps, 
        stream, 
        cublas_wrapper, 
        allocator
    );

    self_decoder->forward(decoder_inputs, layer_weights, decoder_outputs, attn_dyn_params);
    cudaDeviceSynchronize();

    // Free allocated memory
    free(h_decoder_input);
    free(h_all_k_cache);
    free(h_all_v_cache);
    free(h_finished);
    free(h_qkv_weights);
    free(h_output_weights);
    free(h_qkv_bias);
    cudaFree(d_decoder_input);
    cudaFree(d_all_k_cache);
    cudaFree(d_all_v_cache);
    cudaFree(d_finished);
    cudaFree(d_qkv_weights);
    cudaFree(d_output_weights);
    cudaFree(d_qkv_bias);

    return 0;
}
