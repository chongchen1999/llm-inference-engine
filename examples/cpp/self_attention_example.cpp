#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../src/layers/includes/self_attention.h"

// Current example doesn't consider layer_id in masked self-attention
int main() {
    const int h_step = 3;
    const int head_num = 4;
    const int kv_head_num = 2;
    const int head_size = 8;
    const int num_layers = 1;
    const int max_seq_len = 12;
    const int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    const int q_hidden_units = head_num * head_size;

    LlamaAttentionStaticParams attention_static_params;
    attention_static_params.rotary_embedding_dim = 128;
    attention_static_params.rotary_embedding_base = 10000;
    attention_static_params.max_position_embeddings = 2048;
    attention_static_params.use_dynamic_ntk = false; // for dynamic scaling rope

    LlamaAttentionDynamicParams attention_dynamic_params;
    attention_dynamic_params.batch_size = 2;

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    auto cublas_wrapper = std::make_unique<cublasWrapper>(cublas_handle, cublaslt_handle);
    auto allocator = std::make_unique<CudaAllocator>();

    // Prepare input, weight, and output data
    std::vector<float> h_attention_input(q_hidden_units * attention_dynamic_params.batch_size, 1.0f);
    float *d_attention_input;
    cudaMalloc(reinterpret_cast<void **>(&d_attention_input), sizeof(float) * q_hidden_units * attention_dynamic_params.batch_size);

    std::vector<float> h_all_k_cache(num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size, 1.0f);
    float *d_all_k_cache;
    cudaMalloc(reinterpret_cast<void **>(&d_all_k_cache), sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size);

    std::vector<float> h_all_v_cache(num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size, 1.0f);
    float *d_all_v_cache;
    cudaMalloc(reinterpret_cast<void **>(&d_all_v_cache), sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size);

    const int h_layer_id = 0;
    std::vector<bool> h_finished(attention_dynamic_params.batch_size, false);
    bool *d_finished;
    cudaMalloc(reinterpret_cast<void **>(&d_finished), sizeof(bool) * attention_dynamic_params.batch_size);

    std::vector<float> h_qkv_weights(q_hidden_units * hidden_units, 1.0f);
    float *d_qkv_weights;
    cudaMalloc(reinterpret_cast<void **>(&d_qkv_weights), sizeof(float) * q_hidden_units * hidden_units);

    std::vector<float> h_output_weights(q_hidden_units * q_hidden_units, 1.0f);
    float *d_output_weights;
    cudaMalloc(reinterpret_cast<void **>(&d_output_weights), sizeof(float) * q_hidden_units * q_hidden_units);

    std::vector<float> h_qkv_bias(hidden_units, 2.0f);
    float *d_qkv_bias;
    cudaMalloc(reinterpret_cast<void **>(&d_qkv_bias), sizeof(float) * hidden_units);

    float *d_attention_output;
    cudaMalloc(reinterpret_cast<void **>(&d_attention_output), sizeof(float) * q_hidden_units * attention_dynamic_params.batch_size);

    CHECK(cudaMemcpy(d_attention_input, h_attention_input.data(), sizeof(float) * q_hidden_units * attention_dynamic_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_finished, h_finished.data(), sizeof(bool) * attention_dynamic_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_all_k_cache, h_all_k_cache.data(), sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_all_v_cache, h_all_v_cache.data(), sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_qkv_weights, h_qkv_weights.data(), sizeof(float) * q_hidden_units * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_qkv_bias, h_qkv_bias.data(), sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_output_weights, h_output_weights.data(), sizeof(float) * q_hidden_units * q_hidden_units, cudaMemcpyHostToDevice));

    DataType type = getTensorType<float>();
    DataType type_int = getTensorType<int>();
    DataType type_bool = getTensorType<bool>();
    WeightType wtype = getWeightType<float>();

    LlamaAttentionWeights<float> self_attn_weights;
    self_attn_weights.qkv.data = d_qkv_weights;
    self_attn_weights.qkv.shape = {q_hidden_units, hidden_units};
    self_attn_weights.qkv.type = wtype;
    self_attn_weights.qkv.bias = d_qkv_bias;
    self_attn_weights.output.data = d_output_weights;
    self_attn_weights.output.shape = {q_hidden_units, q_hidden_units};
    self_attn_weights.output.type = wtype;

    auto attention_input = std::make_unique<TensorWrapper<float>>(Device::GPU, type, {attention_dynamic_params.batch_size, q_hidden_units}, d_attention_input);
    auto step = std::make_unique<TensorWrapper<int>>(Device::CPU, type_int, {1}, &h_step);
    auto finished = std::make_unique<TensorWrapper<bool>>(Device::GPU, type_bool, {attention_dynamic_params.batch_size}, d_finished);
    auto layer_id = std::make_unique<TensorWrapper<int>>(Device::CPU, type_int, {1}, &h_layer_id);
    auto attention_output = std::make_unique<TensorWrapper<float>>(Device::GPU, type, {attention_dynamic_params.batch_size, q_hidden_units}, d_attention_output);
    auto key_cache = std::make_unique<TensorWrapper<float>>(Device::GPU, type, {num_layers, attention_dynamic_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_k_cache);
    auto value_cache = std::make_unique<TensorWrapper<float>>(Device::GPU, type, {num_layers, attention_dynamic_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_v_cache);

    LLM_CHECK_WITH_INFO(attention_input->data != nullptr, "The data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(step->data != nullptr, "The data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(finished->data != nullptr, "The data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(layer_id->data != nullptr, "The data ptr of tensor inserted into TensorMap is nullptr!");

    TensorMap masked_attn_inputs {
        {"attention_input", attention_input.get()},
        {"sequence_lengths", Tensor(Device::GPU, type, {hidden_units}, d_qkv_bias)},
        {"total_padding_len", Tensor(Device::GPU, type_int, {attention_dynamic_params.batch_size}, d_padding_offset)},
        {"step", step.get()},
        {"finished", finished.get()},
        {"layer_id", layer_id.get()}
    };

    TensorMap masked_attn_outputs {
        {"attention_output", attention_output.get()},
        {"all_k_cache", key_cache.get()},
        {"all_v_cache", value_cache.get()}
    };

    auto self_attn_layer = std::make_unique<LlamaSelfAttentionLayer<float>>(
        head_num, kv_head_num, head_size, attention_static_params, stream, cublas_wrapper.get(), allocator.get()
    );
    self_attn_layer->forward(masked_attn_inputs, masked_attn_outputs, self_attn_weights, attention_dynamic_params);
    cudaDeviceSynchronize();

    cudaFree(d_attention_input);
    cudaFree(d_all_k_cache);
    cudaFree(d_all_v_cache);
    cudaFree(d_finished);
    cudaFree(d_qkv_weights);
    cudaFree(d_output_weights);
    cudaFree(d_qkv_bias);
    free(h_attention_input.data());
    free(h_all_k_cache.data());
    free(h_all_v_cache.data());
    free(h_finished.data());
    free(h_qkv_weights.data());
    free(h_output_weights.data());
    free(h_qkv_bias.data());

    return 0;
}
