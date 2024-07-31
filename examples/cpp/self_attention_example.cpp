#include <iostream>
#include <vector>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../src/layers/includes/self_attention.h"

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

    auto cublas_wrapper = std::make_unique<CublasWrapper>(cublas_handle, cublaslt_handle);
    auto allocator = std::make_unique<CudaAllocator>();

    // Prepare input, weight, and output data
    auto h_attention_input = std::make_unique<float[]>(q_hidden_units * attention_dynamic_params.batch_size);
    float *d_attention_input;
    cudaMalloc(&d_attention_input, sizeof(float) * q_hidden_units * attention_dynamic_params.batch_size);
    std::fill(h_attention_input.get(), h_attention_input.get() + q_hidden_units * attention_dynamic_params.batch_size, 1.0f);

    auto h_all_k_cache = std::make_unique<float[]>(num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size);
    float *d_all_k_cache;
    cudaMalloc(&d_all_k_cache, sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size);

    auto h_all_v_cache = std::make_unique<float[]>(num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size);
    float *d_all_v_cache;
    cudaMalloc(&d_all_v_cache, sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size);

    std::fill(h_all_k_cache.get(), h_all_k_cache.get() + num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size, 1.0f);
    std::fill(h_all_v_cache.get(), h_all_v_cache.get() + num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size, 1.0f);

    const int h_layer_id = 0;
    auto h_finished = std::make_unique<bool[]>(attention_dynamic_params.batch_size);
    bool *d_finished;
    cudaMalloc(&d_finished, sizeof(bool) * attention_dynamic_params.batch_size);
    std::fill(h_finished.get(), h_finished.get() + attention_dynamic_params.batch_size, false);

    auto h_qkv_weights = std::make_unique<float[]>(q_hidden_units * hidden_units);
    float *d_qkv_weights;
    cudaMalloc(&d_qkv_weights, sizeof(float) * q_hidden_units * hidden_units);
    std::fill(h_qkv_weights.get(), h_qkv_weights.get() + q_hidden_units * hidden_units, 1.0f);

    auto h_output_weights = std::make_unique<float[]>(q_hidden_units * q_hidden_units);
    float *d_output_weights;
    cudaMalloc(&d_output_weights, sizeof(float) * q_hidden_units * q_hidden_units);
    std::fill(h_output_weights.get(), h_output_weights.get() + q_hidden_units * q_hidden_units, 1.0f);

    auto h_qkv_bias = std::make_unique<float[]>(hidden_units);
    float *d_qkv_bias;
    cudaMalloc(&d_qkv_bias, sizeof(float) * hidden_units);
    std::fill(h_qkv_bias.get(), h_qkv_bias.get() + hidden_units, 2.0f);

    float *d_attention_output;
    cudaMalloc(&d_attention_output, sizeof(float) * q_hidden_units * attention_dynamic_params.batch_size);

    CHECK(cudaMemcpy(d_attention_input, h_attention_input.get(), sizeof(float) * q_hidden_units * attention_dynamic_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_finished, h_finished.get(), sizeof(bool) * attention_dynamic_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_all_k_cache, h_all_k_cache.get(), sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_all_v_cache, h_all_v_cache.get(), sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_qkv_weights, h_qkv_weights.get(), sizeof(float) * q_hidden_units * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_qkv_bias, h_qkv_bias.get(), sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_output_weights, h_output_weights.get(), sizeof(float) * q_hidden_units * q_hidden_units, cudaMemcpyHostToDevice));

    const DataType type = getTensorType<float>();
    const DataType type_int = getTensorType<int>();
    const DataType type_bool = getTensorType<bool>();

    LlamaAttentionWeights<float> self_attention_weights;
    const WeightType wtype = getWeightType<float>();

    self_attention_weights.qkv.data = d_qkv_weights;
    self_attention_weights.qkv.shape = {q_hidden_units, hidden_units};
    self_attention_weights.qkv.type = wtype;
    self_attention_weights.qkv.bias = d_qkv_bias;
    self_attention_weights.output.data = d_output_weights;
    self_attention_weights.output.shape = {q_hidden_units, q_hidden_units};
    self_attention_weights.output.type = wtype;

    std::unique_ptr<TensorWrapper<float>> attention_input(
        new TensorWrapper<float>(Device::GPU, type, {attention_dynamic_params.batch_size, q_hidden_units}, d_attention_input)
    );

    std::unique_ptr<TensorWrapper<int>> step(
        new TensorWrapper<int>(Device::CPU, type_int, {1}, const_cast<int *>(&h_step))
    );

    std::unique_ptr<TensorWrapper<bool>> finished(
        new TensorWrapper<bool>(Device::GPU, type_bool, {attention_dynamic_params.batch_size}, d_finished)
    );

    std::unique_ptr<TensorWrapper<int>> layer_id(
        new TensorWrapper<int>(Device::CPU, type_int, {1}, const_cast<int *>(&h_layer_id))
    );

    std::unique_ptr<TensorWrapper<float>> attention_output(
        new TensorWrapper<float>(Device::GPU, type, {attention_dynamic_params.batch_size, q_hidden_units}, d_attention_output)
    );

    std::unique_ptr<TensorWrapper<float>> key_cache(
        new TensorWrapper<float>(Device::GPU, type, {num_layers, attention_dynamic_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_k_cache)
    );

    std::unique_ptr<TensorWrapper<float>> value_cache(
        new TensorWrapper<float>(Device::GPU, type, {num_layers, attention_dynamic_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_v_cache)
    );


    LLM_CHECK_WITH_INFO(attention_input->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(step->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(finished->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(layer_id->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");

    TensorMap self_attention_inputs{
        {"attention_input", attention_input.get()},
        {"step", step.get()},
        {"finished", finished.get()},
        {"layer_id", layer_id.get()},
    };

    TensorMap self_attention_outputs{
        {"attention_output", attention_output.get()},
        {"all_k_cache", key_cache.get()},
        {"all_v_cache", value_cache.get()},
    };

    auto self_attn_layer = std::make_unique<LlamaSelfAttentionLayer<float>>(
        head_num,
        kv_head_num,
        head_size,
        &attention_static_params,
        stream,
        cublas_wrapper.get(),
        allocator.get()
    );

    std::cout<< "ready!\n" << std::endl;

    self_attn_layer->forward(
        &self_attention_inputs,
        &self_attention_outputs,
        &self_attention_weights,
        &attention_dynamic_params
    );
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(d_attention_input);
    cudaFree(d_all_k_cache);
    cudaFree(d_all_v_cache);
    cudaFree(d_finished);
    cudaFree(d_qkv_weights);
    cudaFree(d_output_weights);
    cudaFree(d_qkv_bias);
    cudaFree(d_attention_output);

    return 0;
}
