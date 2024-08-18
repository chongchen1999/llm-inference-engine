#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../src/layers/includes/self_attention.h"
#include "../../src/memory/memory_deleter.cuh"

int main() {
    const int h_layer_id = 0;
    const int h_step = 8;
    const int head_num = 8;
    const int kv_head_num = 8;
    const int head_size = 64;
    const int num_layers = 8;
    const int max_seq_len = 256;
    const int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    const int q_hidden_units = head_num * head_size;

    LlamaAttentionStaticParams attention_static_params;
    attention_static_params.rotary_embedding_dim = 128;
    attention_static_params.rotary_embedding_base = 10000;
    attention_static_params.max_position_embeddings = 2048;
    attention_static_params.use_dynamic_ntk = false; // for dynamic scaling rope
    attention_static_params.head_num = head_num;
    attention_static_params.kv_head_num = kv_head_num;
    attention_static_params.head_size = head_size;

    LlamaAttentionDynamicParams attention_dynamic_params;
    attention_dynamic_params.batch_size = 2;

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;

    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    auto cublas_wrapper = new CublasWrapper(cublas_handle, cublaslt_handle);
    auto allocator = new CudaAllocator();

    // Define the sizes for convenience
    const size_t input_size = q_hidden_units * attention_dynamic_params.batch_size;
    const size_t k_cache_size = num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size;
    const size_t v_cache_size = k_cache_size; // same size as k_cache_size
    const size_t qkv_weights_size = q_hidden_units * hidden_units;
    const size_t output_weights_size = q_hidden_units * q_hidden_units;
    const size_t qkv_bias_size = hidden_units;
    const size_t finished_size = attention_dynamic_params.batch_size;

    // Allocate and initialize host memory
    float *h_attention_input = new float[input_size];
    std::fill(h_attention_input, h_attention_input + input_size, 1.0f);

    float *h_all_k_cache = new float[k_cache_size];
    std::fill(h_all_k_cache, h_all_k_cache + k_cache_size, 1.0f);

    float *h_all_v_cache = new float[v_cache_size];
    std::fill(h_all_v_cache, h_all_v_cache + v_cache_size, 1.0f);

    bool *h_finished = new bool[finished_size];
    std::fill(h_finished, h_finished + finished_size, false);

    float *h_qkv_weights = new float[qkv_weights_size];
    std::fill(h_qkv_weights, h_qkv_weights + qkv_weights_size, 1.0f);

    float *h_output_weights = new float[output_weights_size];
    std::fill(h_output_weights, h_output_weights + output_weights_size, 1.0f);

    float *h_qkv_bias = new float[qkv_bias_size];
    std::fill(h_qkv_bias, h_qkv_bias + qkv_bias_size, 2.0f);

    // Allocate device memory
    float *d_attention_input;
    cudaMalloc(&d_attention_input, sizeof(float) * input_size);

    float *d_all_k_cache;
    cudaMalloc(&d_all_k_cache, sizeof(float) * k_cache_size);

    float *d_all_v_cache;
    cudaMalloc(&d_all_v_cache, sizeof(float) * v_cache_size);

    bool *d_finished;
    cudaMalloc(&d_finished, sizeof(bool) * finished_size);

    float *d_qkv_weights;
    cudaMalloc(&d_qkv_weights, sizeof(float) * qkv_weights_size);

    float *d_output_weights;
    cudaMalloc(&d_output_weights, sizeof(float) * output_weights_size);

    float *d_qkv_bias;
    cudaMalloc(&d_qkv_bias, sizeof(float) * qkv_bias_size);

    float *d_attention_output;
    cudaMalloc(&d_attention_output, sizeof(float) * input_size);

    // Copy data from host to device
    CHECK(cudaMemcpy(d_attention_input, h_attention_input, sizeof(float) * input_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_finished, h_finished, sizeof(bool) * finished_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_all_k_cache, h_all_k_cache, sizeof(float) * k_cache_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_all_v_cache, h_all_v_cache, sizeof(float) * v_cache_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_qkv_weights, h_qkv_weights, sizeof(float) * qkv_weights_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(float) * qkv_bias_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_output_weights, h_output_weights, sizeof(float) * output_weights_size, cudaMemcpyHostToDevice));

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

    auto attention_input = new TensorWrapper<float>(
        Device::GPU, type,
        {attention_dynamic_params.batch_size, q_hidden_units},
        d_attention_input
    );

    auto step = new TensorWrapper<int>(
        Device::CPU, type_int,
        {1},
        const_cast<int *>(&h_step)
    );

    auto finished = new TensorWrapper<bool>(
        Device::GPU, type_bool,
        {attention_dynamic_params.batch_size},
        d_finished
    );

    auto layer_id = new TensorWrapper<int>(
        Device::CPU, type_int,
        {1},
        const_cast<int *>(&h_layer_id)
    );

    auto attention_output = new TensorWrapper<float>(
        Device::GPU, type,
        {attention_dynamic_params.batch_size, q_hidden_units},
        d_attention_output
    );

    auto key_cache = new TensorWrapper<float>(
        Device::GPU, type,
        {num_layers, attention_dynamic_params.batch_size, kv_head_num, max_seq_len, head_size},
        d_all_k_cache
    );

    auto value_cache = new TensorWrapper<float>(
        Device::GPU, type,
        {num_layers, attention_dynamic_params.batch_size, kv_head_num, max_seq_len, head_size},
        d_all_v_cache
    );

    LLM_CHECK_WITH_INFO(attention_input->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(step->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(finished->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(layer_id->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");

    TensorMap self_attention_inputs{
        {"attention_input", attention_input},
        {"step", step},
        {"finished", finished},
        {"layer_id", layer_id},
    };

    TensorMap self_attention_outputs{
        {"attention_output", attention_output},
        {"all_k_cache", key_cache},
        {"all_v_cache", value_cache},
    };

    auto self_attn_layer = new LlamaSelfAttentionLayer<float>(
        head_num,
        kv_head_num,
        head_size,
        &attention_static_params,
        stream,
        cublas_wrapper,
        allocator
    );

    std::cout << "ready!\n" << std::endl;

    self_attn_layer->forward(
        &self_attention_inputs,
        &self_attention_outputs,
        &self_attention_weights,
        &attention_dynamic_params
    );
    cudaDeviceSynchronize();

    // Clean up
    deallocate(attention_input, "new");
    deallocate(step, "new");
    deallocate(finished, "new");
    deallocate(layer_id, "new");
    deallocate(attention_output, "new");
    deallocate(key_cache, "new");
    deallocate(value_cache, "new");

    deallocate(h_attention_input, "new[]");
    deallocate(h_all_k_cache, "new[]");
    deallocate(h_all_v_cache, "new[]");
    deallocate(h_finished, "new[]");
    deallocate(h_qkv_weights, "new[]");
    deallocate(h_output_weights, "new[]");
    deallocate(h_qkv_bias, "new[]");

    deallocate(d_attention_input, "cudaMalloc");
    deallocate(d_all_k_cache, "cudaMalloc");
    deallocate(d_all_v_cache, "cudaMalloc");
    deallocate(d_finished, "cudaMalloc");
    deallocate(d_qkv_weights, "cudaMalloc");
    deallocate(d_output_weights, "cudaMalloc");
    deallocate(d_qkv_bias, "cudaMalloc");
    deallocate(d_attention_output, "cudaMalloc");

    deallocate(cublas_wrapper, "new");
    deallocate(allocator, "new");

    return 0;
}
