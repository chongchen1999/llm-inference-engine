#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../src/layers/includes/context_attention.h"

int main(int argc, char **argv) {
    constexpr int head_num = 4;
    constexpr int kv_head_num = 2;
    constexpr int head_size = 8;
    constexpr int num_layers = 1;
    constexpr int max_seq_len = 12; // Max context length for KV cache
    const int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    const int q_hidden_units = head_num * head_size;

    LlamaAttentionStaticParams attention_static_params;
    attention_static_params.rotary_embedding_dim = 128;
    attention_static_params.rotary_embedding_base = 10000;
    attention_static_params.max_position_embeddings = 2048;
    attention_static_params.use_dynamic_ntk = false; // For dynamic scaling ROPE

    LlamaAttentionDynamicParams attn_dyn_params;
    attn_dyn_params.batch_size = 2;
    attn_dyn_params.num_tokens = 14;
    attn_dyn_params.max_q_len = 8;
    attn_dyn_params.max_k_len = 8; // Max actual context length for current batch

    constexpr bool is_free_buffer_after_fwd = true;

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    auto *cublas_wrapper = &CublasWrapper(cublas_handle, cublaslt_handle);
    auto *allocator = &CudaAllocator();

    // Prepare input, weight, and output data
    float *h_attention_input = static_cast<float *>(malloc(sizeof(float) * q_hidden_units * attn_dyn_params.num_tokens));
    float *d_attention_input;
    cudaMalloc(reinterpret_cast<void **>(&d_attention_input), sizeof(float) * q_hidden_units * attn_dyn_params.num_tokens);

    for (int i = 0; i < q_hidden_units * attn_dyn_params.num_tokens; ++i) {
        h_attention_input[i] = 1.0f;
    }

    float *h_qkv_weights = static_cast<float *>(malloc(sizeof(float) * q_hidden_units * hidden_units));
    float *d_qkv_weights;
    cudaMalloc(reinterpret_cast<void **>(&d_qkv_weights), sizeof(float) * q_hidden_units * hidden_units);

    for (int i = 0; i < hidden_units * q_hidden_units; ++i) {
        h_qkv_weights[i] = 1.0f;
    }

    float *h_mask = static_cast<float *>(malloc(sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len));
    float *d_mask;
    cudaMalloc(reinterpret_cast<void **>(&d_mask), sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len);

    for (int i = 0; i < attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len; ++i) {
        h_mask[i] = 1.0f;
    }

    float *h_qkv_bias = static_cast<float *>(malloc(sizeof(float) * hidden_units));
    float *d_qkv_bias;
    cudaMalloc(reinterpret_cast<void **>(&d_qkv_bias), sizeof(float) * hidden_units);

    for (int i = 0; i < hidden_units; ++i) {
        h_qkv_bias[i] = 2.0f;
    }

    // Max_seq_len is the max KV cache length
    float *h_all_k_cache = static_cast<float *>(malloc(sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size));
    float *d_all_k_cache;
    cudaMalloc(reinterpret_cast<void **>(&d_all_k_cache), sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);

    float *h_all_v_cache = static_cast<float *>(malloc(sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size));
    float *d_all_v_cache;
    cudaMalloc(reinterpret_cast<void **>(&d_all_v_cache), sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);

    for (int i = 0; i < num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size; ++i) {
        h_all_k_cache[i] = 1.0f;
        h_all_v_cache[i] = 1.0f;
    }

    // Padding to max_q_len
    int *h_padding_offset = static_cast<int *>(malloc(sizeof(int) * attn_dyn_params.num_tokens));
    int *d_padding_offset;
    cudaMalloc(reinterpret_cast<void **>(&d_padding_offset), sizeof(int) * attn_dyn_params.num_tokens);

    for (int i = 0; i < attn_dyn_params.num_tokens; ++i) {
        h_padding_offset[i] = (i < 7) ? 0 : 1; // Two seqlens are both 7, tokens num = 14
    }

    int *h_history_len = static_cast<int *>(malloc(sizeof(int) * attn_dyn_params.batch_size));
    int *d_history_len;
    cudaMalloc(reinterpret_cast<void **>(&d_history_len), sizeof(int) * attn_dyn_params.batch_size);

    int *h_input_len = static_cast<int *>(malloc(sizeof(int) * attn_dyn_params.batch_size));
    int *d_input_len;
    cudaMalloc(reinterpret_cast<void **>(&d_input_len), sizeof(int) * attn_dyn_params.batch_size);

    int h_layer_id = 0;
    int *h_context_len = static_cast<int *>(malloc(sizeof(int) * attn_dyn_params.batch_size));
    int *d_context_len;
    cudaMalloc(reinterpret_cast<void **>(&d_context_len), sizeof(int) * attn_dyn_params.batch_size);

    for (int i = 0; i < attn_dyn_params.batch_size; ++i) {
        h_history_len[i] = 0; // For KV cache cumsum seqlen and ROPE's timestep compute
        h_input_len[i] = 7;   // Corresponding to padding offset
        h_context_len[i] = h_history_len[i] + h_input_len[i];
    }

    float *d_attention_output;
    cudaMalloc(reinterpret_cast<void **>(&d_attention_output), sizeof(float) * attn_dyn_params.num_tokens * q_hidden_units);

    float *d_output_weights;
    cudaMalloc(reinterpret_cast<void **>(&d_output_weights), sizeof(float) * q_hidden_units * q_hidden_units);

    // Host to device memory copy
    cudaMemcpy(d_attention_input, h_attention_input, sizeof(float) * q_hidden_units * attn_dyn_params.num_tokens, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv_weights, h_qkv_weights, sizeof(float) * q_hidden_units * hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(float) * hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_k_cache, h_all_k_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_v_cache, h_all_v_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_padding_offset, h_padding_offset, sizeof(int) * attn_dyn_params.num_tokens, cudaMemcpyHostToDevice);
    cudaMemcpy(d_history_len, h_history_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_context_len, h_context_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_len, h_input_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len, cudaMemcpyHostToDevice);

    // Prepare input, weight, and output tensor
    const DataType type = getTensorType<float>(); // Note: the type should be a class data member!
    const DataType type_int = getTensorType<int>();

    auto *attention_input = new TensorWrapper<float>(
        Device::GPU,
        type,
        {attn_dyn_params.num_tokens, q_hidden_units},
        d_attention_input
    );

    auto *qkv_bias = new TensorWrapper<float>(
        Device::GPU,
        type,
        {hidden_units},
        d_qkv_bias
    );

    auto *padding_offset = new TensorWrapper<int>(
        Device::GPU,
        type_int,
        {attn_dyn_params.num_tokens},
        d_padding_offset
    );

    auto *history_length = new TensorWrapper<int>(
        Device::GPU,
        type_int,
        {attn_dyn_params.batch_size},
        d_history_len
    );

    auto *input_length = new TensorWrapper<int>(
        Device::GPU,
        type_int,
        {attn_dyn_params.batch_size},
        d_input_len
    );

    auto *layer_id = new TensorWrapper<int>(
        Device::CPU,
        type_int,
        {1},
        &h_layer_id
    );

    auto *context_length = new TensorWrapper<int>(
        Device::GPU,
        type_int,
        {attn_dyn_params.batch_size},
        d_context_len
    );

    auto *attention_mask = new TensorWrapper<float>(
        Device::GPU,,
        type,
        {attn_dyn_params.batch_size, attn_dyn_params.max_q_len, attn_dyn_params.max_k_len},
        d_mask
    );

    auto *attention_output = new TensorWrapper<float>(
        Device::GPU,,
        type,
        {attn_dyn_params.num_tokens, q_hidden_units},
        d_attention_output
    );

    auto *all_k_cache = new TensorWrapper<float>(
        Device::GPU,
        type,
        {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size},
        d_all_k_cache
    );

    auto *all_v_cache = new TensorWrapper<float>(
        Device::GPU,
        type,
        {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size},
        d_all_v_cache
    );

    LLM_CHECK_WITH_INFO(attention_input->data != nullptr, "Tensor inserted in TensorMap is nullptr data!");
    LLM_CHECK_WITH_INFO(qkv_bias->data != nullptr, "Tensor inserted in TensorMap is nullptr data!");
    LLM_CHECK_WITH_INFO(padding_offset->data != nullptr, "Tensor inserted in TensorMap is nullptr data!");
    LLM_CHECK_WITH_INFO(history_length->data != nullptr, "Tensor inserted in TensorMap is nullptr data!");
    LLM_CHECK_WITH_INFO(input_length->data != nullptr, "Tensor inserted in TensorMap is nullptr data!");
    LLM_CHECK_WITH_INFO(layer_id->data != nullptr, "Tensor inserted in TensorMap is nullptr data!");
    LLM_CHECK_WITH_INFO(context_length->data != nullptr, "Tensor inserted in TensorMap is nullptr data!");
    LLM_CHECK_WITH_INFO(attention_mask->data != nullptr, "Tensor inserted in TensorMap is nullptr data!");

    TensorMap context_attention_inputs{
        {"attention_input", attention_input},
        {"qkv_bias", qkv_bias},
        {"padding_offset", padding_offset},
        {"history_length", history_length},
        {"input_length", input_length},
        {"layer_id", layer_id},
        {"context_length", context_length},
        {"attention_mask", attention_mask}
    };

    TensorMap context_attention_outputs{
        {"attention_output", attention_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };

    // Weights are initialized in its constructor (see cpp/models/bert/bertlayerweight.cc)
    LlamaAttentionWeights<float> context_attention_weights;
    const WeightType wtype = getWeightType<float>();

    context_attention_weights.qkv.data = d_qkv_weights;
    context_attention_weights.qkv.shape = {q_hidden_units, hidden_units};
    context_attention_weights.qkv.type = wtype;
    context_attention_weights.qkv.bias = d_qkv_bias;

    context_attention_weights.output.data = d_output_weights;
    context_attention_weights.output.shape = {q_hidden_units, q_hidden_units};
    context_attention_weights.output.type = wtype;

    // Initialize context attention layer
    auto *context_attention = new LlamaContextAttentionLayer<float>(
        head_num,
        kv_head_num,
        head_size,
        attention_static_params,
        stream,
        cublas_wrapper,
        allocator
    );

    // Forward pass
    context_attention->forward(
        context_attention_inputs, 
        context_attention_outputs, 
        context_attention_weights, 
        attn_dyn_params, 
        attention_static_params
    );

    // Free buffer
    cudaDeviceSynchronize();
    free(h_attention_input);
    cudaFree(d_attention_input);
    free(h_qkv_bias);
    cudaFree(d_qkv_bias);
    free(h_all_k_cache);
    cudaFree(d_all_k_cache);
    free(h_all_v_cache);
    cudaFree(d_all_v_cache);
    free(h_padding_offset);
    cudaFree(d_padding_offset);
    free(h_history_len);
    cudaFree(d_history_len);
    free(h_input_len);
    cudaFree(d_input_len);
    free(h_context_len);
    cudaFree(d_context_len);
    cudaFree(d_attention_output);
    return 0;
}
