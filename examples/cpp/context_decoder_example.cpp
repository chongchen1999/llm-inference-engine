#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include "src/layers/includes/context_decoder.h"
#include "src/utils/macro.h"
#include "src/models/tokenizer.h"
#include "src/kernels/includes/input_embedding.h"
#include "src/weights/llama/embedding_weights.h"

int main(int argc, char **argv) {
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;

    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    cublasWrapper *cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    BaseAllocator *allocator = new CudaAllocator;

    constexpr int head_num = 32;
    constexpr int kv_head_num = 32;
    constexpr int head_size = 128;
    constexpr int intermediate_size = 11008;
    constexpr int num_layers = 32;
    constexpr int max_seq_len = 64;
    constexpr float rmsnorm_eps = 1e-6;

    const int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    const int q_hidden_units = head_num * head_size;

    LLaMAAttentionStaticParams attn_static_params;
    attn_static_params.rotary_embedding_dim = 128;
    attn_static_params.rotary_embedding_base = 10000;
    attn_static_params.max_position_embeddings = 2048;
    attn_static_params.use_dynamic_ntk = false; // for dyn scaling rope

    const std::string input = "how old are you";
    Tokenizer tokenizer;
    tokenizer.Initialize("/home/llama2-7b-tokenizer.bin");

    std::vector<int> res = tokenizer.Encode(input);
    std::cout << "Input IDs length is " << res.size() << "\n";

    int *h_input_ids_buf = allocator->Malloc(
        nullptr, sizeof(int) * res.size(), true
    );

    for (int i = 0; i < res.size(); ++i) {
        h_input_ids_buf[i] = res[i]; // [max_context_token_nums_]
    }

    // Ensure all needed input buffers are prepared
    const int context_length = res.size();
    const int cur_input_length = res.size(); // Actual input length

    LLaMAAttentionDynParams attn_dyn_params;
    attn_dyn_params.batch_size = 1;
    attn_dyn_params.num_tokens = cur_input_length;
    attn_dyn_params.max_q_len = attn_dyn_params.num_tokens;
    attn_dyn_params.max_k_len = context_length;

    std::string retString = ""; // response tokens string

    auto input_ids = std::make_unique<TensorWrapper<int>>(
        GPU, getTensorType<int>(), {cur_input_length}
    );
    input_ids->data = allocator->Malloc(
        input_ids->data, sizeof(int) * cur_input_length, false
    );
    CHECK(cudaMemcpy(
        input_ids->data,
        h_input_ids_buf,
        sizeof(int) * cur_input_length,
        cudaMemcpyHostToDevice
    ));

    auto decoder_input = std::make_unique<TensorWrapper<float>>(
        GPU, getTensorType<float>(), {attn_dyn_params.num_tokens, q_hidden_units}
    );
    decoder_input->data = allocator->Malloc(
        decoder_input->data,
        sizeof(float) * attn_dyn_params.num_tokens * q_hidden_units,
        false
    );

    auto embedding = std::make_unique<float[]>(32000 * 4096);
    for (int i = 0; i < 32000 * 4096; ++i) {
        embedding[i] = rand() % 100 / 100000.0f;
    }

    float *d_embedding;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_embedding), sizeof(float) * 32000 * 4096));
    CHECK(cudaMemcpy(d_embedding, embedding.get(), sizeof(float) * 32000 * 4096, cudaMemcpyHostToDevice));

    EmbeddingWeight<float> embed_table;
    WeightType wtype = getWeightType<float>();
    embed_table.shape = {32000, 4096};
    embed_table.type = wtype;
    embed_table.data = d_embedding;

    launchInputEmbedding(input_ids.get(), decoder_input.get(), &embed_table);

    float *d_decoder_output;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_decoder_output), sizeof(float) * q_hidden_units * attn_dyn_params.num_tokens));

    auto h_mask = std::make_unique<float[]>(attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len);
    float *d_mask;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_mask), sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len));
    std::fill(h_mask.get(), h_mask.get() + attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len, 1.0f);

    auto h_all_k_cache = std::make_unique<float[]>(num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);
    float *d_all_k_cache;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_all_k_cache), sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size));
    auto h_all_v_cache = std::make_unique<float[]>(num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);
    float *d_all_v_cache;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_all_v_cache), sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size));

    for (int i = 0; i < num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size; ++i) {
        h_all_k_cache[i] = rand() % 100 / 100000.0f;
        h_all_v_cache[i] = rand() % 100 / 100000.0f;
    }

    auto h_padding_offset = std::make_unique<int[]>(attn_dyn_params.num_tokens);
    int *d_padding_offset;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_padding_offset), sizeof(int) * attn_dyn_params.num_tokens));
    std::fill(h_padding_offset.get(), h_padding_offset.get() + attn_dyn_params.num_tokens, 0);

    auto h_history_len = std::make_unique<int[]>(attn_dyn_params.batch_size);
    int *d_history_len;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_history_len), sizeof(int) * attn_dyn_params.batch_size));
    auto h_input_len = std::make_unique<int[]>(attn_dyn_params.batch_size);
    int *d_input_len;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_input_len), sizeof(int) * attn_dyn_params.batch_size));
    auto h_ctx_len = std::make_unique<int[]>(attn_dyn_params.batch_size);
    int *d_ctx_len;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ctx_len), sizeof(int) * attn_dyn_params.batch_size));

    for (int i = 0; i < attn_dyn_params.batch_size; ++i) {
        h_history_len[i] = 0; // For kv cache cumsum seqlen and rope's timestep compute
        h_input_len[i] = cur_input_length;
        h_ctx_len[i] = context_length;
    }

    auto h_output_norm_weight = std::make_unique<float[]>(q_hidden_units);
    float *d_output_norm_weight;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output_norm_weight), sizeof(float) * q_hidden_units));
    for (int i = 0; i < q_hidden_units; ++i) {
        h_output_norm_weight[i] = rand() % 100 / 100000.0f;
    }

    // Copy data to device
    CHECK(cudaMemcpy(d_all_k_cache, h_all_k_cache.get(), sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_all_v_cache, h_all_v_cache.get(), sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_padding_offset, h_padding_offset.get(), sizeof(int) * attn_dyn_params.num_tokens, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_history_len, h_history_len.get(), sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ctx_len, h_ctx_len.get(), sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_input_len, h_input_len.get(), sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mask, h_mask.get(), sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_output_norm_weight, h_output_norm_weight.get(), sizeof(float) * q_hidden_units, cudaMemcpyHostToDevice));

    int layer_id = 0;
    DataType qtype = getDataType<float>();
    launchContextDecoder(
        nullptr, // no allocation required
        nullptr, // no allocation required
        nullptr, // no allocation required
        d_all_k_cache,
        d_all_v_cache,
        d_decoder_output,
        d_padding_offset,
        d_history_len,
        d_ctx_len,
        d_input_len,
        nullptr,
        d_mask,
        d_output_norm_weight,
        &attn_dyn_params,
        &attn_static_params,
        &cublas_wrapper->cublasLtHandle(),
        &cublas_wrapper->cublasHandle(),
        stream,
        layer_id
    );

    CHECK(cudaFree(d_embedding));
    CHECK(cudaFree(d_decoder_output));
    CHECK(cudaFree(d_all_k_cache));
    CHECK(cudaFree(d_all_v_cache));
    CHECK(cudaFree(d_mask));
    CHECK(cudaFree(d_padding_offset));
    CHECK(cudaFree(d_history_len));
    CHECK(cudaFree(d_input_len));
    CHECK(cudaFree(d_ctx_len));
    CHECK(cudaFree(d_output_norm_weight));

    delete allocator;
    delete cublas_wrapper;

    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);

    return 0;
}
