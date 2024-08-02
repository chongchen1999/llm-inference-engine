#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <memory>
#include "../../src/layers/includes/context_decoder.h"
#include "../../src/utils/macro.h"
#include "../../src/models/tokenizer.h"
#include "../../src/kernels/includes/input_embedding.h"
#include "../../src/weights/includes/embedding_weights.h"

int main(int argc, char **argv) {
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;

    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    auto cublas_wrapper = std::make_unique<CublasWrapper>(cublas_handle, cublaslt_handle);
    auto allocator = std::make_unique<CudaAllocator>();

    constexpr int head_num = 8;
    constexpr int kv_head_num = 8;
    constexpr int head_size = 32;
    constexpr int intermediate_size = 11008;
    constexpr int num_layers = 8;
    constexpr int max_seq_len = 16;
    constexpr float rmsnorm_eps = 1e-6;
    const int vocab_size = 64;
    const int embedding_size = 64;

    const int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    const int q_hidden_units = head_num * head_size;

    LlamaAttentionStaticParams attn_static_params;
    attn_static_params.rotary_embedding_dim = 128;
    attn_static_params.rotary_embedding_base = 10000;
    attn_static_params.max_position_embeddings = 2048;
    attn_static_params.use_dynamic_ntk = false; // for dynamic scaling rope

    /*const std::string input = "how old are you";
    Tokenizer tokenizer;
    tokenizer.Initialize("/home/llama2-7b-tokenizer.bin");

    std::vector<int> res = tokenizer.Encode(input);
    std::cout << "Input IDs length is " << res.size() << "\n";*/

    std::vector<int> res = {5, 7, 12, 5, 4, 6, 55, 54, 27};

    auto h_input_ids_buf = std::make_unique<int[]>(res.size());
    for (int i = 0; i < res.size(); ++i) {
        h_input_ids_buf[i] = res[i];
    }

    const int cur_context_length = res.size();
    const int cur_input_length = res.size(); // Actual input length

    LlamaAttentionDynamicParams attention_dynamic_params;
    attention_dynamic_params.batch_size = 1;
    attention_dynamic_params.num_tokens = cur_input_length;
    attention_dynamic_params.max_q_len = attention_dynamic_params.num_tokens;
    attention_dynamic_params.max_k_len = cur_context_length;

    std::string ret_string; // Response tokens string

    auto input_ids = std::make_unique<TensorWrapper<int>>(
        Device::GPU, getTensorType<int>(), 
        std::vector<int>{cur_input_length}
    );
    allocator->malloc(&input_ids->data, sizeof(int) * cur_input_length, false);

    CHECK(cudaMemcpy(input_ids->data, h_input_ids_buf.get(), sizeof(int) * cur_input_length, cudaMemcpyHostToDevice));

    auto decoder_input = std::make_unique<TensorWrapper<float>>(
        Device::GPU, getTensorType<float>(), 
        std::vector<int>{attention_dynamic_params.num_tokens, q_hidden_units}
    );
    allocator->malloc(
        &decoder_input->data,
        sizeof(float) * attention_dynamic_params.num_tokens * q_hidden_units,
        false
    );

    auto embedding = std::make_unique<float[]>(vocab_size * embedding_size);
    for (int i = 0; i < vocab_size * embedding_size; ++i) {
        embedding[i] = rand() % 100 / 100000.0f;
    }

    float *d_embedding;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_embedding), sizeof(float) * vocab_size * embedding_size));
    CHECK(cudaMemcpy(d_embedding, embedding.get(), sizeof(float) * vocab_size * embedding_size, cudaMemcpyHostToDevice));

    EmbeddingWeight<float> embed_table;
    WeightType wtype = getWeightType<float>();
    embed_table.shape = {vocab_size, embedding_size};
    embed_table.type = wtype;
    embed_table.data = d_embedding;

    launchInputEmbedding(input_ids.get(), decoder_input.get(), &embed_table);

    float *d_decoder_output;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_decoder_output), sizeof(float) * q_hidden_units * attention_dynamic_params.num_tokens));

    auto h_mask = std::make_unique<float[]>(attention_dynamic_params.batch_size * attention_dynamic_params.max_q_len * attention_dynamic_params.max_k_len);
    float *d_mask;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_mask), sizeof(float) * attention_dynamic_params.batch_size * attention_dynamic_params.max_q_len * attention_dynamic_params.max_k_len));
    std::fill(h_mask.get(), h_mask.get() + attention_dynamic_params.batch_size * attention_dynamic_params.max_q_len * attention_dynamic_params.max_k_len, 1.0f);

    auto h_all_k_cache = std::make_unique<float[]>(num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size);
    float *d_all_k_cache;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_all_k_cache), sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size));
    auto h_all_v_cache = std::make_unique<float[]>(num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size);
    float *d_all_v_cache;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_all_v_cache), sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size));

    for (int i = 0; i < num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size; ++i) {
        h_all_k_cache[i] = rand() % 100 / 100000.0f;
        h_all_v_cache[i] = rand() % 100 / 100000.0f;
    }

    auto h_padding_offset = std::make_unique<int[]>(attention_dynamic_params.num_tokens);
    int *d_padding_offset;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_padding_offset), sizeof(int) * attention_dynamic_params.num_tokens));
    std::fill(h_padding_offset.get(), h_padding_offset.get() + attention_dynamic_params.num_tokens, 0);

    auto h_history_len = std::make_unique<int[]>(attention_dynamic_params.batch_size);
    int *d_history_len;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_history_len), sizeof(int) * attention_dynamic_params.batch_size));

    auto h_input_len = std::make_unique<int[]>(attention_dynamic_params.batch_size);
    int *d_input_len;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_input_len), sizeof(int) * attention_dynamic_params.batch_size));

    auto h_ctx_len = std::make_unique<int[]>(attention_dynamic_params.batch_size);
    int *d_ctx_len;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ctx_len), sizeof(int) * attention_dynamic_params.batch_size));

    for (int i = 0; i < attention_dynamic_params.batch_size; ++i) {
        h_history_len[i] = 0; // For kv cache cumsum seqlen and rope's timestep compute
        h_input_len[i] = cur_input_length;
        h_ctx_len[i] = cur_context_length;
    }

    auto h_output_norm_weight = std::make_unique<float[]>(q_hidden_units);
    float *d_output_norm_weight;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output_norm_weight), sizeof(float) * q_hidden_units));
    for (int i = 0; i < q_hidden_units; ++i) {
        h_output_norm_weight[i] = rand() % 100 / 100000.0f;
    }

    // Copy data to device
    CHECK(cudaMemcpy(d_all_k_cache, h_all_k_cache.get(), sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_all_v_cache, h_all_v_cache.get(), sizeof(float) * num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_padding_offset, h_padding_offset.get(), sizeof(int) * attention_dynamic_params.num_tokens, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_history_len, h_history_len.get(), sizeof(int) * attention_dynamic_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ctx_len, h_ctx_len.get(), sizeof(int) * attention_dynamic_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_input_len, h_input_len.get(), sizeof(int) * attention_dynamic_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mask, h_mask.get(), sizeof(float) * attention_dynamic_params.batch_size * attention_dynamic_params.max_q_len * attention_dynamic_params.max_k_len, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_output_norm_weight, h_output_norm_weight.get(), sizeof(float) * q_hidden_units, cudaMemcpyHostToDevice));

    int layer_id = 0;
    DataType type = getTensorType<float>();
    std::vector<std::unique_ptr<LlamaLayerWeight<float>>> layer_weights;
    layer_weights.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        auto layer_weight = std::make_unique<LlamaLayerWeight<float>>(
            head_num, kv_head_num, head_size, intermediate_size, wtype, false
        );
        layer_weight->loadWeightsFromFile();
        layer_weights.push_back(std::move(layer_weight));
    }

    auto padding_offset = std::make_unique<TensorWrapper<int>>(
        Device::GPU, getTensorType<int>(), 
        std::vector<int>{attention_dynamic_params.num_tokens}, d_padding_offset
    );
    auto history_length = std::make_unique<TensorWrapper<int>>(
        Device::GPU, getTensorType<int>(), 
        std::vector<int>{attention_dynamic_params.batch_size}, d_history_len
    );
    auto input_length = std::make_unique<TensorWrapper<int>>(
        Device::GPU, getTensorType<int>(), 
        std::vector<int>{attention_dynamic_params.batch_size}, d_input_len
    );
    auto layer = std::make_unique<TensorWrapper<int>>(
        Device::CPU, getTensorType<int>(), 
        std::vector<int>{1}, &layer_id
    );
    auto context_length = std::make_unique<TensorWrapper<int>>(
        Device::GPU, getTensorType<int>(), 
        std::vector<int>{attention_dynamic_params.batch_size}, d_ctx_len
    );
    auto attention_mask = std::make_unique<TensorWrapper<float>>(
        Device::GPU, getTensorType<float>(), 
        std::vector<int>{attention_dynamic_params.batch_size, attention_dynamic_params.max_q_len, attention_dynamic_params.max_k_len}, 
        d_mask
    );
    auto decoder_output = std::make_unique<TensorWrapper<float>>(
        Device::GPU, getTensorType<float>(), 
        std::vector<int>{attention_dynamic_params.num_tokens, q_hidden_units}, 
        d_decoder_output
    );
    auto all_k_cache = std::make_unique<TensorWrapper<float>>(
        Device::GPU, getTensorType<float>(), 
        std::vector<int>{num_layers, attention_dynamic_params.batch_size, kv_head_num, max_seq_len, head_size}, 
        d_all_k_cache
    );
    auto all_v_cache = std::make_unique<TensorWrapper<float>>(
        Device::GPU, getTensorType<float>(), 
        std::vector<int>{num_layers, attention_dynamic_params.batch_size, kv_head_num, max_seq_len, head_size}, 
        d_all_v_cache
    );
    auto output_norm_weight = std::make_unique<TensorWrapper<float>>(
        Device::GPU, getTensorType<float>(), 
        std::vector<int>{q_hidden_units}, 
        d_output_norm_weight
    );

    LLM_CHECK_WITH_INFO(decoder_input->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(history_length->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(layer->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(context_length->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(output_norm_weight->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(input_length->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");

    std::cout << "in context decoder example cpp: " << layer->deviceString() << std::endl;

    TensorMap decoder_inputs{
        {"decoder_input", decoder_input.get()},
        {"history_length", history_length.get()},
        {"input_length", input_length.get()},
        {"context_length", context_length.get()},
        {"output_norm_weight", output_norm_weight.get()},
        {"layer_id", layer.get()}
    };

    TensorMap decoder_outputs{
        {"decoder_output", decoder_output.get()},
        {"all_k_cache", all_k_cache.get()},
        {"all_v_cache", all_v_cache.get()}
    };


    auto ctx_decoder = std::make_unique<LlamaContextDecoder<float>>(
        head_num,
        kv_head_num,
        head_size,
        intermediate_size,
        num_layers,
        &attn_static_params,
        rmsnorm_eps,
        stream,
        std::move(cublas_wrapper),
        std::move(allocator)
    );

    ctx_decoder->forward(
        &decoder_inputs, 
        &layer_weights, 
        &decoder_outputs, 
        &attention_dynamic_params
    );

    cudaDeviceSynchronize();

    // Free GPU memory
    DeviceSyncAndCheckCudaError();
    cudaFree(d_all_k_cache);
    cudaFree(d_all_v_cache);
    cudaFree(d_padding_offset);
    cudaFree(d_history_len);
    cudaFree(d_ctx_len);
    cudaFree(d_input_len);
    cudaFree(d_mask); 
    cudaFree(d_output_norm_weight);
    cudaFree(d_embedding);

    return 0;
}
