#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdlib>
#include "../../src/layers/includes/context_decoder.h"
#include "../../src/utils/macro.h"
#include "../../src/models/tokenizer.h"
#include "../../src/kernels/includes/input_embedding.cuh"
#include "../../src/weights/includes/embedding_weights.h"
#include "../../src/memory/memory_deleter.cuh"

int main(int argc, char **argv) {
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;

    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    CublasWrapper *cublas_wrapper = new CublasWrapper(cublas_handle, cublaslt_handle);
    CudaAllocator *allocator = new CudaAllocator();

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

    int *h_input_ids = new int[res.size()];
    for (int i = 0; i < res.size(); ++i) {
        h_input_ids[i] = res[i];
    }

    const int cur_context_length = res.size();
    const int cur_input_length = res.size(); // Actual input length

    LlamaAttentionDynamicParams attention_dynamic_params;
    attention_dynamic_params.batch_size = 1;
    attention_dynamic_params.num_tokens = cur_input_length;
    attention_dynamic_params.max_q_len = attention_dynamic_params.num_tokens;
    attention_dynamic_params.max_k_len = cur_context_length;

    std::string ret_string; // Response tokens string

    TensorWrapper<int> *input_ids = new TensorWrapper<int>(
        Device::GPU, getTensorType<int>(), 
        std::vector<int>{cur_input_length}
    );
    allocator->malloc(&input_ids->data, sizeof(int) * cur_input_length, false);
    CHECK(cudaMemcpy(input_ids->data, h_input_ids, sizeof(int) * cur_input_length, cudaMemcpyHostToDevice));

    TensorWrapper<float> *decoder_input = new TensorWrapper<float>(
        Device::GPU, getTensorType<float>(), 
        std::vector<int>{attention_dynamic_params.num_tokens, q_hidden_units}
    );
    allocator->malloc(
        &decoder_input->data,
        sizeof(float) * attention_dynamic_params.num_tokens * q_hidden_units,
        false
    );

    float *embedding = new float[vocab_size * embedding_size];
    for (int i = 0; i < vocab_size * embedding_size; ++i) {
        embedding[i] = static_cast<float>(rand() % 100) / 100000.0f;
    }

    float *d_embedding;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_embedding), sizeof(float) * vocab_size * embedding_size));
    CHECK(cudaMemcpy(d_embedding, embedding, sizeof(float) * vocab_size * embedding_size, cudaMemcpyHostToDevice));

    EmbeddingWeight<float> embed_table;
    WeightType wtype = getWeightType<float>();
    embed_table.shape = {vocab_size, embedding_size};
    embed_table.type = wtype;
    embed_table.data = d_embedding;

    launchInputEmbedding(input_ids, decoder_input, &embed_table);

    float *d_decoder_output;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_decoder_output), sizeof(float) * q_hidden_units * attention_dynamic_params.num_tokens));

    const int mask_size = attention_dynamic_params.batch_size * attention_dynamic_params.max_q_len * attention_dynamic_params.max_k_len;
    float *h_mask = new float[mask_size];
    float *d_mask;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_mask), sizeof(float) * mask_size));
    std::fill(h_mask, h_mask + mask_size, 1.0f);

    const int kv_cache_size = num_layers * attention_dynamic_params.batch_size * kv_head_num * max_seq_len * head_size;
    float *h_all_k_cache = new float[kv_cache_size];
    float *d_all_k_cache;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_all_k_cache), sizeof(float) * kv_cache_size));

    float *h_all_v_cache = new float[kv_cache_size];
    float *d_all_v_cache;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_all_v_cache), sizeof(float) * kv_cache_size));

    for (int i = 0; i < kv_cache_size; ++i) {
        h_all_k_cache[i] = static_cast<float>(rand() % 100) / 100000.0f;
        h_all_v_cache[i] = static_cast<float>(rand() % 100) / 100000.0f;
    }

    int *h_padding_offset = new int[attention_dynamic_params.num_tokens];
    int *d_padding_offset;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_padding_offset), sizeof(int) * attention_dynamic_params.num_tokens));
    std::fill(h_padding_offset, h_padding_offset + attention_dynamic_params.num_tokens, 0.0f);

    int *h_history_len = new int[attention_dynamic_params.batch_size];
    int *d_history_len;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_history_len), sizeof(int) * attention_dynamic_params.batch_size));

    int *h_input_len = new int[attention_dynamic_params.batch_size];
    int *d_input_len;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_input_len), sizeof(int) * attention_dynamic_params.batch_size));

    int *h_context_len = new int[attention_dynamic_params.batch_size];
    int *d_context_len;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_context_len), sizeof(int) * attention_dynamic_params.batch_size));

    for (int i = 0; i < attention_dynamic_params.batch_size; ++i) {
        h_history_len[i] = 0; // For kv cache cumsum seqlen and rope's timestep compute
        h_input_len[i] = cur_input_length;
        h_context_len[i] = cur_context_length;
    }

    float *h_output_norm_weight = new float[q_hidden_units];
    float *d_output_norm_weight;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output_norm_weight), sizeof(float) * q_hidden_units));
    for (int i = 0; i < q_hidden_units; ++i) {
        h_output_norm_weight[i] = static_cast<float>(rand() % 100) / 100000.0f;
    }

    // Copy data to device
    CHECK(cudaMemcpy(d_all_k_cache, h_all_k_cache, sizeof(float) * kv_cache_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_all_v_cache, h_all_v_cache, sizeof(float) * kv_cache_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_padding_offset, h_padding_offset, sizeof(int) * attention_dynamic_params.num_tokens, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_history_len, h_history_len, sizeof(int) * attention_dynamic_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_context_len, h_context_len, sizeof(int) * attention_dynamic_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_input_len, h_input_len, sizeof(int) * attention_dynamic_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mask, h_mask, sizeof(float) * mask_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_output_norm_weight, h_output_norm_weight, sizeof(float) * q_hidden_units, cudaMemcpyHostToDevice));

    int layer_id = 0;
    DataType type = getTensorType<float>();
    auto layer_weights = new std::vector<LlamaLayerWeight<float> *>;
    layer_weights->reserve(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        auto layer_weight = new LlamaLayerWeight<float>(head_num, kv_head_num, head_size, intermediate_size, wtype, false);
        layer_weight->loadWeightsFromFile();
        layer_weights->push_back(std::move(layer_weight));
    }

    TensorWrapper<int> *padding_offset = new TensorWrapper<int>(
        Device::GPU, getTensorType<int>(), 
        std::vector<int>{attention_dynamic_params.num_tokens}, d_padding_offset
    );

    TensorWrapper<int> *history_length = new TensorWrapper<int>(
        Device::GPU, getTensorType<int>(), 
        std::vector<int>{attention_dynamic_params.batch_size}, d_history_len
    );

    TensorWrapper<int> *input_length = new TensorWrapper<int>(
        Device::GPU, getTensorType<int>(), 
        std::vector<int>{attention_dynamic_params.batch_size}, d_input_len
    );

    TensorWrapper<int> *layer = new TensorWrapper<int>(
        Device::CPU, getTensorType<int>(), 
        std::vector<int>{1}, &layer_id
    );

    TensorWrapper<int> *context_length = new TensorWrapper<int>(
        Device::GPU, getTensorType<int>(), 
        std::vector<int>{attention_dynamic_params.batch_size}, d_context_len
    );

    TensorWrapper<float> *attention_mask = new TensorWrapper<float>(
        Device::GPU, getTensorType<float>(), 
        std::vector<int>{attention_dynamic_params.batch_size, attention_dynamic_params.max_q_len, attention_dynamic_params.max_k_len}, 
        d_mask
    );

    TensorWrapper<float> *decoder_output = new TensorWrapper<float>(
        Device::GPU, getTensorType<float>(), 
        std::vector<int>{attention_dynamic_params.num_tokens, q_hidden_units}, 
        d_decoder_output
    );

    TensorWrapper<float> *all_k_cache = new TensorWrapper<float>(
        Device::GPU, getTensorType<float>(), 
        std::vector<int>{num_layers, attention_dynamic_params.batch_size, kv_head_num, max_seq_len, head_size}, 
        d_all_k_cache
    );

    TensorWrapper<float> *all_v_cache = new TensorWrapper<float>(
        Device::GPU, getTensorType<float>(), 
        std::vector<int>{num_layers, attention_dynamic_params.batch_size, kv_head_num, max_seq_len, head_size}, 
        d_all_v_cache
    );

    TensorWrapper<float> *output_norm_weight = new TensorWrapper<float>(
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
        {"decoder_input", decoder_input},
        {"history_length", history_length},
        {"input_length", input_length},
        {"context_length", context_length},
        {"output_norm_weight", output_norm_weight},
        {"layer_id", layer}
    };

    TensorMap decoder_outputs{
        {"decoder_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };


    LlamaContextDecoder<float> *context_decoder = new LlamaContextDecoder<float>(
        head_num,
        kv_head_num,
        head_size,
        intermediate_size,
        num_layers,
        &attn_static_params,
        rmsnorm_eps,
        stream,
        cublas_wrapper,
        allocator
    );
    std::cout << "ready for forwarding!" << std::endl;

    context_decoder->forward(
        &decoder_inputs, 
        layer_weights, 
        &decoder_outputs, 
        &attention_dynamic_params
    );
    DeviceSyncAndCheckCudaError();

    // Deallocate host memory
    deallocate(h_input_ids, "new[]");
    deallocate(embedding, "new[]");
    deallocate(h_all_k_cache, "new[]");
    deallocate(h_all_v_cache, "new[]");
    deallocate(h_mask, "new[]");
    deallocate(h_padding_offset, "new[]");
    deallocate(h_history_len, "new[]");
    deallocate(h_input_len, "new[]");
    deallocate(h_context_len, "new[]");
    deallocate(h_output_norm_weight, "new[]");

    // Deallocate device memory
    deallocate(d_embedding, "cudaMalloc");
    deallocate(d_decoder_output, "cudaMalloc");
    deallocate(d_mask, "cudaMalloc");
    deallocate(d_all_k_cache, "cudaMalloc");
    deallocate(d_all_v_cache, "cudaMalloc");
    deallocate(d_padding_offset, "cudaMalloc");
    deallocate(d_history_len, "cudaMalloc");
    deallocate(d_input_len, "cudaMalloc");
    deallocate(d_context_len, "cudaMalloc");
    deallocate(d_output_norm_weight, "cudaMalloc");

    // Deallocate dynamically allocated objects
    deallocate(cublas_wrapper, "new");
    deallocate(allocator, "new");
    deallocate(input_ids, "new");
    deallocate(decoder_input, "new");
    deallocate(padding_offset, "new");
    deallocate(history_length, "new");
    deallocate(input_length, "new");
    deallocate(layer, "new");
    deallocate(context_length, "new");
    deallocate(attention_mask, "new");
    deallocate(decoder_output, "new");
    deallocate(all_k_cache, "new");
    deallocate(all_v_cache, "new");
    deallocate(output_norm_weight, "new");
    deallocate(context_decoder, "new");

    // Clean up layer_weights
    for (auto layer_weight : *layer_weights) {
        deallocate(layer_weight, "new");
    }
    deallocate(layer_weights, "new");

    return 0;
}
