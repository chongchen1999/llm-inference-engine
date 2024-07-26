#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "src/models/basemodel.h"
#include "src/models/llama/llama.h"
#include "src/utils/macro.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/models/llama/llama_params.h"

// All LLM models are created in the header file.
// Provided are two methods: one for real weight models,
// the other for dummy weight models for functionality.
namespace llm {
    template<typename T>
    BaseModel* createModelWithName(const std::string& model_name) {
        LLM_CHECK_WITH_INFO(model_name == "llama", "Currently, only llama models are supported!");

        const int head_num = 32;
        const int kv_head_num = 32;
        const int head_size = 128;
        const int inter_size = 11008;
        const int num_layers = 32;
        const int max_seq_len = 64;
        const int vocab_size = 32000;
        const int hidden_units = (head_num + 2 * kv_head_num) * head_size;
        const int q_hidden_units = head_num * head_size;
        const bool attn_bias = false;

        LlamaAttentionStaticParams attn_static_params;
        attn_static_params.rotary_embedding_dim = 128;
        attn_static_params.rotary_embedding_base = 10000;
        attn_static_params.max_position_embeddings = 4096;
        attn_static_params.use_dynamic_ntk = false; // true for dynamic scaling rope

        cublasHandle_t cublas_handle;
        cublasLtHandle_t cublaslt_handle;
        cudaStream_t stream;

        cublasCreate(&cublas_handle);
        cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

        auto cublas_wrapper = std::make_unique<cublasWrapper>(cublas_handle, cublaslt_handle);
        cublas_wrapper->setFP32GemmConfig();

        auto allocator = std::make_unique<CudaAllocator>();

        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, 0);

        auto model = std::make_unique<Llama<T>>(
            head_num,
            kv_head_num,
            head_size,
            inter_size,
            num_layers,
            vocab_size,
            attn_static_params,
            max_seq_len,
            stream,
            cublas_wrapper.get(),
            allocator.get(),
            &device_prop
        );

        return model.release(); // Release ownership to return raw pointer
    }

    template<typename T>
    BaseModel *createDummyLLMModel(const std::string& tokenizer_file) {
        auto model = std::unique_ptr<BaseModel>(createModelWithName<T>("llama"));
        model->loadTokenizer(tokenizer_file);
        model->loadWeightsFromDummy();
        return model.release(); // Release ownership to return raw pointer
    }

    template<typename T>
    BaseModel *createRealLLMModel(const std::string& model_dir, const std::string& tokenizer_file) {
        auto model = std::unique_ptr<BaseModel>(createModelWithName<T>("llama"));
        std::cout << "Start creating model..." << std::endl;
        model->loadTokenizer(tokenizer_file);
        model->loadWeights(model_dir);
        std::cout << "Finish creating model..." << std::endl;
        return model.release(); // Release ownership to return raw pointer
    }

} // namespace llm
