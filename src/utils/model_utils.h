#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "src/models/basemodel.h"
#include "src/models/llama/llama.h"
#include "src/utils/macro.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/models/llama/llama_params.h"
#include <nlohmann/json.hpp>

// All LLM models are created in the header file.
// Provided are two methods: one for real weight models,
// the other for dummy weight models for functionality.
namespace llm {
    template<typename T>
    BaseModel *createModelWithName(const std::string &model_name) {
        LLM_CHECK_WITH_INFO(model_name == "llama", "Currently, only llama models are supported!");

        // Read configuration from JSON file
        std::ifstream config_file("/home/tourist/AI-HPC Projects/llm_inference_engine/src/models/llama/llama_config.json");
        nlohmann::json config;
        config_file >> config;

        int head_num = config["head_num"];
        int kv_head_num = config["kv_head_num"];
        int head_size = config["head_size"];
        int inter_size = config["inter_size"];
        int num_layers = config["num_layers"];
        int max_seq_len = config["max_seq_len"];
        int vocab_size = config["vocab_size"];
        int hidden_units = config["hidden_units"];
        int q_hidden_units = config["q_hidden_units"];
        bool attn_bias = config["attn_bias"];

        LlamaAttentionStaticParams attn_static_params;
        attn_static_params.rotary_embedding_dim = config["rotary_embedding_dim"];
        attn_static_params.rotary_embedding_base = config["rotary_embedding_base"];
        attn_static_params.max_position_embeddings = config["max_position_embeddings"];
        attn_static_params.use_dynamic_ntk = config["use_dynamic_ntk"];

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

        auto model = std::make_unique<LlamaModel<T>>(
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
    BaseModel *createDummyLLMModel(const std::string &tokenizer_file) {
        auto model = std::unique_ptr<BaseModel>(createModelWithName<T>("llama"));
        model->loadTokenizer(tokenizer_file);
        model->loadWeightsFromDummy();
        return model.release(); // Release ownership to return raw pointer
    }

    template<typename T>
    BaseModel *createRealLLMModel(const std::string &model_dir, const std::string &tokenizer_file) {
        auto model = std::unique_ptr<BaseModel>(createModelWithName<T>("llama"));
        std::cout << "Start creating model..." << std::endl;
        model->loadTokenizer(tokenizer_file);
        model->loadWeights(model_dir);
        std::cout << "Finish creating model..." << std::endl;
        return model.release(); // Release ownership to return raw pointer
    }

} // namespace llm
