#include <iostream>
#include "../utils/output_utils.h"
#include "includes/ffn.h"
#include "../utils/debug_utils.h"

template<typename T>
LlamaFFNLayer<T>::LlamaFFNLayer(
    int head_num,
    int head_size,
    int intermediate_size,
    cudaStream_t stream,
    CublasWrapper *cublas_wrapper,
    BaseAllocator *allocator
) : 
    head_num(head_num),
    head_size(head_size),
    intermediate_size(intermediate_size),
    stream(stream),
    cublas_wrapper(cublas_wrapper),
    allocator(allocator),
    hidden_units(head_num * head_size) {}

template<typename T>
void LlamaFFNLayer<T>::allocateMemory(LlamaAttentionDynamicParams *dynamic_params) {
    const int num_tokens = dynamic_params->num_tokens;
    DataType type = getTensorType<T>();

    swiglu_input = new TensorWrapper<T>(
        Device::GPU, type, 
        std::vector<int>{num_tokens, 2, intermediate_size}
    );

    down_proj_input = new TensorWrapper<T>(
        Device::GPU, type, 
        std::vector<int>{num_tokens, intermediate_size}
    );

    allocator->malloc(
        &swiglu_input->data,
        sizeof(T) * num_tokens * 2 * intermediate_size,
        false
    );

    allocator->malloc(
        &down_proj_input->data,
        sizeof(T) * num_tokens * intermediate_size,
        false
    );
}

template<typename T>
void LlamaFFNLayer<T>::allocateMemory(const int &batch_size) {
    DataType type = getTensorType<T>();

    swiglu_input = new TensorWrapper<T>(Device::GPU, type, {batch_size, 2, intermediate_size});
    down_proj_input = new TensorWrapper<T>(Device::GPU, type, {batch_size, intermediate_size});

    allocator->malloc(
        &swiglu_input->data,
        sizeof(T) * batch_size * 2 * intermediate_size,
        false
    );

    allocator->malloc(
        &down_proj_input->data,
        sizeof(T) * batch_size * intermediate_size,
        false
    );
}

template<typename T>
void LlamaFFNLayer<T>::freeBuf() {
    allocator->free(swiglu_input->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(down_proj_input->data);
    DeviceSyncAndCheckCudaError();

    delete swiglu_input;
    delete down_proj_input;
}

template<typename T>
void LlamaFFNLayer<T>::forward(
    TensorMap *inputs,
    TensorMap *outputs,
    LlamaFFNWeights<T> *weights,
    LlamaAttentionDynamicParams *dynamic_params
) {
    // printf("got in!\n");
    if (dynamic_params->num_tokens > 0) {
        allocateMemory(dynamic_params);
    } else {
        allocateMemory(dynamic_params->batch_size);
    }

    Tensor *ffn_input = inputs->at("ffn_input");
    Tensor *ffn_output = outputs->at("ffn_output");

    ++count;
    const bool is_context = dynamic_params->is_context;

    #ifdef SAVE_DATA
        saveTensor(ffn_input->as<T>(), "ffn_input.bin", count);
    #else
    #endif
    /*printf("is_transposed %d\n", weights->gate_and_up.is_transposed);
    print_tensor(ffn_input);
    print_weight(&weights->gate_and_up);*/

    // 1. Fused Gate-Up Projection
    launchLinearGemm(
        ffn_input->wrap<T>(),
        &weights->gate_and_up,
        swiglu_input,
        cublas_wrapper,
        false,
        weights->gate_and_up.is_transposed
    );
    DeviceSyncAndCheckCudaError();
    // printf("gate_and_up done!\n");

    #ifdef SAVE_DATA
        saveTensor(swiglu_input, "swiglu_input.bin", count);
    #else
    #endif

    // 2. SwiGLU Activation
    launchSiluAndMul(swiglu_input, down_proj_input);
    DeviceSyncAndCheckCudaError();
    // printf("silu done!\n");

    #ifdef SAVE_DATA
        saveTensor(down_proj_input, "down_proj_input.bin", count);
    #else
    #endif

    // 3. Down Projection
    launchLinearGemm(
        down_proj_input,
        &weights->down,
        ffn_output->wrap<T>(),
        cublas_wrapper,
        false,
        weights->down.is_transposed
    );
    DeviceSyncAndCheckCudaError();
    // printf("down done!\n");

    this->freeBuf();
}

// Explicit template instantiations
template class LlamaFFNLayer<float>;
template class LlamaFFNLayer<half>;
