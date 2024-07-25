#include <iostream>
#include "src/layers/includes/ffn.h"
// #include "src/utils/debug_utils.h"
// (RussWong) note: layers文件夹下，很多操作后面我都加了 `DeviceSyncAndCheckCudaError();`，大家可手动删除或者按照 lesson30 所示添加条件编译代码

template<typename T>
LlamaFFNLayer<T>::LlamaFFNLayer(
    int head_num,
    int head_size,
    int inter_size,
    cudaStream_t stream,
    cublasWrapper *cublas_wrapper,
    BaseAllocator *allocator
) : 
    head_num(head_num),
    head_size(head_size),
    inter_size(inter_size),
    stream(stream),
    cublas_wrapper(cublas_wrapper),
    allocator(allocator),
    hidden_units(head_num * head_size) {}

template<typename T>
void LlamaFFNLayer<T>::allocateMemoryForForward(
    LLaMAAttentionDynParams *params
) {
    int num_tokens = params->num_tokens;
    DataType type = getTensorType<T>();

    swiglu_input = new TensorWrapper<T>(
        Device::GPU,
        type,
        { num_tokens, 2, inter_size }
    );

    down_proj_input = new TensorWrapper<T>(
        Device::GPU,
        type,
        { num_tokens, inter_size }
    );

    allocator->malloc(
        &swiglu_input->data,
        sizeof(T) * num_tokens * 2 * inter_size,
        false
    );

    allocator->malloc(
        &down_proj_input->data,
        sizeof(T) * num_tokens * inter_size,
        false
    );
}

template<typename T>
void LlamaFFNLayer<T>::allocateMemoryForForward(int batch_size) {
    DataType type = getTensorType<T>();

    swiglu_input = new TensorWrapper<T>(
        Device::GPU,
        type,
        { batch_size, 2, inter_size }
    );

    down_proj_input = new TensorWrapper<T>(
        Device::GPU,
        type,
        { batch_size, inter_size }
    );

    allocator->malloc(
        &swiglu_input->data,
        sizeof(T) * batch_size * 2 * inter_size,
        false
    );

    allocator->malloc(
        &down_proj_input->data,
        sizeof(T) * batch_size * inter_size,
        false
    );
}

template<typename T>
void LlamaFFNLayer<T>::freeBuf() {
    allocator->free(swiglu_input->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(down_proj_input->data);
    DeviceSyncAndCheckCudaError();
}

template<typename T>
void LlamaFFNLayer<T>::forward(
    TensorMap *inputs,
    TensorMap *outputs,
    LlamaAttentionWeights<T> *weights,
    LlamaAttentionDynamicParams *params
) {
    if (params->num_tokens > 0) {
        allocateMemoryForForward(params);
    } else {
        allocateMemoryForForward(params->batch_size);
    }

    Tensor *ffn_input = inputs->at("ffn_input");
    Tensor *ffn_output = outputs->at("ffn_output");

    ++count;
    bool is_ctx = params->is_ctx;

#ifdef SAVE_DATA
    saveTensor(ffn_input->as<T>(), "ffn_input.bin", count);
#else
#endif

    // 1. Fused Gate-Up Projection
    launchLinearGemm(
        ffn_input->wrap<T>(),
        weights->gateAndup,
        swiglu_input,
        cublas_wrapper,
        false,
        true
    );
    DeviceSyncAndCheckCudaError();

#ifdef SAVE_DATA
    saveTensor(swiglu_input, "swiglu_input.bin", count);
#else
#endif

    // 2. SwiGLU Activation
    launchSiluAndMul(swiglu_input, down_proj_input);
    DeviceSyncAndCheckCudaError();

#ifdef SAVE_DATA
    saveTensor(down_proj_input, "down_proj_input.bin", count);
#else
#endif

    // 3. Down Projection
    launchLinearGemm(
        down_proj_input,
        weights->down,
        ffn_output->wrap<T>(),
        cublas_wrapper,
        false,
        true
    );
    DeviceSyncAndCheckCudaError();

    this->freeBuf();
}

// Explicit template instantiations
template class LlamaFFNLayer<float>;
template class LlamaFFNLayer<half>;
