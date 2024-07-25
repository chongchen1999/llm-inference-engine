#include <math.h>
#include "src/utils/macro.h"
// #include "src/utils/debug_utils.h"
#include "src/layers/context_attention.h"

// Note: In the layers folder, many operations are followed by `DeviceSyncAndCheckCudaError();`. 
// You can manually remove them or add conditional compilation as shown in lesson30.

template<typename T>
LlamaContextAttentionLayer<T>::LlamaContextAttentionLayer(
    int head_num,
    int kv_head_num,
    int head_size,
    LlamaAttentionStaticParams attn_params,
    cudaStream_t stream,
    cublasWrapper *cublas_wrapper,
    BaseAllocator *allocator
) :
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    stream(stream),
    cublas_wrapper(cublas_wrapper),
    allocator(allocator),
    hidden_units(head_num * head_size),
    attn_static_params(attn_params),
    // TODO: Check if kv_head_num is divisible by head_num
    q_head_per_kv(head_num / kv_head_num),
    scale(1.0f / sqrt(static_cast<float>(head_size))) {}

template<typename T>
void LlamaContextAttentionLayer<T>::allocateMemoryForForward(
    LlamaAttentionDynamicParams *params
) {
    const int batch_size = params->batch_size;
    const int num_tokens = params->num_tokens;
    const int max_q_len = params->max_q_len;
    const int max_k_len = params->max_k_len;
    const DataType type = getTensorType<T>();
    const int qkv_head_num = head_num + 2 * kv_head_num;

    // For qkv linear and bias rope
    qkv_buf_without_padding = new TensorWrapper<T>(
        Device::GPU, type, {num_tokens, qkv_head_num, head_size}
    );
    q_buf_with_padding = new TensorWrapper<T>(
        Device::GPU, type, {batch_size, head_num, max_q_len, head_size}
    );
    k_buf_with_padding = new TensorWrapper<T>(
        Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size}
    );
    v_buf_with_padding = new TensorWrapper<T>(
        Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size}
    );

    // For transpose kv cache
    k_cache_buf = new TensorWrapper<T>(
        Device::GPU, type, {batch_size, head_num, max_k_len, head_size}
    );
    v_cache_buf = new TensorWrapper<T>(
        Device::GPU, type, {batch_size, head_num, max_k_len, head_size}
    );

    // For q*k and softmax
    qk_buf = new TensorWrapper<T>(
        Device::GPU, type, {batch_size, head_num, max_q_len, max_k_len}
    );

    // qk * v
    qkv_buf_with_padding = new TensorWrapper<T>(
        Device::GPU, type, {batch_size, head_num, max_q_len, head_size}
    );

    // Remove padding
    qkv_buf_without_padding_output = new TensorWrapper<T>(
        Device::GPU, type, {num_tokens, head_num, head_size}
    );

    allocator->malloc(
        &qkv_buf_without_padding->data,
        sizeof(T) * num_tokens * qkv_head_num * head_size,
        false
    );

    T *qkv_buf_with_padding_ptr;
    allocator->malloc(
        &qkv_buf_with_padding_ptr,
        sizeof(T) * qkv_head_num * batch_size * max_q_len * head_size,
        false
    );

    const int batch_stride = head_num * batch_size * max_q_len * head_size;
    q_buf_with_padding->data = qkv_buf_with_padding_ptr;
    k_buf_with_padding->data = qkv_buf_with_padding_ptr + head_num * batch_stride;
    v_buf_with_padding->data = qkv_buf_with_padding_ptr + (head_num + kv_head_num) * batch_stride;

    T *kv_cache_buf_ptr;
    allocator->malloc(
        &kv_cache_buf_ptr,
        2 * sizeof(T) * batch_size * head_num * max_k_len * head_size,
        false
    );

    k_cache_buf->data = kv_cache_buf_ptr;
    v_cache_buf->data = kv_cache_buf_ptr + batch_size * head_num * max_k_len * head_size;

    allocator->malloc(
        &qk_buf->data,
        sizeof(T) * batch_size * head_num * max_q_len * max_k_len,
        false
    );

    allocator->malloc(
        &qkv_buf_with_padding->data,
        sizeof(T) * batch_size * max_q_len * head_num * head_size,
        false
    );

    allocator->malloc(
        &qkv_buf_without_padding->data,
        sizeof(T) * num_tokens * head_num * head_size,
        false
    );
}

template<typename T>
void LlamaContextAttentionLayer<T>::freeBuf() {
    allocator->free(qkv_buf_without_padding->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(q_buf_with_padding->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(k_cache_buf->data);
    DeviceSyncAndCheckCudaError();

    // Note: No need to free v_cache_buf because it is included in k_cache_buf->data
    allocator->free(qk_buf->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(qkv_buf_with_padding->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(qkv_buf_without_padding_output->data);
}

template<typename T>
void LlamaContextAttentionLayer<T>::forward(
    TensorMap *inputs,
    TensorMap *outputs,
    LlamaAttentionWeights<T> *weights,
    LlamaAttentionDynamicParams *params,
    LlamaAttentionStaticParams *static_params
) {
    // Allocate intermediate buffers for the layer forward pass
    allocateMemoryForForward(params);

    // 1. qkv linear
    // Shape: [num_tokens, qhiddenunits] * [qhiddenunits, hiddenunits]
    Tensor *attention_input = (*inputs)["attention_input"];
    launchLinearGemm(
        attention_input->wrap<T>(),
        weights->qkv,
        qkv_buf_without_padding,
        cublas_wrapper,
        false,
        true
    );
    DeviceSyncAndCheckCudaError();

    // 2. qkv bias and rope and padding
    // Shape: [num_tokens, hiddenunits] => {batch_size, q(kv)head_num, max_q_len, head_size}
    // Note: qkv bias does not exist in LLaMA
    Tensor *padding_offset = (*inputs)["padding_offset"];
    Tensor *history_length = (*inputs)["history_length"];
    Tensor *input_length = (*inputs)["input_length"];
    Tensor *layer_id = (*inputs)["layer_id"]; // On CPU
    launchFusedQKVAddBiasAndTransposeAndRope(
        q_buf_with_padding,
        k_buf_with_padding,
        v_buf_with_padding,
        qkv_buf_without_padding,
        weights->qkv,
        padding_offset->wrap<int>(),
        history_length->wrap<int>(),
        input_length->wrap<int>(),
        static_params
    );

#ifndef PERF
    DeviceSyncAndCheckCudaError();
#else
#endif

#ifdef SAVE_DATA
    save_tensor(q_buf_with_padding, "q_buf_after_rope.bin", layer_id->as<int>());
#else
#endif

    // 3. Concatenate past KV cache
    // Shape: {batch_size, kv_head_num, max_q_len, headsize} => 
    // {num_layer, batch, maxseqlen[cumsum_seq_len:cumsum_seq_len+cur_seq_len], hidden_units_}
    Tensor *all_k_cache = (*outputs)["all_k_cache"];
    Tensor *all_v_cache = (*outputs)["all_v_cache"];
    launchConcatKVCache(
        k_buf_with_padding,
        v_buf_with_padding,
        layer_id->wrap<int>(),
        input_length->wrap<int>(),
        history_length->wrap<int>(),
        all_k_cache->wrap<T>(),
        all_v_cache->wrap<T>()
    );
    DeviceSyncAndCheckCudaError();

    // 4. MHA/MQA/GQA part: Reduce KV cache size to [num_layer, bs, kv head num, max_seq_len, head size]
    // 0. KV repeat/broadcast to adapt batchgemm shape requirement ([bs, head num, seqlen, head size])
    // Shape: [num_layer, bs, kv head num, max_seq_len, head size] => [bs, q head num, max_k_len, head size]
    Tensor *context_length = (*inputs)["context_length"];
    launchRepeatKVCache(
        all_k_cache->wrap<T>(),
        all_v_cache->wrap<T>(),
        context_length->wrap<int>(),
        layer_id->wrap<int>(),
        k_cache_buf,
        v_cache_buf
    );
    DeviceSyncAndCheckCudaError();

#ifdef SAVE_DATA
    save_tensor(k_cache_buf, "k_buf_after_repeat.bin", layer_id->as<int>());
#else
#endif

    // 1. qk
    // Shape: [bs, qhead, qlen, headsize] * [bs, qhead, klen, headsize] => [bs, head, qlen, klen]
    launchLinearStridedBatchGemm(
        q_buf_with_padding,
        k_cache_buf,
        qk_buf,
        cublas_wrapper,
        false,
        true
    );
    DeviceSyncAndCheckCudaError();

    // 2. Scale + mask + softmax
    Tensor *attention_mask = (*inputs)["attention_mask"];
    launchFusedScaleMaskAndSoftmax(
        qk_buf,
        attention_mask->wrap<T>(),
        qk_buf,
        scale
    );
    DeviceSyncAndCheckCudaError();

    // 3. qk * v
    // Shape: [bs, head, qlen, klen] => [bs, head, qlen, headsize]
    launchLinearStridedBatchGemm(
        qk_buf,
        v_cache_buf,
        qkv_buf_with_padding,
        cublas_wrapper,
        false,
        false
    );
    DeviceSyncAndCheckCudaError();

#ifdef SAVE_DATA
    save_tensor(qkv_buf_with_padding, "qk_v_buf_after_bmm.bin", layer_id->as<int>());
#else
#endif

    // 4. Transpose + reshape
    // Shape: [bs, head, seqlen, headsize] => [bs, seqlen, head, headsize] => [numtokens, hiddenunits] + remove padding
    launchFusedTransposeAndRemovePadding(
        qkv_buf_with_padding,
        padding_offset->wrap<int>(),
        qkv_buf_without_padding_output
    );
    DeviceSyncAndCheckCudaError();

    // 5. Output linear
    // Shape: [numtokens, hiddenunits] => [numtokens, hiddenunits]
    Tensor *attention_output = (*outputs)["attention_output"];
    launchLinearGemm(
        qkv_buf_without_padding_output,
        weights->output,
        attention_output->wrap<T>(),
        cublas_wrapper,
        false,
        true
    );

#ifdef SAVE_DATA
    save_tensor(attention_output->as<T>(), "out_linear_output.bin", layer_id->as<int>());
#else
#endif

    DeviceSyncAndCheckCudaError();
    freeBuf();
}

template class LlamaContextAttentionLayer<float>;
template class LlamaContextAttentionLayer<half>;
