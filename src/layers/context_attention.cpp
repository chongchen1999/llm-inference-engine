#include <math.h>
#include "../utils/macro.h"
#include "../utils/debug_utils.h"
#include "includes/context_attention.h"

// Note: In the layers folder, many operations are followed by `DeviceSyncAndCheckCudaError();`. 
// You can manually remove them or add conditional compilation as shown in lesson30.

template<typename T>
LlamaContextAttentionLayer<T>::LlamaContextAttentionLayer(
    int head_num,
    int kv_head_num,
    int head_size,
    LlamaAttentionStaticParams *attention_static_params,
    cudaStream_t stream,
    CublasWrapper *cublas_wrapper,
    BaseAllocator *allocator
) :
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    stream(stream),
    cublas_wrapper(cublas_wrapper),
    allocator(allocator),
    hidden_units(head_num * head_size),
    attention_static_params(attention_static_params),
    repeats_per_kv(head_num / kv_head_num),
    scale(1.0f / sqrt(static_cast<float>(head_size))) {
        LLM_CHECK_WITH_INFO(head_num % kv_head_num == 0, "kv_head_num must be a factor of head_num");
    }

template<typename T>
void LlamaContextAttentionLayer<T>::allocateMemory(
    LlamaAttentionDynamicParams *dynamic_params
) {
    const int batch_size = dynamic_params->batch_size;
    const int num_tokens = dynamic_params->num_tokens;
    const int max_q_len = dynamic_params->max_q_len;
    const int max_k_len = dynamic_params->max_k_len;
    const DataType type = getTensorType<T>();
    const int qkv_head_num = head_num + 2 * kv_head_num;

    // For qkv linear and bias rope
    lineared_qkv = new TensorWrapper<T>(Device::GPU, type, {num_tokens, qkv_head_num, head_size});
    padded_q = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, head_size});
    padded_k = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size});
    padded_v = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size});

    // For transpose kv cache
    k_cache = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});
    v_cache = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});

    // For q*k and softmax
    qkT = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, max_k_len});

    // qk * v
    padded_qkTv = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, head_size});

    // Remove padding
    transposed_unpadded_qkv = new TensorWrapper<T>(Device::GPU, type, {num_tokens, head_num, head_size});

    allocator->malloc(
        &lineared_qkv->data,
        sizeof(T) * num_tokens * qkv_head_num * head_size,
        false
    );

    allocator->malloc(
        &padded_q->data,
        sizeof(T) * qkv_head_num * batch_size * max_q_len * head_size,
        false
    );
    padded_k->data = padded_q->data + batch_size * max_q_len * head_num * head_size;
    padded_v->data = padded_k->data + batch_size * max_q_len * kv_head_num * head_size;

    allocator->malloc(
        &k_cache->data,
        2 * sizeof(T) * batch_size * head_num * max_k_len * head_size,
        false
    );
    v_cache->data = k_cache->data + batch_size * head_num * max_k_len * head_size;

    allocator->malloc(
        &qkT->data,
        sizeof(T) * batch_size * head_num * max_q_len * max_k_len,
        false
    );

    allocator->malloc(
        &padded_qkTv->data,
        sizeof(T) * batch_size * max_q_len * head_num * head_size,
        false
    );

    allocator->malloc(
        &transposed_unpadded_qkv->data,
        sizeof(T) * num_tokens * head_num * head_size,
        false
    );
}

template<typename T>
void LlamaContextAttentionLayer<T>::freeBuf() {
    allocator->free(lineared_qkv->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(padded_q->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(k_cache->data);
    DeviceSyncAndCheckCudaError();

    // No need to free v_cache because it is included in k_cache->data
    allocator->free(qkT->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(padded_qkTv->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(transposed_unpadded_qkv->data);
    DeviceSyncAndCheckCudaError();

    delete lineared_qkv;
    delete padded_q;
    delete padded_k;
    delete padded_v;
    delete k_cache;
    delete v_cache;
    delete qkT;
    delete padded_qkTv;
    delete transposed_unpadded_qkv;
    lineared_qkv = nullptr;
    padded_q = nullptr;
    padded_k = nullptr;
    padded_v = nullptr;
    k_cache = nullptr;
    v_cache = nullptr;
    qkT = nullptr;
    padded_qkTv = nullptr;
    transposed_unpadded_qkv = nullptr;
}

template<typename T>
void LlamaContextAttentionLayer<T>::forward(
    TensorMap *inputs,
    TensorMap *outputs,
    LlamaAttentionWeights<T> *weights,
    LlamaAttentionDynamicParams *dynamic_params,
    LlamaAttentionStaticParams *static_params
) {
    // printf("in!\n");
    allocateMemory(dynamic_params);
    // printf("allocation pass!\n");

    // 1. qkv linear
    // Shape: [num_tokens, qhiddenunits] * [qhiddenunits, hiddenunits]
    Tensor *attention_input = inputs->at("attention_input");
    launchLinearGemm(
        attention_input->wrap<T>(), //[num_tokens, qhiddenunits]
        &weights->qkv, // [qhiddenunits, hiddenunits]
        lineared_qkv, // [num_tokens, qkv_head_num, head_size]
        cublas_wrapper,
        false,
        weights->qkv.is_transposed
    );
    DeviceSyncAndCheckCudaError();
    // printf("qkv linear pass!\n");

    // 2. qkv add bias and rope and padding
    // Shape: [num_tokens, hiddenunits] => [batch_size, qkv_head_num, max_q_len, head_size]
    // Note: qkv bias does not exist in LLaMA
    Tensor *padding_offset = inputs->at("padding_offset");
    Tensor *history_length = inputs->at("history_length");
    Tensor *input_length = inputs->at("input_length");
    Tensor *layer_id = inputs->at("layer_id"); // On CPU
    launchFusedQKVAddBiasAndTransposeAndRope(
        padded_q,
        padded_k,
        padded_v,
        lineared_qkv,
        &weights->qkv,
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
        saveTensor(padded_q, "q_buf_after_rope.bin", layer_id->as<int>());
    #else
    #endif

    // printf("qkv add bias and rope pass!\n");

    // 3. Concatenate past KV cache
    // Shape: [batch_size, kv_head_num, max_q_len, headsize] => 
    // [num_layer, batch, maxseqlen[cumsum_seq_len:cumsum_seq_len+cur_seq_len], hidden_units_]
    Tensor *all_k_cache = outputs->at("all_k_cache");
    Tensor *all_v_cache = outputs->at("all_v_cache");
    launchConcatKVCache(
        padded_k,
        padded_v,
        layer_id->wrap<int>(),
        input_length->wrap<int>(),
        history_length->wrap<int>(),
        all_k_cache->wrap<T>(),
        all_v_cache->wrap<T>()
    );
    DeviceSyncAndCheckCudaError();
    printf("concat KV cache pass!\n");

    // 4. MHA/MQA/GQA part: Reduce KV cache size to [num_layer, bs, kv head num, max_seq_len, head size]
    // 0. KV repeat/broadcast to adapt batchgemm shape requirement ([bs, head num, seqlen, head size])
    // Shape: [num_layer, bs, kv head num, max_seq_len, head size] => [bs, q head num, max_k_len, head size]
    Tensor *context_length = inputs->at("context_length");
    launchRepeatKVCache(
        all_k_cache->wrap<T>(),
        all_v_cache->wrap<T>(),
        context_length->wrap<int>(),
        layer_id->wrap<int>(),
        k_cache,
        v_cache
    );
    DeviceSyncAndCheckCudaError();

    #ifdef SAVE_DATA
        saveTensor(k_cache, "k_buf_after_repeat.bin", layer_id->as<int>());
    #else
    #endif
    // printf("repeat KV cache pass!\n");

    // 5. qk
    // Shape: [bs, head_num, max_q_len, head_size] * [bs, head_num, max_k_len, head_size]T => 
    //        [bs, head_num, max_q_len, max_k_len]
    launchLinearStridedBatchGemm(
        padded_q,
        k_cache,
        qkT,
        cublas_wrapper,
        false,
        true
    );
    DeviceSyncAndCheckCudaError();
    // printf("qkT pass!");

    // 6. Scale + mask + softmax
    Tensor *attention_mask = inputs->at("attention_mask");
    launchFusedScaleMaskAndSoftmax(
        qkT,
        attention_mask->wrap<T>(),
        qkT,
        scale
    );
    DeviceSyncAndCheckCudaError();
    // printf("scale mask softmax pass!\n");

    // 7. qk * v
    // Shape: [bs, head_num, max_q_len, max_k_len] => [bs, head_num, max_q_len, head_size]
    launchLinearStridedBatchGemm(
        qkT,
        v_cache,
        padded_qkTv,
        cublas_wrapper,
        false,
        false
    );
    DeviceSyncAndCheckCudaError();
    // printf("qkTv pass!\n");

    #ifdef SAVE_DATA
        saveTensor(padded_qkTv, "qk_v_buf_after_bmm.bin", layer_id->as<int>());
    #else
    #endif

    // 8. Transpose + reshape
    // Shape: [bs, head_num, max_q_len, head_size] => 
    //        [bs, max_q_len, head_num, head_size] => 
    //        [numtokens, hiddenunits]
    launchFusedTransposeAndRemovePadding(
        padded_qkTv,
        padding_offset->wrap<int>(),
        transposed_unpadded_qkv
    );
    DeviceSyncAndCheckCudaError();
    // printf("transpose reshape pass!\n");

    // 9. Output linear
    // Shape: [numtokens, hiddenunits] => [numtokens, hiddenunits]
    Tensor *attention_output = outputs->at("attention_output");
    launchLinearGemm(
        transposed_unpadded_qkv,
        &weights->output,
        attention_output->wrap<T>(),
        cublas_wrapper,
        false,
        weights->output.is_transposed
    );

    #ifdef SAVE_DATA
        saveTensor(attention_output->as<T>(), "out_linear_output.bin", layer_id->as<int>());
    #else
    #endif

    DeviceSyncAndCheckCudaError();
    // printf("output linear pass!\n");
    this->freeBuf();
}

template class LlamaContextAttentionLayer<float>;
template class LlamaContextAttentionLayer<half>;