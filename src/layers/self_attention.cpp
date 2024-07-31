#include <math.h>
#include "../utils/debug_utils.h"
#include "includes/self_attention.h"
#include "../utils/output_utils.h"

template<typename T>
LlamaSelfAttentionLayer<T>::LlamaSelfAttentionLayer(
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
    // TODO: Check if kv_head_num is divisible by head_num
    repeats_per_kv(head_num / kv_head_num),
    scale(1.0f / sqrt(static_cast<float>(head_size))) {}

template<typename T>
void LlamaSelfAttentionLayer<T>::allocateMemory(LlamaAttentionDynamicParams *params) {
    const int batch_size = params->batch_size;
    const int num_tokens = params->num_tokens;
    const int max_q_len = params->max_q_len;
    const int max_k_len = params->max_k_len;
    DataType type = getTensorType<T>();
    const int qkv_head_num = head_num + 2 * kv_head_num;

    // () Note: Current step's q, k, v shapes have step or seqlen as 1. 
    // Previous step's kv is directly used from kv cache during gemv.
    qkv_buf = std::unique_ptr<TensorWrapper<T>>(new TensorWrapper<T>(Device::GPU, type, {batch_size, qkv_head_num, head_size}));
    mha_output = std::unique_ptr<TensorWrapper<T>>(new TensorWrapper<T>(Device::GPU, type, {batch_size, hidden_units}));

    allocator->malloc(
        &qkv_buf->data,
        sizeof(T) * batch_size * qkv_head_num * head_size,
        false
    );
    
    allocator->malloc(
        &mha_output->data,
        sizeof(T) * batch_size * hidden_units,
        false
    );
}

template<typename T>
void LlamaSelfAttentionLayer<T>::freeBuf() {
    allocator->free(qkv_buf->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(mha_output->data);
    DeviceSyncAndCheckCudaError();
}

// Params order of the launcher function in LaMAContextAttentionLayer<T>::forward: 
// (input[Tensor], input[Tensor], ..., weight[Weight], output[*])
template<typename T>
void LlamaSelfAttentionLayer<T>::forward(
    TensorMap *inputs,
    TensorMap *outputs,
    LlamaAttentionWeights<T> *weights,
    LlamaAttentionDynamicParams *params
) {   
    // Note: Allocate intermediate buffer for layer forward
    allocateMemory(params);
    printf("allocated!\n");

    // 1. qkv linear
    // Shape: [bs, 1, q_hidden_units] * [q_hidden_units, qkv_hidden_units] = [bs, 1, qkv_hidden_units]
    Tensor *attention_input = inputs->at("attention_input"); // [bs, 1, q_hidden_units]
    printf("attention_input shape:\n");
    print_tensor(attention_input);
    launchLinearGemm(
        attention_input->wrap<T>(),
        &weights->qkv,
        qkv_buf.get(),
        cublas_wrapper,
        false,
        weights->qkv.is_transposed
    );
    DeviceSyncAndCheckCudaError();
    printf("qkv linear!\n");

    // 2. biasRope
    Tensor *attention_output = outputs->at("attention_output");

    // kv cache shape = [bs, kv_head_num, max_seq_len_head_size]
    Tensor *key_cache = outputs->at("all_k_cache");
    Tensor *value_cache = outputs->at("all_v_cache");
    Tensor *finished = inputs->at("finished");
    Tensor *step = inputs->at("step"); // [1] on CPU
    Tensor *layer_id = inputs->at("layer_id"); // [1] on CPU

    launchRope(
        qkv_buf.get(),
        step->wrap<int>(),
        attention_static_params
    );
    DeviceSyncAndCheckCudaError();
    printf("rope!\n");

    printf("qkv_buf shape:\n");
    print_tensor(qkv_buf.get());
    printf("qkv_bias shape:\n");
    print_weight(&weights->qkv);
    printf("layer_id shape:\n");
    print_tensor(layer_id);
    printf("key_cache shape:\n");
    print_tensor(key_cache);
    printf("value_cache shape:\n");
    print_tensor(value_cache);
    printf("finished shape:\n");
    print_tensor(finished);
    printf("step shape:\n");
    print_tensor(step);
    printf("mha_output shape:\n");
    print_tensor(mha_output.get());

    // 3. fused masked mha
    launchDecoderMaskedMultiHeadAttention<T>(
        qkv_buf.get(),
        &weights->qkv,
        layer_id->wrap<int>(),
        key_cache->wrap<T>(),
        value_cache->wrap<T>(),
        finished->wrap<bool>(),
        step->wrap<int>(),
        mha_output.get(),
        attention_static_params
    );
    DeviceSyncAndCheckCudaError();
    printf("mha!\n");

#ifdef SAVE_DATA
    saveTensor(
        mha_output,
        "self_decoder_qk_v_after_bmm.bin",
        layer_id->as<int>()
    );
#endif

    // 4. attention output linear
    launchLinearGemm(
        mha_output.get(),
        &weights->output,
        attention_output->wrap<T>(),
        cublas_wrapper,
        false,
        weights->output.is_transposed
    );
    DeviceSyncAndCheckCudaError();
    printf("output linear!\n");

#ifdef SAVE_DATA
    saveTensor(
        mha_output,
        "self_decoder_outlinear_out.bin",
        layer_id->as<int>()
    );
#endif

    this->freeBuf();
}

template class LlamaSelfAttentionLayer<float>;
template class LlamaSelfAttentionLayer<half>;
