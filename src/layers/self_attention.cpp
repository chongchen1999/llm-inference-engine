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
    repeats_per_kv(head_num / kv_head_num),
    scale(1.0f / sqrt(static_cast<float>(head_size))) {
        LLM_CHECK_WITH_INFO(head_num % kv_head_num == 0, "kv_head_num must be a factor of head_num");
    }

template<typename T>
void LlamaSelfAttentionLayer<T>::allocateMemory(LlamaAttentionDynamicParams *dynamic_params) {
    const int batch_size = dynamic_params->batch_size;
    const int num_tokens = dynamic_params->num_tokens;
    const int max_q_len = dynamic_params->max_q_len;
    const int max_k_len = dynamic_params->max_k_len;
    DataType type = getTensorType<T>();
    const int qkv_head_num = head_num + 2 * kv_head_num;

    // () Note: Current step's q, k, v shapes have step or seqlen as 1. 
    // Previous step's kv is directly used from kv cache during gemv.
    qkv_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, qkv_head_num, head_size});
    mha_output = new TensorWrapper<T>(Device::GPU, type, {batch_size, hidden_units});

    allocator->malloc(&qkv_buf->data, sizeof(T) * batch_size * qkv_head_num * head_size, false);
    allocator->malloc(&mha_output->data, sizeof(T) * batch_size * hidden_units, false);
}

template<typename T>
void LlamaSelfAttentionLayer<T>::freeBuf() {
    allocator->free(qkv_buf->data);
    DeviceSyncAndCheckCudaError();

    allocator->free(mha_output->data);
    DeviceSyncAndCheckCudaError();

    delete qkv_buf;
    delete mha_output;
    qkv_buf = nullptr;
    mha_output = nullptr;
}

// Params order of the launcher function in LaMAContextAttentionLayer<T>::forward: 
// (input[Tensor], input[Tensor], ..., weight[Weight], output[*])
template<typename T>
void LlamaSelfAttentionLayer<T>::forward(
    TensorMap *inputs,
    TensorMap *outputs,
    LlamaAttentionWeights<T> *weights,
    LlamaAttentionDynamicParams *dynamic_params
) {   
    // Note: Allocate intermediate buffer for layer forward
    allocateMemory(dynamic_params);
    printf("allocated!\n");

    // 1. qkv linear
    // Shape: [bs, 1, q_hidden_units] * [q_hidden_units, qkv_hidden_units] = [bs, 1, qkv_hidden_units]
    Tensor *attention_input = inputs->at("attention_input"); // [bs, 1, q_hidden_units]
    printf("attention_input shape:\n");
    print_tensor(attention_input);
    launchLinearGemm(
        attention_input->wrap<T>(),
        &weights->qkv,
        qkv_buf,
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
        qkv_buf,
        step->wrap<int>(),
        attention_static_params
    );
    DeviceSyncAndCheckCudaError();

    // 3. fused masked mha
    launchDecoderMaskedMultiHeadAttention<T>(
        qkv_buf,
        &weights->qkv,
        layer_id->wrap<int>(),
        key_cache->wrap<T>(),
        value_cache->wrap<T>(),
        finished->wrap<bool>(),
        step->wrap<int>(),
        mha_output,
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
        mha_output,
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
