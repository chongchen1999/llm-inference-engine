#include <math.h>
// #include "src/utils/debug_utils.h"
#include "src/layers/includes/masked_self_attention.h"

// Note: In the layers folder, many operations have `DeviceSyncAndCheckCudaError();` added afterwards. 
// You can manually remove these or add conditional compilation code as shown in lesson30.

template<typename T>
LlamaSelfAttentionLayer<T>::LlamaSelfAttentionLayer(
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
void LlamaSelfAttentionLayer<T>::allocateMemoryForForward(LlamaAttentionDynamicParams *params) {
    const int batch_size = params->batch_size;
    const int num_tokens = params->num_tokens;
    const int max_q_len = params->max_q_len;
    const int max_k_len = params->max_k_len;
    DataType type = getTensorType<T>();
    const int qkv_head_num = head_num + 2 * kv_head_num;

    // () Note: Current step's q, k, v shapes have step or seqlen as 1. 
    // Previous step's kv is directly used from kv cache during gemv.
    qkv_buf = new TensorWrapper<T>(Device::GPU, type, { batch_size, qkv_head_num, head_size });
    mha_output = new TensorWrapper<T>(Device::GPU, type, { batch_size, hidden_units });

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

// () Note: Params order of the launcher function in LaMAContextAttentionLayer<T>::forward: 
// (input[Tensor], input[Tensor], ..., weight[Weight], output[*])
template<typename T>
void LlamaSelfAttentionLayer<T>::forward(
    TensorMap *inputs,
    TensorMap *outputs,
    LlamaAttentionWeights<T> *weights,
    LlamaAttentionDynamicParams *params
) {   
    // () Note: Allocate intermediate buffer for layer forward
    allocateMemoryForForward(&params);

    // 1. qkv linear
    // Shape: [bs, 1, q_hidden_units] * [q_hidden_units, hidden_units] = [bs, 1, hidden_units]
    Tensor *attention_input = (*inputs)["attention_input"];
    launchLinearGemm(
        attention_input->warp<T>(),
        weights->qkv,
        qkv_buf,
        cublas_wrapper,
        false,
        true
    );
    DeviceSyncAndCheckCudaError();

    // 2. biasRope
    Tensor *attention_output = (*outputs)["attention_output"];
    // kv cache shape = [bs, kv_head_num, max_seq_len_head_size]
    Tensor *key_cache = (*outputs)["all_k_cache"];
    Tensor *value_cache = (*outputs)["all_v_cache"];
    Tensor *finished = (*inputs)["finished"];
    Tensor *step = (*inputs)["step"]; // [1] on CPU
    Tensor *layer_id = (*inputs)["layer_id"]; // [1] on CPU

    launchRope(
        qkv_buf,
        step->wrap<int>(),
        &attn_static_params
    );
    DeviceSyncAndCheckCudaError();

    // 3. fused masked mha
    launchDecoderMaskedMultiHeadAttention<T>(
        qkv_buf,
        weights->qkv,
        layer_id->wrap<int>(),
        key_cache->wrap<T>(),
        value_cache->wrap<T>(),
        finished->wrap<bool>(),
        step->wrap<int>(),
        mha_output,
        &attn_static_params
    );
    DeviceSyncAndCheckCudaError();

#ifdef SAVE_DATA
    save_tensor(
        mha_output,
        "self_decoder_qk_v_after_bmm.bin",
        layer_id->as<int>()
    );
#endif

    // 4. attention output linear
    launchLinearGemm(
        mha_output,
        weights->output,
        attention_output->wrap<T>(),
        cublas_wrapper,
        false,
        true
    );
    DeviceSyncAndCheckCudaError();

#ifdef SAVE_DATA
    save_tensor(
        mha_output,
        "self_decoder_outlinear_out.bin",
        layer_id->as<int>()
    );
#endif

    this->freeBuf();
}

template class LlamaSelfAttentionLayer<float>;
template class LlamaSelfAttentionLayer<half>;
