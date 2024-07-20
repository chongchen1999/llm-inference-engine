// launchAddFusedQKVBiasTransposeAndRoPE kernel can be used in prompt phase and launchRoPE kernel is used in token generation phase

// 1.add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
// QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].

// 2.For q and k, apply RoPE, then send to attention.

// 3.rebuild padding to do mha

// input: qkv_buf : qkv continouns buf when no padding
// shape = [num_tokens, qkv_head_num, head_size], 因为各句子长度不一，所以不用bs * seqlen表示
// output: q shape = [bs, head num, seqlen, head size], if k v is this shape, maybe need tranpose in successor steps, ep in cublas
// kv shape = [bs, kv head num, seqlen, head size]
// ps: seqlen = max_q_len here
#include <math.h>
#include <stdio.h>
// #include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/includes/qkv_bias_and_rope.h"

__device__ __forceinline__ float2 getRopeFreqCosisin(int zid, int rot_embed_dim, float base, float t_step) {
    const float inv_freq = t_step / powf(base, (float)zid / rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

__device__ __forceinline__ float2 getRopeRes(float x0, float x1, const float2 cos_isin) {
    return {x0 * cos_isin.x - x1 * cos_isin.y, x1 * cos_isin.x + x0 * cos_isin.y} ;
}

template <typename T>
__global__ void addFusedQKVBiasTransposeKernel(T *q_buf,
                                               T *k_buf,
                                               T *v_buf,
                                               T *QKV,
                                               const T *qkv_bias, /*optional*/
                                               const int *padding_offset, // created before qkv linear
                                               const int *history_length,
                                               const int *input_length, // actual length of each seq
                                               const int batch_size,
                                               const int seq_len, // max_seq_len to pad to
                                               const int token_num,
                                               const int head_num,
                                               const int kv_head_num,
                                               const int head_size,
                                               const int rotary_embedding_dim,
                                               float rotary_embedding_base, // default 10000 in llama
                                               int max_position_embeddings, /*default 2048 in llama*/
                                               bool use_dynamic_ntk) { /*placeholder for ntk RoPE*/
    int token_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;
    int token_padding_offset = padding_offset[token_id];

    // 1. prapare rebuilding , do rebuild padding and transpose when store
    int dst_token_id = token_id + token_padding_offset; // token id after rebuild padding

    int batch_id = dst_token_id / seq_len;       // seqlen is max_seq_len for padding used to unify all seq's length
    int local_token_id = dst_token_id % seq_len; // 每个seq中的局部token id

    // 2. bias add
    int qkv_head_num = head_num + 2 * kv_head_num;
    int base_offset = token_id * qkv_head_num * head_size + head_id * head_size + tid;
    int q_id = base_offset;
    int k_id = base_offset + head_num * head_size;
    int v_id = base_offset + (head_num + kv_head_num) * head_size;

    float v = QKV[v_id];

    // transpose
    int dst_q_id = batch_id * (seq_len * head_num * head_size) +
                   head_id * (seq_len * head_size) +
                   local_token_id * head_size + tid;
    int dst_kv_id = batch_id * (seq_len * kv_head_num * head_size) +
                    head_id * (seq_len * head_size) +
                    local_token_id * head_size + tid;

    if (head_id < kv_head_num) { // note: for MQA and GQA
        v_buf[dst_kv_id] = v;
    }

    // 3. RoPE
    const int cur_seq_history_len = history_length[batch_id];

    // ()note: 多轮对话下要结合history length求得全局的cos和sin
    const int context_length = cur_seq_history_len + input_length[batch_id];

    // ()note: timestep为cos(m*theta)中的m
    const int timestep = cur_seq_history_len + local_token_id; 
    if (tid >= rotary_embedding_dim / 2) {
        return;
    } // tid = [0,1,2,...,63]

    float2 cos_isin = getRopeFreqCosisin(tid * 2, rotary_embedding_dim, rotary_embedding_base, timestep);

    float2 q_rotate = getRopeRes(QKV[q_id], QKV[q_id + head_size / 2], cos_isin);
    float2 k_rotate = getRopeRes(QKV[k_id], QKV[k_id + head_size / 2], cos_isin);

    // ()note: write result back into q k v
    q_buf[dst_q_id] = q_rotate.x;
    q_buf[dst_q_id + head_size / 2] = q_rotate.y;
    if (head_id < kv_head_num) { // for MQA and GQA
        k_buf[dst_kv_id] = k_rotate.x;
        k_buf[dst_kv_id + head_size / 2] = k_rotate.y;
    }
}

// input: qkv_buf : qkv continouns buf when no padding
// shape = [num_tokens, qkv_head_num, head_size]
// output: q shape = [bs, head num, seqlen, head size], if k v is this shape, maybe need tranpose in successor steps, ep in cublas
//         k/v shape = [bs, kv head num, seqlen, head size]
template <typename T>
void launchAddFusedQKVBiasTransposeAndRope(TensorWrapper<T> *q_buf,
                                           TensorWrapper<T> *k_buf,
                                           TensorWrapper<T> *v_buf,
                                           TensorWrapper<T> *QKV,
                                           BaseWeight<T> &qkv, // output
                                           // Tensor* qkv_bias,
                                           TensorWrapper<int> *padding_offset,
                                           TensorWrapper<int> *history_length,
                                           TensorWrapper<int> *input_length,
                                           LlamaAttentionStaticParams &params) {
    int token_num = QKV->shape[0];
    int qkv_head_num = QKV->shape[1];
    int head_size = QKV->shape[2];

    int batch_size = q_buf->shape[0];
    int head_num = q_buf->shape[1];
    int seq_len = q_buf->shape[2];

    LLM_CHECK_WITH_INFO(k_buf->shape[1] == v_buf->shape[1], "k and v should have same head_num");
    LLM_CHECK_WITH_INFO(k_buf->shape[1] == (qkv_head_num - head_num) / 2, "k and v should have same head_num");
    int kv_head_num = k_buf->shape[1];

    dim3 grid(token_num, head_num);
    dim3 block(head_size);
    addFusedQKVBiasTransposeKernel<T><<<grid, block>>>(q_buf->data,
                                                       k_buf->data,
                                                       v_buf->data,
                                                       QKV->data,
                                                       /*optional*/qkv.bias,
                                                       padding_offset->data,
                                                       history_length->data,
                                                       input_length->data,
                                                       batch_size,
                                                       seq_len,
                                                       token_num,
                                                       head_num,
                                                       kv_head_num,
                                                       head_size,
                                                       params.rotary_embedding_dim,
                                                       params.rotary_embedding_base,
                                                       params.max_position_embeddings,
                                                       params.use_dynamic_ntk);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(q_buf->data);
#else
#endif
}

template void launchAddFusedQKVBiasTransposeAndRope(TensorWrapper<float> *q_buf,
                                                    TensorWrapper<float> *k_buf,
                                                    TensorWrapper<float> *v_buf,
                                                    TensorWrapper<float> *QKV,
                                                    BaseWeight<float> &qkv,
                                                    TensorWrapper<int> *padding_offset,
                                                    TensorWrapper<int> *history_length,
                                                    TensorWrapper<int> *input_length,
                                                    LlamaAttentionStaticParams &params);

template void launchAddFusedQKVBiasTransposeAndRope(TensorWrapper<half> *q_buf,
                                                    TensorWrapper<half> *k_buf,
                                                    TensorWrapper<half> *v_buf,
                                                    TensorWrapper<half> *QKV,
                                                    BaseWeight<half> &qkv,
                                                    TensorWrapper<int> *padding_offset,
                                                    TensorWrapper<int> *history_length,
                                                    TensorWrapper<int> *input_length,
                                                    LlamaAttentionStaticParams &params);

// note: this kernel is called in self decoder, not context decoder
template<typename T>
__global__ void rope_kernel_for_self_decoder(T* q,
                                             T* k,
                                             const int batch_size,
                                             const int head_num,
                                             const int kv_head_num,
                                             const int head_size,
                                             const int step,
                                             int rotary_embedding_dim,
                                             float rotary_embedding_base) {
    int tid = threadIdx.x;
    int q_head_id = blockIdx.x;
    int q_batch_id = blockIdx.y;
    // ()note: !!should add () to head_num / kv_head_num, or res is wrong
    int kv_head_id = q_head_id / (head_num / kv_head_num);
    int kv_batch_id = q_batch_id;

    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
    if (tid >= rotary_embedding_dim / 2) {
        return;
    }
    // RoPE
    float k_reg = k[k_offset];
    float k_rotate_reg = k[k_offset + head_size / 2];
    float2 cos_sin = getRopeFreqCosisin(tid * 2, rotary_embedding_dim, rotary_embedding_base, step - 1);
    float2 q_rotate = getRopeRes(q[q_offset], q[q_offset + head_size / 2], cos_sin);
    float2 k_rotate = make_float2(0,0);
    k_rotate.x = cos_sin.x * k_reg - cos_sin.y * k_rotate_reg;
    k_rotate.y = cos_sin.x * k_rotate_reg + cos_sin.y * k_reg;

    q[q_offset] = q_rotate.x;
    q[q_offset + head_size / 2] = q_rotate.y;
    k[k_offset] = k_rotate.x;
    k[k_offset + head_size / 2] = k_rotate.y;
}
// TODO: fp16 self decoder rope has not implemented yet
template<>
__global__ void rope_kernel_for_self_decoder(half* q,
                    half* k,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base) {}

// note: all TensorWrapper's shape cant see here, we can see it in context_decoder.cpp or self_decoder.cpp
template<typename T>
void launchRope(TensorWrapper<T>* qkv_buf,
                TensorWrapper<int>* step,
                LlamaAttentionStaticParams& static_params) {
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    int head_num = 32; // only for llama
    const int head_size = qkv_buf->shape[2];
    LLM_CHECK(batch_size == 1);
    LLM_CHECK(qkv_head_num == 96);
    LLM_CHECK(head_size == 128);
    const int cur_step = step->getVal();
    T* qkv_data = qkv_buf->data;
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size;

    int   rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    int   max_position_embeddings = static_params.max_position_embeddings;
    dim3 grid(head_num, batch_size);
    dim3 block(head_size); 
    rope_kernel_for_self_decoder<T><<<grid, block>>>(q,
                                                    k,
                                                    batch_size,
                                                    head_num,
                                                    head_num, // only for llama, kv head = head
                                                    head_size,
                                                    cur_step,
                                                    rotary_embedding_dim,
                                                    rotary_embedding_base);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(q);
#else
#endif
}

template void launchRope(TensorWrapper<float> *qkv_buf,
                         TensorWrapper<int> *step,
                         LlamaAttentionStaticParams &static_params);

template void launchRope(TensorWrapper<half> *qkv_buf,
                         TensorWrapper<int> *step,
                         LlamaAttentionStaticParams &static_params);