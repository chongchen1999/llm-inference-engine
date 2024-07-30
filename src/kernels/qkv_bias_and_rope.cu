#pragma once

#include <math.h>
#include <stdio.h>
#include "includes/qkv_bias_and_rope.h"

// Utility functions for RoPE (Rotary Positional Encoding)
__device__ __forceinline__ float2 getRopeFreqCis(
    const int &zid,
    const int &rot_embed_dim,
    const float &base,
    const float &m
) {
    const float inv_freq = m / powf(base, static_cast<float>(zid) / rot_embed_dim);
    return {cosf(inv_freq), sinf(inv_freq)};
}

__device__ __forceinline__ float2 getRopeRes(
    const float &x0,
    const float &x1,
    const float &cos_,
    const float &sin_
) {
    return {x0 * cos_ - x1 * sin_, x1 * cos_ + x0 * sin_};
}

// Kernel to add bias, transpose, and apply RoPE to QKV tensors
template <typename T>
__global__ void fusedAddQKVBiasAndTransposeAndRope(
    T *q_buf,
    T *k_buf,
    T *v_buf,
    const T *QKV,
    const T *qkv_bias, // Optional
    const int *padding_offset,
    const int *history_length,
    const int *input_length,
    const int batch_size,
    const int seq_len, // Max_seq_len for padding
    const int token_num,
    const int head_num, // aka q_head_num
    const int kv_head_num,
    const int head_size,
    const int rotary_embedding_dim,
    const float rotary_embedding_base, // Default 10000 in LLaMA
    int max_position_embeddings, // Default 2048 in LLaMA
    bool use_dynamic_ntk // Placeholder for ntk RoPE
) {
    // grid_shape: [num_tokens, head_num + 2 * kv_head_num]
    // block_shape: [head_size]
    const int token_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int tid = threadIdx.x;
    const int token_padding_offset = padding_offset[token_id];

    // Prepare rebuilding, do rebuild padding and transpose when storing
    const int dst_token_id = token_id + token_padding_offset; // Token id after rebuild padding
    const int batch_id = dst_token_id / seq_len; // seq_len is max_seq_len for padding
    const int local_token_id = dst_token_id % seq_len; // Token id in the batch

    // Bias addition and transpose
    const int qkv_head_num = head_num + 2 * kv_head_num;
    const int base_offset = token_id * qkv_head_num * head_size;
    const int q_id = base_offset + head_id * head_size + tid;
    const int k_id = base_offset + (head_num + head_id) * head_size + tid;
    const int v_id = base_offset + (head_num + kv_head_num + head_id) * head_size + tid;

    const int dst_q_id = batch_id * (seq_len * head_num * head_size) +
                         head_id * (seq_len * head_size) +
                         local_token_id * head_size + tid;
    const int dst_kv_id = batch_id * (seq_len * kv_head_num * head_size) +
                          head_id * (seq_len * head_size) +
                          local_token_id * head_size + tid;

    // For MQA and GQA
    if (head_id < kv_head_num) {
        v_buf[dst_kv_id] = QKV[v_id];
    }

    // Apply RoPE
    const int cur_seq_history_len = history_length[batch_id];
    const int context_length = cur_seq_history_len + input_length[batch_id];
    const int timestep = cur_seq_history_len + local_token_id;

    if (tid >= rotary_embedding_dim / 2) {
        return;
    }

    float2 cis = getRopeFreqCis(tid * 2, rotary_embedding_dim, rotary_embedding_base, timestep);
    const int half_head_size = head_size >> 1;
    float2 q_rotate = getRopeRes(QKV[q_id], QKV[q_id + half_head_size], cis.x, cis.y);
    float2 k_rotate = getRopeRes(QKV[k_id], QKV[k_id + half_head_size], cis.x, cis.y);

    // Write result back into q, k, v
    q_buf[dst_q_id] = q_rotate.x;
    q_buf[dst_q_id + half_head_size] = q_rotate.y;

    if (head_id < kv_head_num) { // For MQA and GQA
        k_buf[dst_kv_id] = k_rotate.x;
        k_buf[dst_kv_id + half_head_size] = k_rotate.y;
    }
}

/* Input: QKV: QKV continuous buffer when no padding
   Shape = [num_tokens, qkv_head_num, head_size]
   Output: q shape = [bs, head_num, seqlen, head_size]
           k, v shape = [bs, kv_head_num, seqlen, head_size]
*/
template <typename T>
void launchFusedQKVAddBiasAndTransposeAndRope(
    TensorWrapper<T> *q_buf,
    TensorWrapper<T> *k_buf,
    TensorWrapper<T> *v_buf,
    TensorWrapper<T> *QKV,
    BaseWeight<T> *qkv, // Bias
    TensorWrapper<int> *padding_offset,
    TensorWrapper<int> *history_length,
    TensorWrapper<int> *input_length,
    LlamaAttentionStaticParams *params
) {
    int token_num = QKV->shape[0];
    int qkv_head_num = QKV->shape[1];
    int head_size = QKV->shape[2];

    int batch_size = q_buf->shape[0];
    int head_num = q_buf->shape[1];
    int seq_len = q_buf->shape[2];

    LLM_CHECK_WITH_INFO(k_buf->shape[1] == v_buf->shape[1], "k and v should have same head_num");
    LLM_CHECK_WITH_INFO(k_buf->shape[1] == (qkv_head_num - head_num) / 2, "k and v should have same head_num");
    LLM_CHECK_WITH_INFO(q_buf->shape[3] == head_size, "head_size does not match!");
    int kv_head_num = k_buf->shape[1];

    dim3 grid(token_num, head_num);
    dim3 block(head_size);

    fusedAddQKVBiasAndTransposeAndRope<T><<<grid, block>>>(
        q_buf->data,
        k_buf->data,
        v_buf->data,
        QKV->data,
        qkv->bias,
        padding_offset->data,
        history_length->data,
        input_length->data,
        batch_size,
        seq_len,
        token_num,
        head_num,
        kv_head_num,
        head_size,
        params->rotary_embedding_dim,
        params->rotary_embedding_base,
        params->max_position_embeddings,
        params->use_dynamic_ntk
    );

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(q_buf->data);
#endif
}

template void launchFusedQKVAddBiasAndTransposeAndRope(
    TensorWrapper<float> *q_buf,
    TensorWrapper<float> *k_buf,
    TensorWrapper<float> *v_buf,
    TensorWrapper<float> *QKV,
    BaseWeight<float> *qkv,
    TensorWrapper<int> *padding_offset,
    TensorWrapper<int> *history_length,
    TensorWrapper<int> *input_length,
    LlamaAttentionStaticParams *params
);

template void launchFusedQKVAddBiasAndTransposeAndRope(
    TensorWrapper<half> *q_buf,
    TensorWrapper<half> *k_buf,
    TensorWrapper<half> *v_buf,
    TensorWrapper<half> *QKV,
    BaseWeight<half> *qkv,
    TensorWrapper<int> *padding_offset,
    TensorWrapper<int> *history_length,
    TensorWrapper<int> *input_length,
    LlamaAttentionStaticParams *params
);

// Kernel to apply RoPE to self-decoder
template<typename T>
__global__ void selfDecoderRope(
    T *q,
    T *k,
    const int batch_size,
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int step,
    int rotary_embedding_dim,
    float rotary_embedding_base
) {
    int tid = threadIdx.x;
    int q_head_id = blockIdx.x;
    int q_batch_id = blockIdx.y;

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
    float2 cos_sin = getRopeFreqCis(tid * 2, rotary_embedding_dim, rotary_embedding_base, step - 1);
    float2 q_rotate = getRopeRes(q[q_offset], q[q_offset + head_size / 2], cos_sin.x, cos_sin.y);
    float2 k_rotate = {0, 0};
    k_rotate.x = cos_sin.x * k_reg - cos_sin.y * k_rotate_reg;
    k_rotate.y = cos_sin.x * k_rotate_reg + cos_sin.y * k_reg;

    q[q_offset] = q_rotate.x;
    q[q_offset + head_size / 2] = q_rotate.y;
    k[k_offset] = k_rotate.x;
    k[k_offset + head_size / 2] = k_rotate.y;
}

// TODO: fp16 self decoder rope has not been implemented yet
template<>
__global__ void selfDecoderRope(
    half *q,
    half *k,
    const int batch_size,
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int step,
    int rotary_embedding_dim,
    float rotary_embedding_base
) {}

// Launch function for self-decoder RoPE
template<typename T>
void launchRope(
    TensorWrapper<T> *qkv_buf,
    TensorWrapper<int> *step,
    LlamaAttentionStaticParams *static_params
) {
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int head_num = 32; // Only for LLaMA
    const int head_size = qkv_buf->shape[2];

    LLM_CHECK(batch_size == 1);
    LLM_CHECK(qkv_head_num == 96);
    LLM_CHECK(head_size == 128);

    const int cur_step = step->getVal();
    T *qkv_data = qkv_buf->data;
    T *q = qkv_data;
    T *k = qkv_data + head_num * head_size;

    int rotary_embedding_dim = static_params->rotary_embedding_dim;
    float rotary_embedding_base = static_params->rotary_embedding_base;

    dim3 grid(head_num, batch_size);
    dim3 block(head_size);

    selfDecoderRope<T><<<grid, block>>>(
        q,
        k,
        batch_size,
        head_num,
        head_num, // Only for LLaMA, kv_head = head
        head_size,
        cur_step,
        rotary_embedding_dim,
        rotary_embedding_base
    );

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(q);
#endif
}

template void launchRope(
    TensorWrapper<float> *qkv_buf,
    TensorWrapper<int> *step,
    LlamaAttentionStaticParams *static_params
);

template void launchRope(
    TensorWrapper<half> *qkv_buf,
    TensorWrapper<int> *step,
    LlamaAttentionStaticParams *static_params
);