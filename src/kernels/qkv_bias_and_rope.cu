#include "includes/qkv_bias_and_rope.cuh"
#include "includes/rope_utils.cuh"

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
    const int max_position_embeddings, // Default 2048 in LLaMA
    const bool use_dynamic_ntk // Placeholder for ntk RoPE
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

    const int dst_q_id = 
        batch_id * (seq_len * head_num * head_size) +
        head_id * (seq_len * head_size) + local_token_id * head_size + tid;
    const int dst_kv_id = 
        batch_id * (seq_len * kv_head_num * head_size) +
        head_id * (seq_len * head_size) + local_token_id * head_size + tid;

    // For MQA and GQA
    if (head_id < kv_head_num) {
        v_buf[dst_kv_id] = QKV[v_id];
    }

    // Apply RoPE
    const int cur_seq_history_len = history_length[batch_id];
    // const int context_length = cur_seq_history_len + input_length[batch_id];
    const int time_step = cur_seq_history_len + local_token_id;

    if (tid >= rotary_embedding_dim / 2) {
        return;
    }

    float2 cis = getRopeFreqCis(tid * 2, rotary_embedding_dim, rotary_embedding_base, time_step);
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
    LlamaAttentionStaticParams *static_params
) {
    const int token_num = QKV->shape[0];
    const int qkv_head_num = QKV->shape[1];
    const int head_size = QKV->shape[2];

    const int batch_size = q_buf->shape[0];
    const int head_num = q_buf->shape[1];
    const int seq_len = q_buf->shape[2];

    LLM_CHECK_WITH_INFO(k_buf->shape[1] == v_buf->shape[1], "k and v should have same head_num");
    LLM_CHECK_WITH_INFO(k_buf->shape[1] == (qkv_head_num - head_num) / 2, "k and v should have same head_num");
    LLM_CHECK_WITH_INFO(q_buf->shape[3] == head_size, "head_size does not match!");
    const int kv_head_num = k_buf->shape[1];

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
        static_params->rotary_embedding_dim,
        static_params->rotary_embedding_base,
        static_params->max_position_embeddings,
        static_params->use_dynamic_ntk
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
    LlamaAttentionStaticParams *static_params
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
    LlamaAttentionStaticParams *static_params
);