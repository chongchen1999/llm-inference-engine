#include "includes/rope.cuh"
#include "includes/rope_utils.cuh"

template<typename T>
__global__ void applyRope(
    T *q,
    T *k,
    const int batch_size,
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int step,
    const int rotary_embedding_dim,
    const float rotary_embedding_base
) {
    const int tid = threadIdx.x;
    const int q_head_id = blockIdx.x;
    const int q_batch_id = blockIdx.y;

    const int kv_head_id = q_head_id / (head_num / kv_head_num);
    const int kv_batch_id = q_batch_id;

    const int batch_stride = head_num * head_size;
    const int kv_batch_stride = kv_head_num * head_size;
    const int head_stride = head_size;
    const int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    const int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;

    if (tid >= rotary_embedding_dim / 2) {
        return;
    }

    // RoPE
    const int half_head_size = head_size >> 1;
    float2 cis = getRopeFreqCis(tid * 2, rotary_embedding_dim, rotary_embedding_base, step - 1);
    float2 q_rotate = getRopeRes(q[q_offset], q[q_offset + half_head_size], cis.x, cis.y);
    float2 k_rotate = getRopeRes(k[k_offset], k[k_offset + half_head_size], cis.x, cis.y);

    q[q_offset] = q_rotate.x;
    q[q_offset + half_head_size] = q_rotate.y;
    k[k_offset] = k_rotate.x;
    k[k_offset + half_head_size] = k_rotate.y;
}

// TODO: fp16 self decoder rope has not been implemented yet
template<>
__global__ void applyRope(
    half *q,
    half *k,
    const int batch_size,
    const int head_num,
    const int kv_head_num,
    const int head_size,
    const int step,
    const int rotary_embedding_dim,
    const float rotary_embedding_base
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
    const int head_num = static_params->head_num;
    const int kv_head_num = static_params->kv_head_num;
    const int head_size = qkv_buf->shape[2];

    const int cur_step = step->getVal();
    T *qkv_data = qkv_buf->data;
    T *q = qkv_data;
    T *k = qkv_data + head_num * head_size;

    const int rotary_embedding_dim = static_params->rotary_embedding_dim;
    const float rotary_embedding_base = static_params->rotary_embedding_base;

    dim3 grid(head_num, batch_size);
    dim3 block(head_size);

    applyRope<T><<<grid, block>>>(
        q,
        k,
        batch_size,
        head_num,
        kv_head_num,
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