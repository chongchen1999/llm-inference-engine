#include "src/kernels/includes/concat_past_kv.h"
// #include "src/utils/cuda_debug_utils.cuh"
#include <iostream>

template <typename T>
__global__ void appendKeyValueCache(
    T *kv_dst,                              // [num layers, bs, kv head num, max_seq_len, head_size]
    const size_t layer_offset,
    const T *kv_src,                        // [bs, kv_head num, max_q_len, head_size]
    const int kv_head_num,
    const int head_size,
    const int *cur_query_length,
    const int *history_length,
    const int max_q_len,
    const int max_seq_len
) {
    const int tid = threadIdx.x;
    const int token_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;

    T *kv_cache_dst = kv_dst + layer_offset;

    const int cur_seq_len = cur_query_length[batch_id];
    const int cumsum_seq_len = history_length[batch_id];

    if (token_id < cur_seq_len) {
        int src_offset = batch_id * kv_head_num * max_q_len * head_size +
                         head_id * max_q_len * head_size +
                         token_id * head_size + tid;
        int dst_offset = batch_id * kv_head_num * max_seq_len * head_size +
                         head_id * max_seq_len * head_size +
                         (cumsum_seq_len + token_id) * head_size + tid;

        kv_cache_dst[dst_offset] = kv_src[src_offset];
    }
}

template <typename T>
void launchConcatKVCache(
    TensorWrapper<T> *k_src,                // from qkv bias and rope [batch_size, kv_head_num, max_q_len, head_size]
    TensorWrapper<T> *v_src,
    TensorWrapper<int> *layer_id,          // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
    TensorWrapper<int> *cur_query_length, // current epoch or local input length, [batch_size]
    TensorWrapper<int> *history_length,
    TensorWrapper<T> *k_dst,               // [num_layers, batch_size, kv_head_num, max_seq_len, head_size]
    TensorWrapper<T> *v_dst
) {
    const int batch_size = k_src->shape[0];
    const int kv_head_num = k_src->shape[1];
    const int max_q_len = k_src->shape[2];
    const int head_size = k_src->shape[3];

    const int max_seq_len = k_dst->shape[3];
    const int layer = layer_id->getVal();
    size_t layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size;

    dim3 grid(max_q_len, batch_size, kv_head_num); // last dimension for key/value
    dim3 block(head_size);

    appendKeyValueCache<T><<<grid, block>>>(
        k_dst->data,
        layer_offset,
        k_src->data,
        kv_head_num,
        head_size,
        cur_query_length->data,
        history_length->data,
        max_q_len,
        max_seq_len
    );

    appendKeyValueCache<T><<<grid, block>>>(
        v_dst->data,
        layer_offset,
        v_src->data,
        kv_head_num,
        head_size,
        cur_query_length->data,
        history_length->data,
        max_q_len,
        max_seq_len
    );
}

// Explicit template instantiation
template void launchConcatKVCache(
    TensorWrapper<float> *k_src,
    TensorWrapper<float> *v_src,
    TensorWrapper<int> *layer_id,
    TensorWrapper<int> *cur_query_length,
    TensorWrapper<int> *history_length,
    TensorWrapper<float> *k_dst,
    TensorWrapper<float> *v_dst
);

template void launchConcatKVCache(
    TensorWrapper<half> *k_src,
    TensorWrapper<half> *v_src,
    TensorWrapper<int> *layer_id,
    TensorWrapper<int> *cur_query_length,
    TensorWrapper<int> *history_length,
    TensorWrapper<half> *k_dst,
    TensorWrapper<half> *v_dst
);
