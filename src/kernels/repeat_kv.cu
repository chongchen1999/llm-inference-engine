#include "src/kernels/includes/repeat_kv.h"
// #include "src/utils/cuda_debug_utils.cuh"
#include <iostream>

/*
    dim3 block(block_size = 256);
    dim3 grid(
        (max_k_len * head_size + block_size - 1) / block_size,
        batch_size,
        head_num
    );
*/

template <typename T>
__global__ void repeatKVCache(
    T *kv_dst, // [batch_size, head_num, max_k_len, head_size]
    const T *kv_src, // [num_layers, batch_size, kv_head_num, max_seq_len, head_size]
    const int layer_offset,
    const int head_num,
    const int repeated_heads_per_kv,
    const int head_size,
    const int *context_length,
    const int max_k_len,
    const int max_seq_len
) {
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x; // process [.., .., max_seq_len, head_size]

    const T *val_src = kv_src + layer_offset;
    T *val_dst = kv_dst;

    const int seq_len = context_length[batch_id];
    const int kv_head_id = gid % head_size;
    const int kv_seq_id = gid / head_size;
    const int kv_head_num = head_num / repeated_heads_per_kv;
    const int kv_head_id = head_id / repeated_heads_per_kv;

    // Only fetch context_length (< max_seq_len) KV data from all KV cache of the current sequence
    if (kv_seq_id < seq_len) {
        const int src_idx = batch_id * kv_head_num * max_seq_len * head_size +
                            kv_head_id * max_seq_len * head_size +
                            kv_seq_id * head_size +
                            kv_head_id;

        const int dst_idx = batch_id * head_num * head_size * max_k_len +
                            head_id * head_size * max_k_len +
                            kv_seq_id * head_size +
                            kv_head_id;

        val_dst[dst_idx] = val_src[src_idx];
    }
}

template <typename T>
void launchRepeatKVCache(
    TensorWrapper<T> *k_cache_src, // [num_layers, batch_size, kv_head_num, max_seq_len, head_size]
    TensorWrapper<T> *v_cache_src, // [num_layers, batch_size, kv_head_num, max_seq_len, head_size]
    TensorWrapper<int> *context_length,
    TensorWrapper<int> *layer_id,
    TensorWrapper<T> *k_cache_dst, // [batch_size, head_num, max_k_len, head_size]
    TensorWrapper<T> *v_cache_dst // [batch_size, head_num, max_k_len, head_size]
) {
    const int batch_size = context_length->shape[0];
    const int kv_head_num = k_cache_src->shape[2];
    const int max_seq_len = k_cache_src->shape[3];
    const int head_num = k_cache_dst->shape[1];
    const int max_k_len = k_cache_dst->shape[2];
    const int head_size = k_cache_dst->shape[3];
    const int layer = layer_id->getVal(); // Ensure layer_id is on CPU if using getVal<int>()

    const int layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size;
    const int repeated_heads_per_kv = head_num / kv_head_num;
    const int block_size = 256;

    dim3 block(block_size);
    dim3 grid(
        (max_k_len * head_size + block_size - 1) / block_size,
        batch_size,
        head_num
    );

    repeatKVCache<T><<<grid, block>>>(
        v_cache_dst->data,
        v_cache_src->data,
        layer_offset,
        head_num,
        repeated_heads_per_kv,
        head_size,
        context_length->data,
        max_k_len,
        max_seq_len
    );

    repeatKVCache<T><<<grid, block>>>(
        k_cache_dst->data,
        k_cache_src->data,
        layer_offset,
        head_num,
        repeated_heads_per_kv,
        head_size,
        context_length->data,
        max_k_len,
        max_seq_len
    );

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(k_cache_dst->data);
#endif
}

// Explicit template instantiations
template void launchRepeatKVCache(
    TensorWrapper<float> *k_cache_src,
    TensorWrapper<float> *v_cache_src,
    TensorWrapper<int> *context_length,
    TensorWrapper<int> *layer_id,
    TensorWrapper<float> *k_cache_dst,
    TensorWrapper<float> *v_cache_dst
);

template void launchRepeatKVCache(
    TensorWrapper<half> *k_cache_src,
    TensorWrapper<half> *v_cache_src,
    TensorWrapper<int> *context_length,
    TensorWrapper<int> *layer_id,
    TensorWrapper<half> *k_cache_dst,
    TensorWrapper<half> *v_cache_dst
);