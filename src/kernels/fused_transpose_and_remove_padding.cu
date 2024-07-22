#include <iostream>
// #include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/includes/fused_transpose_and_remove_padding.h"

/* 
    [bs, head nums, seqlen, head size] => 
    [bs, seqlen, head nums, head size] => 
    [num tokens, head nums, head size]
    padding_offset.shape = [num_tokens]

    dim3 grid(num_tokens);
    dim3 block(std::min(head_num * head_size, 1024));
*/

template <typename T>
__global__ void fusedTransposeAndRemovePadding(
    const T *src,                // [bs, head nums, seqlen, head size]
    T *dst,                      // [num tokens, head nums, head size]
    const int num_tokens,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int head_size,
    const int *padding_offset    // [num_tokens]
) {
    const int token_id = blockIdx.x;
    // Map to input id
    const int batch_id = (blockIdx.x + padding_offset[token_id]) / seq_len;
    const int seq_id = (blockIdx.x + padding_offset[token_id]) % seq_len;
    const int tid = threadIdx.x;
    const int hidden_units = head_num * head_size;

    // Compute the offset of transpose and remove padding before or after
    const int src_base_offset = batch_id * head_num * seq_len * head_size + seq_id * head_size;
    const int dst_base_offset = token_id * head_num * head_size;

    #pragma unroll
    for (int i = tid; i < hidden_units; i += blockDim.x) {
        const int head_id = i / head_size;
        const int head_inner_id = i % head_size;
        dst[dst_base_offset + i] = src[src_base_offset + head_id * seq_len * head_size + head_inner_id];
    }
}

template <typename T>
void launchFusedTransposeAndRemovePadding(
    TensorWrapper<T> *qkv_buf_with_padding,  // [bs, head nums, seqlen, head size]
    TensorWrapper<int> *padding_offset,      // [num_tokens]
    TensorWrapper<T> *qkv_buf_without_padding // [num tokens, head nums, head size]
) {
    const int batch_size = qkv_buf_with_padding->shape[0];
    const int head_num = qkv_buf_with_padding->shape[1];
    const int seq_len = qkv_buf_with_padding->shape[2];
    const int head_size = qkv_buf_with_padding->shape[3];
    const int num_tokens = qkv_buf_without_padding->shape[0];

    dim3 grid(num_tokens);
    dim3 block(std::min(head_num * head_size, 1024));

    fusedTransposeAndRemovePadding<T><<<grid, block>>>(
        qkv_buf_with_padding->data,
        qkv_buf_without_padding->data,
        num_tokens,
        batch_size,
        seq_len,
        head_num,
        head_size,
        padding_offset->data
    );

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(qkv_buf_without_padding->data);
#endif
}

template void launchFusedTransposeAndRemovePadding(
    TensorWrapper<float> *qkv_buf_with_padding,
    TensorWrapper<int> *padding_offset,
    TensorWrapper<float> *qkv_buf_without_padding
);

template void launchFusedTransposeAndRemovePadding(
    TensorWrapper<half> *qkv_buf_with_padding,
    TensorWrapper<int> *padding_offset,
    TensorWrapper<half> *qkv_buf_without_padding
);
