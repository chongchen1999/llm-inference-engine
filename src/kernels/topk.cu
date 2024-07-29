#include <float.h> // FLT_MIN
#include <cuda.h>
#include <iostream>
#include "includes/topk.h"
#include <cub/cub.cuh>

// Note: a and b are two topK reductions outputting a single topK

template <typename T, int K>
struct TopkReduceFunctor {
    __device__ TopK<T, K> operator()(const TopK<T, K>& a, const TopK<T, K>& b) {
        TopK<T, K> res = a;
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            res.insertQueue(b.val[i], b.id[i]);
        }
        return res;
    }
};

// gridsize: bs * beam_num * blocks_per_beam 
// blocksize: 1024
// shape infer: [bs, beam_num, vocab size] => [bs, beam_num, blocks_per_beam, K]
template<typename T, int K, int block_size, int blocks_per_beam>
__global__ void reduceTopK1(
    const T *const probs,
    const int vocab_size, 
    int *const topK_ids,
    T *const topK_vals
) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_id = blockIdx.x / blocks_per_beam;
    const int beam_lane = blockIdx.x % blocks_per_beam;
    const int cur_block_offset = beam_lane * block_size;
    const int blocks_stride = blocks_per_beam * block_size;
    const int base_data_offset = row_id * vocab_size;

    TopK<T, K> thread_topK;
    thread_topK.init();

    // Thread local reduce
    #pragma unroll
    for (int data_id = cur_block_offset + tid; data_id < vocab_size; data_id += blocks_stride) {
        const int data_offset = base_data_offset + data_id;
        const T data = probs[data_offset];
        thread_topK.insertQueue(data, data_id);
    }

    // Block reduce
    using BlockReduceTopk = cub::BlockReduce<TopK<T, K>, block_size>;
    __shared__ typename BlockReduceTopk::TempStorage temp_storage;

    TopK<T, K> block_topk = BlockReduceTopk(temp_storage).Reduce(thread_topK, TopkReduceFunctor<T, K>());

    if (tid == 0) {
        #pragma unroll
        for (int k_offset = 0; k_offset < K; ++k_offset) {
            const int offset = row_id * blocks_per_beam * K + beam_lane * K + k_offset;
            topK_vals[offset] = block_topk.val[k_offset];
            topK_ids[offset] = block_topk.id[k_offset];
        }
    }
}

// shape infer: [bs, beam_num, blocks_per_beam, K] => [bs, beam_num, K]
// ids are global word ids from beam width * vocab size
// gridSize = bs * beam_num
// blockSize = 128
template<typename T, int K, int block_size, int blocks_per_beam>
__global__ void reduceTopK2(
    const int *topK_ids, // [bs, beam_num, blocks_per_beam, K]
    const T *topK_vals, // [bs, beam_num, blocks_per_beam, K]
    int *final_topK_ids, // [bs, beam_num, K]
    T *final_topK_vals // [bs, beam_num, K]
) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_id = blockIdx.x;
    const int reduced_beam_size = blocks_per_beam * K;

    TopK<T, K> thread_topK;
    // Thread local reduce
    #pragma unroll
    for (int i = tid; i < reduced_beam_size; i += blockDim.x) {
        const int data_offset = blockIdx.x * blocks_per_beam * K + i;
        thread_topK.insertQueue(topK_vals[data_offset], topK_ids[i]);
    }

    // Block reduce
    typedef cub::BlockReduce<TopK<T, K>, block_size> BlockReduceTopk;
    __shared__ typename BlockReduceTopk::TempStorage temp_storage;
    TopK<T, K> block_topk = BlockReduceTopk(temp_storage).Reduce(thread_topK, TopkReduceFunctor<T, K>());

    if (tid == 0) {
        for (int k_offset = 0; k_offset < K; ++k_offset) {
            const int offset = blockIdx.x * K + k_offset;
            final_topK_vals[offset] = block_topk.val[k_offset];
            final_topK_ids[offset] = block_topk.id[k_offset];
        }
    }
}

template <typename T>
void launchTopKForBeamSearch(
    TensorWrapper<T> *probs, // [batch_size * beam_num, vocab_size]
    TensorWrapper<int> *topk_ids, // [batch_size, beam_num, blocks_per_beam, K]
    TensorWrapper<T> *topk_vals, // [batch_size, beam_num, blocks_per_beam, K]
    TensorWrapper<int> *final_topk_ids, // [bs, beam_num, K]
    TensorWrapper<T> *final_topk_vals // [bs, beam_num, K]
) {
    // Support both beam search and sampling topK by integrating beam width into batch size
    // Shape of probs is [bs * bw, vocab_size]
    const int bsxbm = probs->shape[0];
    const int vocab_size = probs->shape[1];
    const int blocks_per_beam = 8;
    const int beam_num = 1;
    const int K = 5;

    // Buffer size
    const int topK_val_buf_size = bsxbm * blocks_per_beam * K;
    const int topK_ids_buf_size = bsxbm * blocks_per_beam * K;
    const int final_topK_val_buf_size = bsxbm * K;

    T *const topK_vals = topk_vals->data;
    int *const topK_ids = topk_ids->data;
    T *const final_topK_vals = final_topk_vals->data;
    int *const final_topK_ids = final_topk_ids->data;

    // GPU configuration
    const int block_num1 = bsxbm * blocks_per_beam;
    dim3 grid_round1(block_num1);
    dim3 block_round1(1024);
    reduceTopK1<T, K, 1024, blocks_per_beam><<<grid_round1, block_round1>>>(probs->data, vocab_size, topK_ids, topK_vals);

    const int block_num2 = bsxbm;
    dim3 grid_round2(block_num2);
    dim3 block_round2(128);
    reduceTopK2<T, K, 128, blocks_per_beam><<<grid_round2, block_round2>>>(topK_ids, topK_vals, final_topK_ids, final_topK_vals);
}

template void launchTopKForBeamSearch<float>(
    TensorWrapper<float> *probs,
    TensorWrapper<int> *topk_ids,
    TensorWrapper<float> *topk_vals,
    TensorWrapper<int> *final_topk_ids,
    TensorWrapper<float> *final_topk_vals
);

template void launchTopKForBeamSearch<half>(
    TensorWrapper<half> *probs,
    TensorWrapper<int> *topk_ids,
    TensorWrapper<half> *topk_vals,
    TensorWrapper<int> *final_topk_ids,
    TensorWrapper<half> *final_topk_vals
);
