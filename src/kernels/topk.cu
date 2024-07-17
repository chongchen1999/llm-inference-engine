#include <float.h> // FLT_MIN
#include <cuda.h>
#include <iostream>
#include "src/kernels/includes/topk.h"
#include <cub/cub.cuh>

// Note: a and b are two topK reductions outputting a single topK
template<typename T, int K>
__device__ topK<T, K> reduce_functor(const topK<T, K>& a, const topK<T, K>& b) {
    topK<T, K> res = a;
    for (int i = 0; i < K; ++i) {
        res.insertHeap(b.val[i], b.id[i]);
    }
    return res;
}

// gridsize: bs * beamwidth * BlockPerBeam 
// blocksize: 256
// shape infer: [bs, beamwidth, vocab size] => [bs, beamwidth, BlockPerBeam, K]
template<typename T, int K, int blockSize, int BlockPerBeam>
__global__ void topK_kernel_round1(const T *probs, const int vocab_size, 
                                   int *topK_ids, T *topK_vals) {
    typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
    __shared__ typename blockreduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int row_id = bid / BlockPerBeam;
    int block_lane = bid % BlockPerBeam;

    topK<T, K> thread_topK;
    thread_topK.init();

    // thread local reduce
    for (int data_id = tid + block_lane * blockSize; data_id < vocab_size; data_id += BlockPerBeam * blockSize) {
        int data_offset = data_id + row_id * vocab_size;
        T data = probs[data_offset];
        thread_topK.insertHeap(data, data_id);
    }

    // block local reduce
    topK<T, K> block_topK = blockreduce(temp_storage).Reduce(thread_topK, reduce_functor<T, K>);

    if (tid == 0) {
        for (int k_offset = 0; k_offset < K; ++k_offset) {
            topK_vals[row_id * BlockPerBeam * K + block_lane * K + k_offset] = block_topK.val[k_offset];
            topK_ids[row_id * BlockPerBeam * K + block_lane * K + k_offset] = block_topK.id[k_offset];
        }
    }
}

// shape infer: [bs, beamwidth, BlockPerBeam, K] => [bs, beamwidth, K]
// ids are global word ids from beam width * vocab size
// gridSize = bs
// blockSize = 256
template<typename T, int K, int blockSize, int BlockPerBeam>
__global__ void topK_kernel_round2(const int *topK_ids, const T *topK_vals,
                                   int *final_topK_ids, T *final_topK_vals) {
    typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
    __shared__ typename blockreduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int row_id = bid;

    topK<T, K> thread_topK;

    // thread local reduce    
    for (int i = tid; i < BlockPerBeam * K; i += blockDim.x) {
        int data_offset = bid * BlockPerBeam * K + i;
        thread_topK.insertHeap(topK_vals[data_offset], topK_ids[i]);
    }

    // block reduce
    topK<T, K> block_topK = blockreduce(temp_storage).Reduce(thread_topK, reduce_functor<T, K>);

    if (tid == 0) {
        for (int k_offset = 0; k_offset < K; ++k_offset) {
            final_topK_vals[bid * K + k_offset] = block_topK.val[k_offset];
            final_topK_ids[bid * K + k_offset] = block_topK.id[k_offset];
        }
    }    
}

template <typename T>
void launchTopKforBeamSearch(TensorWrapper<T> *probs,
                             TensorWrapper<int> *topk_ids,
                             TensorWrapper<T> *topk_vals,
                             TensorWrapper<int> *final_topk_ids,
                             TensorWrapper<T> *final_topk_vals) {
    // Support both beam search and sampling topK by integrating beam width into batch size
    // Shape of probs is [bs * bw, vocab_size]
    int bsxbm = probs->shape[0];
    int vocab_size = probs->shape[1];
    constexpr int BlockPerBeam = 8;
    constexpr int beamwidth = 1;
    constexpr int K = 5;

    // Buffer size
    int topK_val_buf_size = bsxbm * BlockPerBeam * K;
    int topK_ids_buf_size = bsxbm * BlockPerBeam * K;
    int final_topK_val_buf_size = bsxbm * K;

    T *topK_vals = topk_vals->data;
    int *topK_ids = topk_ids->data;
    T *final_topK_vals = final_topk_vals->data;
    int *final_topK_ids = final_topk_ids->data;    

    // GPU configuration
    int maxBlockNums = 1024;
    int BlockNums1 = std::min(bsxbm * BlockPerBeam, maxBlockNums);
    int BlockNums2 = std::min(bsxbm, maxBlockNums);

    dim3 grid_round1(BlockNums1);
    dim3 block_round1(256);
    dim3 grid_round2(BlockNums2);
    dim3 block_round2(256);

    topK_kernel_round1<T, K, 256, BlockPerBeam><<<grid_round1, block_round1>>>(probs->data, vocab_size, topK_ids, topK_vals);
    topK_kernel_round2<T, K, 256, BlockPerBeam><<<grid_round2, block_round2>>>(topK_ids, topK_vals, final_topK_ids, final_topK_vals);
}

template void launchTopKforBeamSearch(TensorWrapper<float> *probs,
                                      TensorWrapper<int> *topk_ids,
                                      TensorWrapper<float> *topk_vals,
                                      TensorWrapper<int> *final_topk_ids,
                                      TensorWrapper<float> *final_topk_vals);

template void launchTopKforBeamSearch(TensorWrapper<half> *probs,
                                      TensorWrapper<int> *topk_ids,
                                      TensorWrapper<half> *topk_vals,
                                      TensorWrapper<int> *final_topk_ids,
                                      TensorWrapper<half> *final_topk_vals);
