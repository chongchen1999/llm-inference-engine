#include "includes/scale_and_mask_and_softmax.cuh"
#include "../utils/tensor.h"
#include "../utils/cuda_debug_utils.cuh"
#include <float.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <functional>

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
    static const T identity = static_cast<T>(0);
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a > b ? a : b;
    }
    static const T identity = static_cast<T>(-1e9);
};

template <template <typename> class Operator, typename T>
__device__ __forceinline__ void warpReduce(T &val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = Operator<T>()(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
}

template <template <typename> class Operator, typename T>
__device__ void blockReduce(T &val) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_nums = (blockDim.x + 31) >> 5;
    __shared__ T warp[32]; // threads in a block must be less than 1024

    warpReduce<Operator, T>(val);
    if (lane_id == 0) {
        warp[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < warp_nums ? warp[lane_id] : Operator<T>::identity;
        warpReduce<Operator, T>(val);
    }
}

/*
    Q: [bs, head_num, max_q_len, head_size]
    K: [bs, head_num, max_k_len, head_size]
    Q(KT): [bs, head_num, max_q_len, max_k_len]
    softmax(scale * Q(KT)): [bs, head_num, max_q_len, max_k_len]

    dim3 grid(q_length, batch_size, head_nums);
    dim3 block((k_length + 32 - 1) / 32 * 32);
*/

template <typename T, int nums_per_thread>
__global__ void fusedScaleMaskAndSoftmax(
    T *attention_weights,
    const T *qk,
    const T *mask,
    const int batch_size,
    const int head_num,
    const int q_len,
    const int k_len,
    const float scale
) {
    // const int q_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;
    const int tid = threadIdx.x;
    if (tid >= k_len) {
        return;
    }

    __shared__ float inv_sum, shared_max;

    #pragma unroll
    for (int row_start = blockIdx.x; row_start < q_len; row_start += gridDim.x) {
        const int qk_base_offset = batch_id * head_num * q_len * k_len + head_id * q_len * k_len + row_start * k_len + tid;
        const int mask_base_offset = batch_id * q_len * k_len + row_start * k_len + tid;

        T qk_data = static_cast<T>(0);
        T mask_data = static_cast<T>(0);
        T thread_max = FLT_MIN;
        T data[nums_per_thread];
        
        #pragma unroll
        for (int col_start = 0; col_start < nums_per_thread; ++col_start) {
            qk_data = qk[qk_base_offset + col_start * blockDim.x];
            mask_data = mask[mask_base_offset + col_start * blockDim.x];
            data[col_start] = scale * qk_data + (1.0f - mask_data) * (-10000.0f);
            thread_max = fmax(data[col_start], thread_max);
        }

        blockReduce<MaxOp, T>(thread_max);
        if (tid == 0) {
            shared_max = thread_max;
        }
        __syncthreads();

        T thread_sum = 0.0f;
        #pragma unroll
        for (int col_start = 0; col_start < nums_per_thread; ++col_start) {
            data[col_start] = expf(data[col_start] - shared_max);
            thread_sum += data[col_start];
        }

        blockReduce<SumOp, T>(thread_sum);
        if (threadIdx.x == 0) {
            inv_sum = 1.0f / (thread_sum + 1e-6f);
        }
        __syncthreads();

        #pragma unroll
        for (int col_start = 0; col_start < nums_per_thread; ++col_start) {
            attention_weights[qk_base_offset + col_start * blockDim.x] = (data[col_start] * inv_sum);
        }
    }
}

template <int nums_per_thread>
__global__ void fusedScaleMaskAndSoftmax_half(
    half *attention_weights,
    const half *qk,
    const half *mask,
    const int batch_size,
    const int head_num,
    const int q_len,
    const int k_len,
    const float scale
) {
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;

    Vec_t *attn_score_vec = reinterpret_cast<Vec_t *>(attention_weights);
    const Vec_t *qk_buf_vec = reinterpret_cast<const Vec_t *>(qk);
    const Vec_t *attn_mask_vec = reinterpret_cast<const Vec_t *>(mask);
    Vec_t ONE = ScalarCast2Vector::scalarCastToVector<Vec_t>(__float2half(1.0f));
    Vec_t NEG_INF = ScalarCast2Vector::scalarCastToVector<Vec_t>(__float2half(-10000.0f));
    Vec_t scale_vec = ScalarCast2Vector::scalarCastToVector<Vec_t>(__float2half(scale));

    __shared__ float inv_sum, s_max;
    if (threadIdx.x * vec_size >= k_len) {
        return;
    }

    #pragma unroll
    for (int row_start = blockIdx.x; row_start < q_len; row_start += gridDim.x) {
        int qk_offset = 0;
        int mask_offset = 0;
        Vec_t qk_data;
        Vec_t mask_data;
        float thread_max = FLT_MIN;
        Vec_t data[nums_per_thread];

        for (int col_start = 0; col_start < nums_per_thread; ++col_start) {
            qk_offset = batch_id * head_num * q_len * k_len / 2 +
                        head_id * q_len * k_len / 2 +
                        row_start * k_len / 2 +
                        col_start * blockDim.x + threadIdx.x;
            qk_data = qk_buf_vec[qk_offset];

            mask_offset = batch_id * q_len * k_len / 2 +
                          row_start * k_len / 2 +
                          col_start * blockDim.x + threadIdx.x;
            mask_data = attn_mask_vec[mask_offset];
            Vec_t mask_vec_reg = __hmul2(__hsub2(ONE, mask_data), NEG_INF);

            data[col_start] = __hadd2(__hmul2(scale_vec, qk_data), mask_vec_reg);
            thread_max = fmax(fmax((float)data[col_start].x, (float)data[col_start].y), thread_max);
        }

        blockReduce<MaxOp, float>(thread_max);
        if (threadIdx.x == 0) {
            s_max = thread_max;
        }
        __syncthreads();

        float thread_sum = 0.0f;
        #pragma unroll
        for (int col_start = 0; col_start < nums_per_thread; ++col_start) {
            data[col_start] = h2exp(__hsub2(data[col_start], ScalarCast2Vector::scalarCastToVector<Vec_t>(s_max)));
            thread_sum += (float)(__hadd(data[col_start].x, data[col_start].y));
        }

        blockReduce<SumOp, float>(thread_sum);
        if (threadIdx.x == 0) {
            inv_sum = 1 / (thread_sum + 1e-6f);
        }
        __syncthreads();

        #pragma unroll
        for (int col_start = 0; col_start < nums_per_thread; ++col_start) {
            qk_offset = batch_id * head_num * q_len * k_len / 2 +
                        head_id * q_len * k_len / 2 +
                        row_start * k_len / 2 +
                        col_start * blockDim.x + threadIdx.x;
            attn_score_vec[qk_offset] = __hmul2(data[col_start], ScalarCast2Vector::scalarCastToVector<Vec_t>(inv_sum));
        }
    }
}

template <typename T>
void launchFusedScaleMaskAndSoftmax(
    TensorWrapper<T> *qk,
    TensorWrapper<T> *mask,
    TensorWrapper<T> *attention_weights,
    float scale
) {
    const int q_length = qk->shape[2];
    const int batch_size = qk->shape[0];
    const int head_nums = qk->shape[1];
    const int k_length = qk->shape[3];

    dim3 grid(q_length, batch_size, head_nums);
    dim3 block((k_length + 31) / 32 * 32);

    if (block.x > 2048 && block.x <= 4096) {
        block.x /= 4;
        block.x = (block.x + 31) / 32 * 32;
        assert(block.x < 1024);
        fusedScaleMaskAndSoftmax<T, 4><<<grid, block>>>(
            attention_weights->data,
            qk->data,
            mask->data,
            batch_size,
            head_nums,
            q_length,
            k_length,
            scale
        );
    } else if (block.x > 1024) {
        block.x /= 2;
        block.x = (block.x + 32 - 1) / 32 * 32;
        assert(block.x < 1024);
        fusedScaleMaskAndSoftmax<T, 2><<<grid, block>>>(
            attention_weights->data,
            qk->data,
            mask->data,
            batch_size,
            head_nums,
            q_length,
            k_length,
            scale
        );
    } else {
        block.x = (block.x + 32 - 1) / 32 * 32;
        assert(block.x < 1024);
        fusedScaleMaskAndSoftmax<T, 1><<<grid, block>>>(
            attention_weights->data,
            qk->data,
            mask->data,
            batch_size,
            head_nums,
            q_length,
            k_length,
            scale
        );
    }

    #ifdef PRINT_DATA
        print_data<<<1, 1>>>(attention_weights->data);
    #else
    #endif
}

template <>
void launchFusedScaleMaskAndSoftmax(
    TensorWrapper<half> *qk,
    TensorWrapper<half> *mask,
    TensorWrapper<half> *attention_weights,
    float scale
) {
    const int q_length = qk->shape[2];
    const int batch_size = qk->shape[0];
    const int head_nums = qk->shape[1];
    const int k_length = qk->shape[3];

    LLM_CHECK_WITH_INFO(k_length % 2 == 0, "Currently, K_len should be divided by 2 under half type!");

    dim3 grid(q_length, batch_size, head_nums);
    dim3 block((k_length + 31) / 32 * 32);

    if (block.x > 2048 && block.x <= 4096) {
        block.x /= 4;
        block.x = (block.x + 31) / 32 * 32;
        assert(block.x < 1024);
        fusedScaleMaskAndSoftmax_half<4><<<grid, block>>>(
            attention_weights->data,
            qk->data,
            mask->data,
            batch_size,
            head_nums,
            q_length,
            k_length,
            scale
        );
    } else if (block.x > 1024) {
        block.x /= 2;
        block.x = (block.x + 32 - 1) / 32 * 32;
        assert(block.x < 1024);
        fusedScaleMaskAndSoftmax_half<2><<<grid, block>>>(
            attention_weights->data,
            qk->data,
            mask->data,
            batch_size,
            head_nums,
            q_length,
            k_length,
            scale
        );
    } else {
        block.x = (block.x + 32 - 1) / 32 * 32;
        assert(block.x < 1024);
        fusedScaleMaskAndSoftmax_half<1><<<grid, block>>>(
            attention_weights->data,
            qk->data,
            mask->data,
            batch_size,
            head_nums,
            q_length,
            k_length,
            scale
        );
    }

    #ifdef PRINT_DATA
        print_data<<<1, 1>>>(attention_weights->data);
    #else
    #endif   
}

template void launchFusedScaleMaskAndSoftmax(
    TensorWrapper<float> *qk,
    TensorWrapper<float> *mask,
    TensorWrapper<float> *attention_weights,
    float scale
);

template void launchFusedScaleMaskAndSoftmax(
    TensorWrapper<half> *qk,
    TensorWrapper<half> *mask,
    TensorWrapper<half> *attention_weights,
    float scale
);
