#include "src/kernels/includes/fused_scale_mask_and_softmax.h"
#include "src/utils/tensor.h"
// #include "src/utils/cuda_debug_utils.cuh"
#include <float.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a > b ? a : b;
    }
};

template <template <typename> class ReductionOp, typename T>
__device__ __forceinline__ T warpReduce(T val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <template <typename> class ReductionOp, typename T>
__device__ T blockReduce(T val) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int warp_nums = (blockDim.x + 31) >> 5;
    static __shared__ T warp[32]; // threads in a block must be less than 1024
    val = warpReduce<ReductionOp, T>(val);
    if (lane_id == 0) {
        warp[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warp[tid] : 0;
    return warpReduce<ReductionOp, T>(warp_val);
}

/*
    Q: [bs, head_num, max_q_len, head_size]
    K: [bs, head_num, max_k_len, head_size]
    Q(KT): [bs, head_num, max_q_len, max_k_len]
    softmax(scale * Q(KT)): [bs, head_num, max_q_len, max_k_len]

dim3 grid(q_length, batch_size, head_nums);
dim3 block((k_length + 32 - 1) / 32 * 32);
*/

template <typename T, int NUMS_PER_THREAD_PER_ROW>
__global__ void fusedScaleMaskAndSoftmax_float(
    T *attn_score,
    T *qk,
    T *mask,
    int batch_size,
    int head_num,
    int q_len,
    int k_len,
    float scale
) {
    const int q_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;
    const int tid = threadIdx.x;
    if (tid >= k_len) {
        return;
    }

    __shared__ float inv_sum, shared_max;

    #pragma unroll
    for (int row_start = blockIdx.x; row_start < q_len; row_start += gridDim.x) {
        const int qk_base_offset = batch_id * head_num * q_len * k_len +
                                   head_id * q_len * k_len +
                                   row_start * k_len + tid;
        const int mask_base_offset = batch_id * q_len * k_len + row_start * k_len + tid;
        T qk_data = static_cast<T>(0);
        T mask_data = static_cast<T>(0);
        T thread_max = FLT_MIN;
        T data[NUMS_PER_THREAD_PER_ROW];
        
        #pragma unroll
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; ++col_start) {
            qk_data = qk[qk_base_offset + col_start * blockDim.x];
            mask_data = mask[mask_base_offset + col_start * blockDim.x];
            data[col_start] = scale * qk_data + (1.0 - mask_data) * (-10000.0f);
            thread_max = fmax(data[col_start], thread_max);
        }

        T max_val = blockReduce<MaxOp, T>(thread_max);
        if (tid == 0) {
            shared_max = max_val;
        }
        __syncthreads();

        T thread_sum = 0.0f;
        #pragma unroll
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; ++col_start) {
            data[col_start] = expf(data[col_start] - shared_max);
            thread_sum += data[col_start];
        }

        T sum = blockReduce<SumOp, T>(thread_sum);
        if (threadIdx.x == 0) {
            inv_sum = 1 / (sum + 1e-6f);
        }
        __syncthreads();

        #pragma unroll
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; ++col_start) {
            attn_score[qk_base_offset + col_start * blockDim.x] = (data[col_start] * inv_sum);
        }
    }
}

template <typename T_half, int NUMS_PER_THREAD_PER_ROW>
__global__ void fusedScaleMaskAndSoftmax_half(
    T_half *attn_score,
    T_half *qk,
    T_half *mask,
    int batch_size,
    int head_num,
    int q_len,
    int k_len,
    float scale
) {
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int vec_size = Vec<T_half>::size;
    using Vec_t = typename Vec<T_half>::Type;

    Vec_t *attn_score_vec = reinterpret_cast<Vec_t *>(attn_score);
    Vec_t *qk_buf_vec = reinterpret_cast<Vec_t *>(qk);
    Vec_t *attn_mask_vec = reinterpret_cast<Vec_t *>(mask);
    Vec_t ONE = scalar_cast2_vector<Vec_t>(__float2half(1.0f));
    Vec_t NEG_INF = scalar_cast2_vector<Vec_t>(__float2half(-10000.0f));
    Vec_t scale_vec = scalar_cast2_vector<Vec_t>(__float2half(scale));

    __shared__ float inv_sum, s_max;
    if (threadIdx.x * vec_size >= k_len) {
        return;
    }

    for (int row_start = blockIdx.x; row_start < q_len; row_start += gridDim.x) {
        int qk_offset = 0;
        int mask_offset = 0;
        Vec_t qk_data;
        Vec_t mask_data;
        float thread_max = FLT_MIN;
        Vec_t data[NUMS_PER_THREAD_PER_ROW];

        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; ++col_start) {
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

        float max_val = blockReduce<MaxOp, float>(thread_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float thread_sum = 0.0f;
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; ++col_start) {
            data[col_start] = h2exp(__hsub2(data[col_start], scalar_cast2_vector<Vec_t>(s_max)));
            thread_sum += (float)(__hadd(data[col_start].x, data[col_start].y));
        }

        float sum = blockReduce<SumOp, float>(thread_sum);
        if (threadIdx.x == 0) {
            inv_sum = 1 / (sum + 1e-6f);
        }
        __syncthreads();

        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; ++col_start) {
            qk_offset = batch_id * head_num * q_len * k_len / 2 +
                        head_id * q_len * k_len / 2 +
                        row_start * k_len / 2 +
                        col_start * blockDim.x + threadIdx.x;
            attn_score_vec[qk_offset] = __hmul2(data[col_start], scalar_cast2_vector<Vec_t>(inv_sum));
        }
    }
}

#define LAUNCH_SOFTMAX(dtype, vec_size)                                                                         \
    if (block.x > 2048 && block.x <= 4096) {                                                                    \
        constexpr int NUMS_PER_THREAD_PER_ROW = 4;                                                              \
        block.x /= 4 * vec_size;                                                                                \
        block.x = (block.x + 31) / 32 * 32;                                                                     \
        assert(block.x < 1024);                                                                                 \
        fusedScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data, \
                                                                                     (dtype *)qk->data,         \
                                                                                     (dtype *)mask->data,       \
                                                                                     batch_size,                \
                                                                                     head_nums,                 \
                                                                                     q_length,                  \
                                                                                     k_length,                  \
                                                                                     scale);                    \
    } else if (block.x > 1024) {                                                                                 \
        constexpr int NUMS_PER_THREAD_PER_ROW = 2;                                                              \
        block.x /= 2 * vec_size;                                                                                \
        block.x = (block.x + 32 - 1) / 32 * 32;                                                                 \
        assert(block.x < 1024);                                                                                 \
        fusedScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data, \
                                                                                     (dtype *)qk->data,         \
                                                                                     (dtype *)mask->data,       \
                                                                                     batch_size,                \
                                                                                     head_nums,                 \
                                                                                     q_length,                  \
                                                                                     k_length,                  \
                                                                                     scale);                    \
    } else {                                                                                                    \
        constexpr int NUMS_PER_THREAD_PER_ROW = 1;                                                              \
        block.x /= vec_size;                                                                                     \
        assert(block.x < 1024);                                                                                 \
        fusedScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data, \
                                                                                     (dtype *)qk->data,         \
                                                                                     (dtype *)mask->data,       \
                                                                                     batch_size,                \
                                                                                     head_nums,                 \
                                                                                     q_length,                  \
                                                                                     k_length,                  \
                                                                                     scale);                    \
    }

template <typename T>
void launchFusedScaleMaskAndSoftmax(
    TensorWrapper<T> *qk,
    TensorWrapper<T> *mask,
    TensorWrapper<T> *attn_score,
    float scale
) {
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    const int q_length = qk->shape[2];
    const int batch_size = qk->shape[0];
    const int head_nums = qk->shape[1];
    const int k_length = qk->shape[3];
    bool is_half = sizeof(T) == 2;

    // TODO: Should enhance it by padding to support odd ones
    if (is_half) {
        LLM_CHECK_WITH_INFO(k_length % 2 == 0, "Currently, K_len should be divided by 2 under half type!");
    }

    dim3 grid(q_length, batch_size, head_nums);
    dim3 block((k_length + 31) / 32 * 32);

    if (is_half) {
        LAUNCH_SOFTMAX(half, 2);
    } else {
        LAUNCH_SOFTMAX(float, 1);
    }

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(attn_score->data);
#else
#endif
}

template void launchFusedScaleMaskAndSoftmax(
    TensorWrapper<float> *qk,
    TensorWrapper<float> *mask,
    TensorWrapper<float> *attn_score,
    float scale
);

template void launchFusedScaleMaskAndSoftmax(
    TensorWrapper<half> *qk,
    TensorWrapper<half> *mask,
    TensorWrapper<half> *attn_score,
    float scale
);
