#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cstdio>
#include "../utils/cuda_debug_utils.cuh"
#include "includes/decoder_self_attention.cuh"

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
    dim3 grid(head_num * batch_size);
    dim3 block(head_size);
*/
template <typename T>
__global__ void maskedMultiHeadAttention(
    T *q, // [bs, head_num, 1, head_size] or [bs, head_num, head_size]
    T *k, // [bs, kv_head_num, 1, head_size] or [bs, kv_head_num, head_size]
    T *v, // [bs, kv_head_num, 1, head_size] or [bs, kv_head_num, head_size]
    const T *qkv_bias, // bias, [qkv_head_num * head_size] aka [qkv_hidden_units]
    T *k_cache, // [layer_num, batch_size, kv_num_heads, max_seq_len, head_size]
    T *v_cache, // [layer_num, batch_size, kv_num_heads, max_seq_len, head_size]
    T *mha_output, // [batch_size, num_heads, head_size]
    const int batch_size, 
    const int head_num, 
    const int kv_head_num,
    const int max_seq_len, 
    const int head_size, 
    const int step,
    const int rotary_embedding_dim, 
    const float rotary_embedding_base
) {
    const int q_hidden_units_id = blockIdx.x / head_num;
    const int q_head_id = blockIdx.x - q_hidden_units_id * head_num;
    const int tid = threadIdx.x;

    const int kv_hidden_units_id = q_hidden_units_id;
    const int kv_repeats_num = head_num / kv_head_num;
    const int kv_head_id = q_head_id / kv_repeats_num;

    const int q_hidden_units = head_num * head_size;
    const int kv_hidden_units = kv_head_num * head_size;
    const int head_stride = head_size;
    // const int q_offset = q_hidden_units_id * q_hidden_units + q_head_id * head_stride + tid;
    // const int kv_offset = kv_hidden_units_id * kv_hidden_units + kv_head_id * head_stride + tid;

    const int vec_size = Vec<T>::size;
    const int vec_q_offset = q_hidden_units_id * q_hidden_units + q_head_id * head_stride + tid * vec_size;
    const int vec_kv_offset = kv_hidden_units_id * kv_hidden_units + kv_head_id * head_stride + tid * vec_size;

    // [layer_num, batch_size, kv_num_heads, max_seq_len, head_size]
    const int cache_offset = 
        kv_hidden_units_id * kv_head_num * max_seq_len * head_size +
        kv_head_id * max_seq_len * head_size + tid * vec_size;

    const int step_stride = head_size;
    const float scale = rsqrt(static_cast<float>(head_size));
    using Vec_t = typename Vec<T>::Type;

    extern __shared__ char shared_qk[]; //head_size * sizeof(T) + cur_step * sizeof(float);
    T *shared_q = reinterpret_cast<T *>(shared_qk);
    float *logits = reinterpret_cast<float *>(shared_q + head_size);
    Vec_t *vec_shared_q = reinterpret_cast<Vec_t *>(shared_q);

    Vec_t &vec_q = *reinterpret_cast<Vec_t *>(q + vec_q_offset);
    Vec_t &vec_k = *reinterpret_cast<Vec_t *>(k + vec_kv_offset);
    Vec_t &vec_v = *reinterpret_cast<Vec_t *>(v + vec_kv_offset);

    if (tid * vec_size < head_size) {
        if (qkv_bias) {
            const Vec_t q_bias = *reinterpret_cast<const Vec_t *>(qkv_bias + q_head_id * head_size + tid * vec_size);
            const Vec_t k_bias = *reinterpret_cast<const Vec_t *>(qkv_bias + (head_num + kv_head_id) * head_size + tid * vec_size);
            const Vec_t v_bias = *reinterpret_cast<const Vec_t *>(qkv_bias + (head_num + kv_head_num + kv_head_id) * head_size + tid * vec_size);
            VectorizedOperator<Vec_t>::add_assign(vec_q, q_bias);
            VectorizedOperator<Vec_t>::add_assign(vec_k, k_bias);
            VectorizedOperator<Vec_t>::add_assign(vec_v, v_bias);
        }
        vec_shared_q[tid] = vec_q;
    }
    __syncthreads();

    const Vec_t vec_zero = ScalarCast2Vector::scalarCastToVector<Vec_t, float>(0.0f);
    const Vec_t vec_scale = ScalarCast2Vector::scalarCastToVector<Vec_t, float>(scale);

    *reinterpret_cast<Vec_t *>(k_cache + (step - 1) * step_stride + cache_offset) = vec_k;
    #pragma unroll
    for (int kv_id = 0; kv_id < step; ++kv_id) {
        Vec_t vec_cached_k = vec_zero;
        Vec_t vec_qkT = vec_zero; // q * K^T
        if (tid * vec_size < head_size) {
            vec_cached_k = *reinterpret_cast<Vec_t *>(k_cache + kv_id * step_stride + cache_offset);
            vec_qkT = VectorizedOperator<Vec_t>::mul(vec_shared_q[tid], vec_cached_k);
            VectorizedOperator<Vec_t>::mul_assign(vec_qkT, vec_scale);
        }
        
        T qk_acc = vec_qkT.x + vec_qkT.y + vec_qkT.z + vec_qkT.w;
        blockReduce<SumOp, T>(qk_acc);
        if (tid == 0) {
            logits[kv_id] = qk_acc;
        }
        __syncthreads();
    }

    const T local_logit = tid < step ? static_cast<T>(logits[tid]) : static_cast<T>(0);
    __shared__ float row_max, sum_exp;
    T block_max = local_logit;
    blockReduce<MaxOp, T>(block_max);
    if (tid == 0) {
        row_max = block_max;
    }
    __syncthreads();

    const T cur_exp = tid < step ? expf(local_logit - row_max) : static_cast<T>(0);
    T block_sum_exp = cur_exp;
    blockReduce<SumOp, T>(block_sum_exp);
    if (tid == 0) {
        sum_exp = block_sum_exp + 1e-6f;
    }
    __syncthreads();

    if (tid < step) {
        logits[tid] = static_cast<T>(cur_exp / sum_exp);
    }
    __syncthreads();

    if (tid * vec_size < head_size) {
        /*if (!blockIdx.x && !threadIdx.x) {
            printf("got into mha_output STEP 0!\n");
        }*/
        Vec_t vec_attention_score = vec_zero;
        *reinterpret_cast<Vec_t *>(v_cache + (step - 1) * step_stride + cache_offset) = vec_v;

        #pragma unroll
        for (int kv_id = 0; kv_id < step; ++kv_id) {
            Vec_t vec_cached_v = *reinterpret_cast<Vec_t *>(v_cache + kv_id * step_stride + cache_offset);
            VectorizedOperator<Vec_t>::add_assign(
                vec_attention_score, 
                VectorizedOperator<Vec_t>::mul(
                    vec_cached_v, 
                    ScalarCast2Vector::scalarCastToVector<Vec_t, float>(logits[kv_id])
                )
            );
        }

        *reinterpret_cast<Vec_t *>(mha_output + vec_q_offset) = vec_attention_score;
    }
}

template <>
__global__ void maskedMultiHeadAttention(
    half *q, 
    half *k, 
    half *v, 
    const half *qkv_bias, 
    half *k_cache, 
    half *v_cache, 
    half *mha_output,
    const int batch_size, 
    const int head_num, 
    const int kv_head_num,
    const int max_seq_len, 
    const int head_size, 
    const int step,
    const int rotary_embedding_dim, 
    const float rotary_embedding_base
) {
    // Note: To sync with newest fp32 MHA
}

template <typename T>
void launchDecoderMaskedMultiHeadAttention(
    TensorWrapper<T> *qkv_buf, // [bs, qkv_head_num, 1, head_size] or [bs, qkv_head_num, head_size]
    BaseWeight<T> *qkv, // bias, [qkv_head_num * head_size]
    TensorWrapper<int> *layer_id, // [layer_num]
    TensorWrapper<T> *k_cache, // [layer_num, batch_size, kv_num_heads, max_seq_len, head_size]
    TensorWrapper<T> *v_cache, // [layer_num, batch_size, kv_num_heads, max_seq_len, head_size]
    TensorWrapper<bool> *finished, // [batch_size]
    TensorWrapper<int> *step, // [layer_num]
    TensorWrapper<T> *mha_output, // [batch_size, num_heads, head_size] or [bs, q_hidden_units]
    LlamaAttentionStaticParams *static_params
) {
    // printf("get in decoder self attention!\n");
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int head_size = qkv_buf->shape[2];

    const int kv_head_num = k_cache->shape[2];
    const int max_seq_len = k_cache->shape[3];
    const int head_num = qkv_head_num - 2 * kv_head_num;

    const int cur_step = step->getVal();
    const int layer = layer_id->getVal();
    const int layer_offset = layer * max_seq_len * batch_size * kv_head_num * head_size;

    const int smem_size_bytes = head_size * sizeof(T) + cur_step * sizeof(float);
    T *const qkv_data = qkv_buf->data;
    T *const q = qkv_data;
    T *const k = qkv_data + head_num * head_size;
    T *const v = qkv_data + (head_num + kv_head_num) * head_size;

    const int rotary_embedding_dim = static_params->rotary_embedding_dim;
    const float rotary_embedding_base = static_params->rotary_embedding_base;

    dim3 grid(head_num * batch_size);
    dim3 block(head_size);

    // printf("ready to get into fused kernel!\n");
    maskedMultiHeadAttention<T><<<grid, block, smem_size_bytes>>>(
        q, 
        k, 
        v, 
        qkv->bias, 
        k_cache->data + layer_offset,
        v_cache->data + layer_offset, 
        mha_output->data, 
        batch_size,
        head_num, 
        kv_head_num, 
        max_seq_len, 
        head_size, 
        cur_step,
        rotary_embedding_dim,
        rotary_embedding_base
    );
    #ifdef PRINT_DATA
        print_data<<<1, 1>>>(mha_output->data, true);
    #else
    #endif
}

template void launchDecoderMaskedMultiHeadAttention(
    TensorWrapper<float> *qkv_buf,
    BaseWeight<float> *qkv,
    TensorWrapper<int> *layer_id,
    TensorWrapper<float> *k_cache,
    TensorWrapper<float> *v_cache,
    TensorWrapper<bool> *finished,
    TensorWrapper<int> *step,
    TensorWrapper<float> *mha_output,
    LlamaAttentionStaticParams *static_params
);

template void launchDecoderMaskedMultiHeadAttention(
    TensorWrapper<half> *qkv_buf,
    BaseWeight<half> *qkv,
    TensorWrapper<int> *layer_id,
    TensorWrapper<half> *k_cache,
    TensorWrapper<half> *v_cache,
    TensorWrapper<bool> *finished,
    TensorWrapper<int> *step,
    TensorWrapper<half> *mha_output,
    LlamaAttentionStaticParams *static_params
);