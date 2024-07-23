#include <iostream>
#include <stdio.h>
#include <math.h>
// #include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/includes/fused_decoder_self_attention.h"

// kv cache shape = [numlayers, bs, kv head num, max_seq_len, head size]
// bug1: scale's dtype must be float, not int
// bug2: mha_kernel_params struct's pointer is on CPU, not GPU, which causes we don't run the CUDA kernel, so add cudacheck is a must!
// bug3: blockreduce res should use tid=0 to write into smem
// bug4: GQA, kv_head_num brd to head_num, we can automatically do this by head id index like lmdeploy
// half or float version: the logits and mha output both are fp32 type, q k v are all accessed vectorizedly

template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<typename T>
__device__ T blockReduceSum(T val) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_nums = (blockDim.x + 31) >> 5;
    static __shared__ T warpsum[32]; // why add static? or will report incomplete type

    val = warpReduceSum<T>(val);
    if (lane_id == 0) {
        warpsum[warp_id] = val;
    }
    __syncthreads();

    T warp_val = tid < warp_nums ? warpsum[tid] : static_cast<T>(0.0f);
    return warpReduceSum<T>(warp_val);
}

template<typename T>
__device__ T warpReduceMax(T val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_nums = (blockDim.x + 31) >> 5;
    static __shared__ T warpmax[32];

    val = warpReduceMax(val);
    if (lane_id == 0) {
        warpmax[warp_id] = val;
    }
    __syncthreads();

    T warp_val = tid < warp_nums ? warpmax[tid] : static_cast<T>(0);
    return warpReduceMax(warp_val);
}

/*
    dim3 grid(head_num * batch_size);
    dim3 block(head_size);
*/
template<typename T>
__global__ void maskedMultiHeadAttention(
    T *q, // [bs, head_num, 1, head_size]
    T *k, 
    T *v, 
    T *qkv_bias, // bias, [qkv_head_num * head_size]
    T *k_cache, // [layer_num, batch_size, kv_num_heads, max_seq_len, head_size]
    T *v_cache, 
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
    const int tid = threadIdx.x;
    const int q_batch_id = blockIdx.x / head_num;
    const int q_head_id = blockIdx.x % head_num;

    const int kv_batch_id = q_batch_id;
    const int repeated_kv_heads = head_num / kv_head_num;
    const int kv_head_id = q_head_id / repeated_kv_heads;

    const int batch_stride = head_num * head_size;
    const int kv_batch_stride = kv_head_num * head_size;
    const int head_stride = head_size;
    const int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    const int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;

    const int vec_size = Vec<T>::size;
    const int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
    const int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;
    const int cache_offset = kv_batch_id * kv_head_num * max_seq_len * head_size +
                             kv_head_id * max_seq_len * head_size + tid * vec_size;
    const int step_stride = head_size;
    const float scale = rsqrt(static_cast<float>(head_size));
    using Vec_t = typename Vec<T>::Type;
    Vec_t qvec, kvec, vvec;
    const T *const q_mem = q;
    const T *const k_mem = k;
    const T *const v_mem = v;

    if (tid * vec_size < head_size) {
        qvec = *reinterpret_cast<const Vec_t *>(const_cast<T *>(q_mem + q_offset_vec));
        kvec = *reinterpret_cast<const Vec_t *>(const_cast<T *>(k_mem + k_offset_vec));
        vvec = *reinterpret_cast<const Vec_t *>(const_cast<T *>(v_mem + k_offset_vec));
    }

    extern __shared__ char sqk[];
    T *const sq_scalar = reinterpret_cast<T *>(sqk);
    float *const logits = reinterpret_cast<float *>(sq_scalar + head_size);
    Vec_t *const sq = reinterpret_cast<Vec_t *>(sq_scalar);

    if (tid * vec_size < head_size) {
        sq[tid] = qvec;
    }
    __syncthreads();

    const float zero = 0.0f;
    const Vec_t zero_f4 = scalar_cast_vec<Vec_t, T>(zero);
    const float4 scale_f4 = scalar_cast_vec<float4, float>(scale);

    for (int iter = 0; iter < step; ++iter) {
        Vec_t kvec_qk = tid * vec_size < head_size ? *reinterpret_cast<Vec_t *>(&k_cache[iter * step_stride + cache_offset]) : zero_f4;
        if (iter == step - 1 && tid * vec_size < head_size) {
            *reinterpret_cast<Vec_t *>(&k_cache[iter * step_stride + cache_offset]) = kvec;
            kvec_qk = kvec;
        }
        Vec_t qk = zero_f4;
        qk.x = (tid * vec_size < head_size) ? sq[tid].x * kvec_qk.x * scale_f4.x : zero;
        qk.y = (tid * vec_size < head_size) ? sq[tid].y * kvec_qk.y * scale_f4.y : zero;
        qk.z = (tid * vec_size < head_size) ? sq[tid].z * kvec_qk.z * scale_f4.z : zero;
        qk.w = (tid * vec_size < head_size) ? sq[tid].w * kvec_qk.w * scale_f4.w : zero;
        T qk_acc = qk.x + qk.y + qk.z + qk.w;
        T attn_score = blockReduceSum<T>(qk_acc);
        if (tid == 0) {
            logits[iter] = attn_score;
        }
        __syncthreads();
    }

    const T local_logits = tid < step ? static_cast<T>(logits[tid]) : 0;
    __shared__ float row_max, fenmu;
    const T block_max = blockReduceMax<T>(local_logits);
    if (tid == 0) {
        row_max = block_max;
    }
    __syncthreads();

    const T fenzi = tid < step ? expf(logits[tid] - row_max) : 0;
    const T block_fenmu = blockReduceSum<T>(fenzi);
    if (tid == 0) {
        fenmu = block_fenmu + 1e-6f;
    }
    __syncthreads();

    if (tid < step) {
        logits[tid] = static_cast<T>(fenzi / fenmu);
    }
    __syncthreads();

    if (tid * vec_size < head_size) {
        Vec_t O = scalar_cast_vec<Vec_t, T>(0.0f);
        for (int iter = 0; iter < step; ++iter) {
            Vec_t vvec_qkv = *reinterpret_cast<Vec_t *>(&v_cache[iter * step_stride + cache_offset]);
            if (iter == step - 1) {
                *reinterpret_cast<Vec_t *>(&v_cache[iter * step_stride + cache_offset]) = vvec;
                vvec_qkv = vvec;
            }
            O.x += vvec_qkv.x * logits[iter];
            O.y += vvec_qkv.y * logits[iter];
            O.z += vvec_qkv.z * logits[iter];
            O.w += vvec_qkv.w * logits[iter];
        }
        *reinterpret_cast<Vec_t *>(&mha_output[q_offset_vec]) = O;
    }
}

template<>
__global__ void maskedMultiHeadAttention(
    half *q, half *k, half *v, half *qkv_bias, half *k_cache, half *v_cache, half *mha_output,
    const int batch_size, const int head_num, const int kv_head_num,
    const int max_seq_len, const int head_size, const int step,
    const int rotary_embedding_dim, const float rotary_embedding_base) {
    // Note: To sync with newest fp32 MHA
}

template<typename T>
void launchDecoderMaskedMultiHeadAttention(
    TensorWrapper<T> *qkv_buf, // [bs, qkv_head_num, 1, head_size]
    BaseWeight<T> *qkv, // bias, [qkv_head_num * head_size]
    TensorWrapper<int> *layer_id, // [layer_num]
    TensorWrapper<T> *k_cache, // [layer_num, batch_size, kv_num_heads, max_seq_len, head_size]
    TensorWrapper<T> *v_cache, // [layer_num, batch_size, kv_num_heads, max_seq_len, head_size]
    TensorWrapper<bool> *finished, // [batch_size]
    TensorWrapper<int> *step, // ?[max_seq_len]
    TensorWrapper<T> *mha_output, // [batch_size, num_heads, head_size]
    LlamaAttentionStaticParams *static_params
) {
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
    const T *const qkv_data = qkv_buf->data;
    const T *const q = qkv_data;
    const T *const k = qkv_data + head_num * head_size;
    const T *const v = qkv_data + (head_num + kv_head_num) * head_size;

    const int rotary_embedding_dim = static_params->rotary_embedding_dim;
    const float rotary_embedding_base = static_params->rotary_embedding_base;

    dim3 grid(head_num * batch_size);
    dim3 block(head_size);

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
