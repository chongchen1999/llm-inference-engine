#include <iostream>
#include <stdio.h>
#include <math.h>
// #include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/includes/fused_decoder_and_self_attn.h"

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
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int warp_nums = (blockDim.x + 31) >> 5;
    static __shared__ T warpsum[32]; // why add static? or will report incomplete type
    val = warpReduceSum<T>(val);
    if (lane_id == 0) {
        warpsum[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpsum[tid] : (T)0.0f;
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
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31) / 32;
    static __shared__ T warpmax[64];
    val = warpReduceMax(val);
    if (lane_id == 0) {
        warpmax[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpmax[tid] : (T)0;
    return warpReduceMax(warp_val);
}

inline __device__ float2 getRopeFreq(int zid, int rot_embed_dim, float base, float t_step) {
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim); // rot_embed_dim = 128
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef) {
    float2 rot_v;
    rot_v.x = coef.x * data - coef.y * data_rotate;
    rot_v.y = coef.x * data_rotate + coef.y * data;
    return rot_v;
}

template<typename T>
__global__ void masked_MHA_kernel(T *q, T *k, T *v, T *qkv_bias, T *k_cache, T *v_cache, T *mha_output,
                                  const int batch_size, const int head_num, const int kv_head_num,
                                  const int max_seq_len, const int head_size, const int step,
                                  int rotary_embedding_dim, float rotary_embedding_base) {
    int tid = threadIdx.x;
    int q_batch_id = blockIdx.x / head_num;
    int q_head_id = blockIdx.x % head_num;
    int kv_head_id = q_head_id / (head_num / kv_head_num);
    int kv_batch_id = q_batch_id;
    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
    int vec_size = Vec<T>::size;
    int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
    int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;
    int cache_offset = kv_batch_id * kv_head_num * max_seq_len * head_size +
                       kv_head_id * max_seq_len * head_size + tid * vec_size;
    int step_stride = head_size;
    float scale = rsqrt(float(head_size));
    using Vec_t = typename Vec<T>::Type;
    Vec_t qvec, kvec, vvec;
    const T *q_mem = q;
    const T *k_mem = k;
    const T *v_mem = v;
    if (tid * vec_size < head_size) {
        qvec = *reinterpret_cast<Vec_t *>(const_cast<T *>(&q_mem[q_offset_vec]));
        kvec = *reinterpret_cast<Vec_t *>(const_cast<T *>(&k_mem[k_offset_vec]));
        vvec = *reinterpret_cast<Vec_t *>(const_cast<T *>(&v_mem[k_offset_vec]));
    }
    extern __shared__ char sqk[];
    T *sq_scalar = reinterpret_cast<T *>(sqk);
    float *logits = reinterpret_cast<float *>(sq_scalar + head_size);
    Vec_t *sq = reinterpret_cast<Vec_t *>(sq_scalar);
    if (tid * vec_size < head_size) {
        sq[tid] = qvec;
    }
    __syncthreads();
    float zero = 0.0f;
    Vec_t zero_f4 = scalar_cast_vec<Vec_t, T>(zero);
    float4 scale_f4 = scalar_cast_vec<float4, float>(scale);
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
    T local_logits = tid < step ? (T)logits[tid] : 0;
    __shared__ float row_max, fenmu;
    T block_max = blockReduceMax<T>(local_logits);
    if (tid == 0) {
        row_max = block_max;
    }
    __syncthreads();
    T fenzi = tid < step ? expf(logits[tid] - row_max) : 0;
    T block_fenmu = blockReduceSum<T>(fenzi);
    if (tid == 0) {
        fenmu = block_fenmu + 1e-6;
    }
    __syncthreads();
    if (tid < step) {
        logits[tid] = (T)(fenzi / fenmu);
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
__global__ void masked_MHA_kernel(half *q, half *k, half *v, half *qkv_bias, half *k_cache, half *v_cache, half *mha_output,
                                  const int batch_size, const int head_num, const int kv_head_num,
                                  const int max_seq_len, const int head_size, const int step,
                                  int rotary_embedding_dim, float rotary_embedding_base) {
    // () note: to sync with newest fp32 mha
}

template<typename T>
void launchDecoderMaskedMHA(TensorWrapper<T> *qkv_buf, BaseWeight<T> &qkv, TensorWrapper<int> *layer_id,
                            TensorWrapper<T> *k_cache, TensorWrapper<T> *v_cache, TensorWrapper<bool> *finished,
                            TensorWrapper<int> *step, TensorWrapper<T> *mha_output,
                            LlamaAttentionStaticParams &static_params) {
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int kv_head_num = k_cache->shape[2];
    const int max_seq_len = k_cache->shape[3];
    int head_num = qkv_head_num - 2 * kv_head_num;
    const int head_size = qkv_buf->shape[2];
    const int cur_step = step->getVal();
    const int layer = layer_id->getVal();
    const int layer_offset = layer * max_seq_len * batch_size * kv_head_num * head_size;
    size_t smem_size_bytes = head_size * sizeof(T) + cur_step * sizeof(float);
    T *qkv_data = qkv_buf->data;
    T *q = qkv_data;
    T *k = qkv_data + head_num * head_size;
    T *v = qkv_data + (head_num + kv_head_num) * head_size;
    int rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    dim3 grid(head_num * batch_size);
    dim3 block(head_size); // vec size = 4 for fp32
    masked_MHA_kernel<T><<<grid, block, smem_size_bytes>>>(q, k, v, qkv.bias, k_cache->data + layer_offset,
                                                           v_cache->data + layer_offset, mha_output->data, batch_size,
                                                           head_num, kv_head_num, max_seq_len, head_size, cur_step,
                                                           rotary_embedding_dim, rotary_embedding_base);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(mha_output->data, true);
#else
#endif
}

template void launchDecoderMaskedMHA(TensorWrapper<float> *qkv_buf, BaseWeight<float> &qkv, TensorWrapper<int> *layer_id,
                                     TensorWrapper<float> *k_cache, TensorWrapper<float> *v_cache,
                                     TensorWrapper<bool> *finished, TensorWrapper<int> *step,
                                     TensorWrapper<float> *mha_output, LlamaAttentionStaticParams &static_params);

template void launchDecoderMaskedMHA(TensorWrapper<half> *qkv_buf, BaseWeight<half> &qkv, TensorWrapper<int> *layer_id,
                                     TensorWrapper<half> *k_cache, TensorWrapper<half> *v_cache,
                                     TensorWrapper<bool> *finished, TensorWrapper<int> *step,
                                     TensorWrapper<half> *mha_output, LlamaAttentionStaticParams &static_params);
