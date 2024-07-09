#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/rmsnorm_kernel.h"

// Bug fixes:
// 1. Second warpReduceSum returns 0, because blockDim.x < 32, blockDim.x / 32 = 0.
// 2. Output buffer values remain unchanged because write into the output address was unsuccessful.
// 3. Output buffer's first 32 values are correct, the rest are wrong. Ensure correct row stride when using vectors.
// 4. Add __syncthreads() in fp32/fp16 kernel to get the correct results. Missing it results in some values being zero.

template<typename T>
__device__ T warpReduceSum(T val) {
    #pragma unroll
    for(int i = 16; i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

template<typename T>
__device__ T blockReduceSum(T val) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warpnum = (blockDim.x + 31) / 32;
    static __shared__ T warp_sum[64];
    val = warpReduceSum<T>(val);
    if(lane_id == 0) {
        warp_sum[warp_id] = val;
    }
    __syncthreads();

    T sum = tid < warpnum ? warp_sum[tid] : 0;
    sum = warpReduceSum<T>(sum);
    return sum;
}

// Kernel used at the beginning of every decoder layer and the end of 32 decoder layers.
// Assumes head size can be divided by 4 and 2.
template <typename T>
__global__ void RMSNorm(T* decoder_out, T* decoder_residual, T* scale, float eps, int num_tokens, int hidden_units) {
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    float thread_sum = 0.0f;
    Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_units);
    Vec_t* rsd = reinterpret_cast<Vec_t*>(decoder_residual + blockIdx.x * hidden_units);

    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t vec = dout[idx];
        rsd[idx] = vec;
        thread_sum += vec.x * vec.x;
        thread_sum += vec.y * vec.y;
        thread_sum += vec.z * vec.z;
        thread_sum += vec.w * vec.w;
    }

    thread_sum = blockReduceSum<float>(thread_sum);
    __shared__ float inv_mean;
    if (threadIdx.x == 0) {
        inv_mean = rsqrtf((float)thread_sum / hidden_units + eps);
    }
    __syncthreads();

    Vec_t* s = reinterpret_cast<Vec_t*>(scale);
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t out = dout[idx];
        dout[idx].x = out.x * inv_mean * s[idx].x;
        dout[idx].y = out.y * inv_mean * s[idx].y;
        dout[idx].z = out.z * inv_mean * s[idx].z;
        dout[idx].w = out.w * inv_mean * s[idx].w;
    }
}

template <>
__global__ void RMSNorm(half* decoder_out, half* decoder_residual, half* scale, float eps, int num_tokens, int hidden_units) {
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t* s;
    Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);
    Vec_t* rsd = decoder_residual ? reinterpret_cast<Vec_t*>(decoder_residual + batch_id * hidden_units) : nullptr;
    float thread_accm = 0.0f;

    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t out = dout[i];
        if (rsd) {
            rsd[i] = out;
        }
        thread_accm += __half2float(out.x) * __half2float(out.x);
        thread_accm += __half2float(out.y) * __half2float(out.y);
    }

    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0) {
        inv_fenmu = rsqrtf(float(blocksum / hidden_units) + eps);
    }
    __syncthreads();

    s = reinterpret_cast<Vec_t*>(scale);
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t dout_h2 = dout[i];
        dout[i].x = s[i].x * __float2half(__half2float(dout_h2.x) * inv_fenmu);
        dout[i].y = s[i].y * __float2half(__half2float(dout_h2.y) * inv_fenmu);
    }    
}

template<typename T>
void launchRMSNorm(TensorWrapper<T>* decoder_out, TensorWrapper<T>* decoder_residual, 
                   LayerNormWeight<T>& attn_norm_weight, 
                   float eps, bool is_last) {
    int num_tokens = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int vec_size = Vec<T>::size;
    int num_threads = hidden_units / 4;
    T* rsd = decoder_residual->data;
    dim3 grid(num_tokens);
    dim3 block(num_threads);

    RMSNorm<T><<<grid, block>>>(decoder_out->data, rsd, attn_norm_weight.gamma, eps, num_tokens, hidden_units);

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(decoder_out->data);
#endif
}

template void launchRMSNorm(TensorWrapper<float>* decoder_out, TensorWrapper<float>* decoder_residual, 
                            LayerNormWeight<float>& attn_norm_weight, 
                            float eps, bool is_last);
template void launchRMSNorm(TensorWrapper<half>* decoder_out, TensorWrapper<half>* decoder_residual, 
                            LayerNormWeight<half>& attn_norm_weight, 
                            float eps, bool is_last);
