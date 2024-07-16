#include <stdio.h>
// #include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/includes/fused_rsd_and_norm.h"

// bugs1: 2nd warpreducesum returns 0, because blockDim.x < 32, blockDim.x / 32 = 0
// bugs2: output buffer values are the same as ones before call, that's because we didn't successfully write into the output address
// bugs3: output buffer's 1st 32 values are right, the latter is wrong, because when we use vec, the element numbers of a row is hidden_units / vec_size, we should note the row stride to move the pointer carefully
// bugs4: not update residual, new residual = input + residual

template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

// notes:!!!when blocksize < 32, use blockDim.x / 32 to get warp nums is wrong, we should ceil it instead
template<typename T>
__device__ T blockReduceSum(T val) {
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int laneid = tid & 31;
    int warpnum = (blockDim.x + 31) >> 5;
    static __shared__ T warpsum[32]; // threads <= 1024
    val = warpReduceSum<T>(val);
    if (laneid == 0) {
        warpsum[wid] = val;
    }
    __syncthreads();

    T sum = tid < warpnum ? warpsum[tid] : (T)0.0f;
    sum = warpReduceSum<T>(sum); // though 0th own the sum, but don't need to shfl sync
    return sum;
}

// 1. This kernel is used after self-attention in every layer
// 2. I allocate thread number by assuming head size can be divided by 4 and 2
template<typename T>
__global__ void FusedAddBiasResidualRMSNorm(T *residual, 
                                            T *decoder_out, // [num tokens, hidden_units]
                                            const T *bias,  // [hidden_units]
                                            const T *scale, // [hidden_units], RMSNorm weights
                                            float eps, // RMSNorm eps
                                            int num_tokens, 
                                            int hidden_units) {
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *vec_residual, *vec_bias, *vec_scale;
    Vec_t tmp;
    Vec_t *vec_decoder_out = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units); // note the offset should divide vec size

    T thread_accm = static_cast<T>(0);
    vec_residual = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units); // note the offset should divide vec size
    if (bias != nullptr) {
        vec_bias = reinterpret_cast<Vec_t *>(const_cast<T *>(bias));
    } 

    #pragma unroll
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        if (residual != nullptr) {
            vec_decoder_out[i].x += vec_residual[i].x;
            vec_decoder_out[i].y += vec_residual[i].y;
            vec_decoder_out[i].z += vec_residual[i].z;
            vec_decoder_out[i].w += vec_residual[i].w;

            // update residual to be used in add residual kernel at the end of every decoder layer
            vec_residual[i].x = vec_decoder_out[i].x;
            vec_residual[i].y = vec_decoder_out[i].y;
            vec_residual[i].z = vec_decoder_out[i].z;
            vec_residual[i].w = vec_decoder_out[i].w;
        }

        // update rsd by rsd + bias when bias is valid
        if (bias != nullptr) {
            vec_decoder_out[i].x += vec_bias[i].x;
            vec_decoder_out[i].y += vec_bias[i].y;
            vec_decoder_out[i].z += vec_bias[i].z;
            vec_decoder_out[i].w += vec_bias[i].w;
        }

        thread_accm += vec_decoder_out[i].x * vec_decoder_out[i].x;
        thread_accm += vec_decoder_out[i].y * vec_decoder_out[i].y;
        thread_accm += vec_decoder_out[i].z * vec_decoder_out[i].z;
        thread_accm += vec_decoder_out[i].w * vec_decoder_out[i].w;
    } // add residual

    // mean(x^2)
    T block_sum = blockReduceSum<T>(thread_accm);
    __shared__ float inv_fenmu;
    if (tid == 0) {
        inv_fenmu = rsqrt(block_sum / hidden_units + eps);
        // debug info: printf("inv_fenmu on GPU is %f\n", inv_fenmu);
    }
    __syncthreads();
    // RMSNorm
    if (scale != nullptr) {
        vec_scale = reinterpret_cast<Vec_t *>(const_cast<T *>(scale));
    }
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        vec_decoder_out[i].x = vec_scale[i].x * vec_decoder_out[i].x * inv_fenmu;
        vec_decoder_out[i].y = vec_scale[i].y * vec_decoder_out[i].y * inv_fenmu;
        vec_decoder_out[i].z = vec_scale[i].z * vec_decoder_out[i].z * inv_fenmu;
        vec_decoder_out[i].w = vec_scale[i].w * vec_decoder_out[i].w * inv_fenmu;
    }
}

template<>
__global__ void FusedAddBiasResidualRMSNorm(half *residual, 
                                            half *decoder_out, // [num tokens, hidden_units]
                                            const half *bias, //[hidden_units]
                                            const half *scale, //[hidden_units], RMSNorm weights
                                            float eps, // RMSNorm eps
                                            int num_tokens, 
                                            int hidden_units) {
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *vec_residual, *vec_bias, *vec_scale;
    Vec_t vec_decoder_out, tmp;
    float thread_accm = 0.0f;
    if (residual != nullptr && bias != nullptr) {
        vec_residual = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units); // note the offset should divide vec size
        vec_bias = reinterpret_cast<Vec_t *>(const_cast<half *>(bias));
    }
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        vec_decoder_out = reinterpret_cast<Vec_t *>(decoder_out)[batch_id * hidden_units / vec_size + i]; // note the offset should divide vec size
        tmp = __hadd2(__hadd2(vec_decoder_out, vec_residual[i]), vec_bias[i]);
        thread_accm += __half2float(tmp.x) * __half2float(tmp.x) + __half2float(tmp.y) * __half2float(tmp.y);
    } // add residual
    // mean(x^2)
    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ Vec_t inv_fenmu;
    if (tid == 0) {
        // debug info: printf("blocksum on GPU is %f\n", blocksum);
        inv_fenmu = scalar_cast_vec<Vec_t>(__float2half(rsqrt(blocksum / hidden_units + eps)));
        // debug info: printf("inv_fenmu on GPU is %f\n", inv_fenmu);
    }
    // RMSNorm
    Vec_t *out = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units); // note before vec the stride is batch_id * hiddenunits w/o / vec_size
    if (scale != nullptr) {
        vec_scale = reinterpret_cast<Vec_t *>(const_cast<half *>(scale));
    }

    #pragma unroll
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        out[i] = __hmul2(__hmul2(vec_scale[i], out[i]), inv_fenmu);
    } 
}

template<typename T>
void launchFusedAddBiasResidualRMSNorm(TensorWrapper<T> *residual, 
                                       TensorWrapper<T> *decoder_out, // [num tokens, hidden_units]
                                       BaseWeight<T> &norm,
                                       T *scale, // RMSNorm weights
                                       float eps) {
    int batch_size = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    T *bias = norm.bias;
    T *gamma = scale;
    int vec_size = Vec<T>::size;
    int num_threads = hidden_units / vec_size; // assume head size can be divided by 4 and 2
    dim3 grid(batch_size);
    dim3 block(num_threads);
    FusedAddBiasResidualRMSNorm<T><<<grid, block>>>(residual->data, 
                                                    decoder_out->data,
                                                    bias,
                                                    gamma,
                                                    eps,
                                                    batch_size,
                                                    hidden_units);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(decoder_out->data);
#else
#endif
}

template void launchFusedAddBiasResidualRMSNorm(TensorWrapper<float> *residual, 
                                                TensorWrapper<float> *decoder_out, // [num tokens, hidden_units]
                                                BaseWeight<float> &norm,
                                                float *scale, // RMSNorm weights
                                                float eps);

template void launchFusedAddBiasResidualRMSNorm(TensorWrapper<half> *residual, 
                                                TensorWrapper<half> *decoder_out, // [num tokens, hidden_units]
                                                BaseWeight<half> &norm,
                                                half *scale, // RMSNorm weights
                                                float eps);
