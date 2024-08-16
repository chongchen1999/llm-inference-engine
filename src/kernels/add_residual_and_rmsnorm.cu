#include <stdio.h>
#include "../utils/cuda_debug_utils.cuh"
#include "includes/add_residual_and_rmsnorm.cuh"



template <typename T>
__device__ __forceinline__ void warpReduceSum(T &val) {
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
}

// Note: When block size < 32, using `blockDim.x / 32` to get warp numbers is incorrect; use `ceil` instead
template <typename T>
__device__ void blockReduceSum(T &val) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_num = (blockDim.x + 31) >> 5;

    __shared__ T warp_sum[32]; // threads <= 1024
    warpReduceSum<T>(val);

    if (lane_id == 0) {
        warp_sum[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < warp_num) ? warp_sum[lane_id] : static_cast<T>(0);
        warpReduceSum<T>(val);
    }
}

/*
    Kernel used after self-attention in every layer.
    Allocate thread number assuming head size can be divided by 4 and 2.
    dim3 grid(num_tokens);
    dim3 block(num_threads);
*/
template <typename T>
__global__ void fusedAddBiasResidualAndRMSNorm(
    T *residual,       // [num tokens, hidden_units]
    T *decoder_out,    // [num tokens, hidden_units]
    const T *bias,     // [hidden_units]
    const T *scale,    // [hidden_units], RMSNorm weights (gamma)
    const float eps,   // RMSNorm epsilon
    const int num_tokens,
    const int hidden_units
) {
    const int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    const int token_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int vec_hidden_units = hidden_units / vec_size;

    Vec_t *vec_residual = (residual != nullptr) ? reinterpret_cast<Vec_t *>(residual + token_id * hidden_units) : nullptr;
    Vec_t *vec_bias = (bias != nullptr) ? reinterpret_cast<Vec_t *>(const_cast<T *>(bias)) : nullptr;
    Vec_t *vec_scale = (scale != nullptr) ? reinterpret_cast<Vec_t *>(const_cast<T *>(scale)) : nullptr;

    Vec_t *vec_decoder_out = reinterpret_cast<Vec_t *>(decoder_out + token_id * hidden_units); // Note the offset

    T block_sum = static_cast<T>(0);

    #pragma unroll
    for (int i = tid; i < vec_hidden_units; i += blockDim.x) {
        Vec_t &cur_vec_decoder_out = vec_decoder_out[i];

        if (vec_residual != nullptr) {
            Vec_t &cur_vec_residual = vec_residual[i];
            cur_vec_decoder_out.x += cur_vec_residual.x;
            cur_vec_decoder_out.y += cur_vec_residual.y;
            cur_vec_decoder_out.z += cur_vec_residual.z;
            cur_vec_decoder_out.w += cur_vec_residual.w;

            // Update residual for use in add residual kernel
            cur_vec_residual.x = cur_vec_decoder_out.x;
            cur_vec_residual.y = cur_vec_decoder_out.y;
            cur_vec_residual.z = cur_vec_decoder_out.z;
            cur_vec_residual.w = cur_vec_decoder_out.w;
        }

        // Update residual by adding bias if bias is valid
        if (vec_bias != nullptr) {
            Vec_t &cur_vec_bias = vec_bias[i];
            cur_vec_decoder_out.x += cur_vec_bias.x;
            cur_vec_decoder_out.y += cur_vec_bias.y;
            cur_vec_decoder_out.z += cur_vec_bias.z;
            cur_vec_decoder_out.w += cur_vec_bias.w;
        }

        block_sum += cur_vec_decoder_out.x * cur_vec_decoder_out.x;
        block_sum += cur_vec_decoder_out.y * cur_vec_decoder_out.y;
        block_sum += cur_vec_decoder_out.z * cur_vec_decoder_out.z;
        block_sum += cur_vec_decoder_out.w * cur_vec_decoder_out.w;
    }

    // Sum of squares
    blockReduceSum<T>(block_sum);
    __shared__ T inv_rms;

    if (tid == 0) {
        inv_rms = rsqrt(block_sum / hidden_units + eps);
    }
    __syncthreads();

    // RMSNorm
    if (vec_scale != nullptr) {
        for (int i = tid; i < vec_hidden_units; i += blockDim.x) {
            Vec_t &cur_vec_decoder_out = vec_decoder_out[i];
            Vec_t &cur_vec_scale = vec_scale[i];

            cur_vec_decoder_out.x = cur_vec_scale.x * cur_vec_decoder_out.x * inv_rms;
            cur_vec_decoder_out.y = cur_vec_scale.y * cur_vec_decoder_out.y * inv_rms;
            cur_vec_decoder_out.z = cur_vec_scale.z * cur_vec_decoder_out.z * inv_rms;
            cur_vec_decoder_out.w = cur_vec_scale.w * cur_vec_decoder_out.w * inv_rms;
        }
    }
}

template <>
__global__ void fusedAddBiasResidualAndRMSNorm(
    half *residual,    // [num tokens, hidden_units]
    half *decoder_out, // [num tokens, hidden_units]
    const half *bias,  // [hidden_units]
    const half *scale, // [hidden_units], RMSNorm weights
    const float eps,         // RMSNorm epsilon
    const int num_tokens,
    const int hidden_units
) {
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;

    Vec_t *vec_residual = (residual != nullptr && bias != nullptr) ? reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units) : nullptr;
    Vec_t *vec_bias = (bias != nullptr) ? reinterpret_cast<Vec_t *>(const_cast<half *>(bias)) : nullptr;
    Vec_t *vec_scale = (scale != nullptr) ? reinterpret_cast<Vec_t *>(const_cast<half *>(scale)) : nullptr;

    Vec_t vec_decoder_out, tmp;
    float block_sum = 0.0f;

    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        vec_decoder_out = reinterpret_cast<Vec_t *>(decoder_out)[batch_id * hidden_units / vec_size + i];
        tmp = __hadd2(__hadd2(vec_decoder_out, vec_residual[i]), vec_bias[i]);
        block_sum += __half2float(tmp.x) * __half2float(tmp.x) + __half2float(tmp.y) * __half2float(tmp.y);
    }

    // Mean of squares
    blockReduceSum<float>(block_sum);
    __shared__ Vec_t inv_rms;

    if (tid == 0) {
        inv_rms = ScalarCast2Vector::scalarCastToVector<Vec_t>(
            __float2half(rsqrt(block_sum / hidden_units + eps))
        );
    }
    __syncthreads();

    Vec_t *out = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units);

    #pragma unroll
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        out[i] = __hmul2(__hmul2(vec_scale[i], out[i]), inv_rms);
    }
}

template <typename T>
void launchFusedAddBiasResidualAndRMSNorm(
    TensorWrapper<T> *residual, // bias or residual
    TensorWrapper<T> *decoder_out, // [num tokens, hidden_units]
    BaseWeight<T> *norm,
    T *scale,                      // RMSNorm weights (gamma)
    float eps
) {
    const int num_tokens = decoder_out->shape[0];
    const int hidden_units = decoder_out->shape[1];
    T *bias = norm->bias;
    T *gamma = scale;
    const int vec_size = Vec<T>::size;
    const int num_threads = hidden_units / vec_size; // Assume head size can be divided by 4 and 2

    dim3 grid(num_tokens);
    dim3 block(num_threads);

    fusedAddBiasResidualAndRMSNorm<T><<<grid, block>>>(
        residual->data,
        decoder_out->data,
        bias,
        gamma,
        eps,
        num_tokens,
        hidden_units
    );

    #ifdef PRINT_DATA
        print_data<<<1, 1>>>(decoder_out->data);
    #endif
}

template void launchFusedAddBiasResidualAndRMSNorm<float>(
    TensorWrapper<float> *residual,
    TensorWrapper<float> *decoder_out, // [num tokens, hidden_units]
    BaseWeight<float> *norm,
    float *scale,                      // RMSNorm weights
    float eps
);

template void launchFusedAddBiasResidualAndRMSNorm<half>(
    TensorWrapper<half> *residual,
    TensorWrapper<half> *decoder_out, // [num tokens, hidden_units]
    BaseWeight<half> *norm,
    half *scale,                      // RMSNorm weights
    float eps
);
