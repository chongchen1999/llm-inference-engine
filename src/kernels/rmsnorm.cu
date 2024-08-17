#include <stdio.h>
#include "includes/rmsnorm.cuh"

template<typename T>
__device__ __forceinline__ void warpReduceSum(T &val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
}

template<typename T>
__device__ void blockReduceSum(T &val) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_num = (blockDim.x + 31) >> 5;
    __shared__ T warp_sum[32];

    warpReduceSum<T>(val);
    if (lane_id == 0) {
        warp_sum[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < warp_num ? warp_sum[lane_id] : static_cast<T>(0);
        warpReduceSum<T>(val);
    }
}

// This kernel is used at the beginning of every decoder layer and at the end of 32 decoder layers
// Allocate threads assuming head size can be divided by 4 and 2
// This kernel copies decoder_out to decoder_residual and then normalizes decoder_out; weights is gamma in RMSNorm
template <typename T>
__global__ void RMSNorm(
    T *decoder_out,          // [num tokens, q_hidden_units]
    T *decoder_residual,     // [num tokens, q_hidden_units]
    const T *weights,       // [hidden_units], aka gamma
    const float eps,
    const int num_tokens,
    const int hidden_units
) {
    const int tid = threadIdx.x;
    const int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;

    float block_sum = 0.0f;
    Vec_t *vec_decoder_out = reinterpret_cast<Vec_t *>(decoder_out + blockIdx.x * hidden_units);
    Vec_t *vec_residual = reinterpret_cast<Vec_t *>(decoder_residual + blockIdx.x * hidden_units);

    #pragma unroll
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t vec_temp = vec_decoder_out[idx];
        vec_residual[idx] = vec_temp;
        Vec_t vec_sqr = VectorizedOperator<Vec_t>::mul(vec_temp, vec_temp);
        block_sum += vec_sqr.x;
        block_sum += vec_sqr.y;
        block_sum += vec_sqr.z;
        block_sum += vec_sqr.w;
    }

    blockReduceSum<T>(block_sum);
    __shared__ T inv_mean;

    if (tid == 0) {
        inv_mean = rsqrtf(block_sum / hidden_units + eps);
    }
    __syncthreads();

    const Vec_t *vec_weights = reinterpret_cast<const Vec_t *>(weights);

    #pragma unroll
    for (int idx = tid; idx < hidden_units / vec_size; idx += blockDim.x) {
        vec_decoder_out[idx] = VectorizedOperator<Vec_t>::mul(
            VectorizedOperator<Vec_t>::mul(vec_decoder_out[idx], vec_weights[idx]), 
            ScalarCast2Vector::scalarCastToVector<Vec_t, float>(inv_mean)
        );
    }
}

template<>
__global__ void RMSNorm(
    half *decoder_out,        // [num tokens, q_hidden_units]
    half *decoder_residual,   // [num tokens, q_hidden_units]
    const half *weights,     // [hidden_units]
    const float eps,
    const int num_tokens,
    const int hidden_units
) {
    const int tid = threadIdx.x;
    const int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;

    Vec_t *vec_decoder_out = reinterpret_cast<Vec_t *>(decoder_out + blockDim.x * hidden_units);
    Vec_t *vec_residual = (decoder_residual != nullptr) ? 
        reinterpret_cast<Vec_t *>(decoder_residual + blockDim.x * hidden_units) : nullptr;
    
    float block_sum = 0.0f;

    #pragma unroll
    for (int i = threadIdx.x; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t temp = vec_decoder_out[i]; // Note: offset should divide vec_size
        if (decoder_residual != nullptr) {
            vec_residual[i] = temp;
        }
        block_sum += __half2float(temp.x) * __half2float(temp.x);
        block_sum += __half2float(temp.y) * __half2float(temp.y);
    }

    // Mean(x^2)
    blockReduceSum<float>(block_sum);
    __shared__ float inv_mean;

    if (tid == 0) {
        inv_mean = rsqrtf(block_sum / hidden_units + eps);
    }
    __syncthreads();

    // RMSNorm
    const Vec_t *vec_weights = reinterpret_cast<const Vec_t *>(weights);

    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t temp = vec_decoder_out[i];
        vec_decoder_out[i].x = vec_weights[i].x * __float2half(__half2float(temp.x) * inv_mean);
        vec_decoder_out[i].y = vec_weights[i].y * __float2half(__half2float(temp.y) * inv_mean);
    }
}

template<typename T>
void launchRMSNorm(
    TensorWrapper<T> *decoder_out,           // [num tokens, hidden_units]
    TensorWrapper<T> *decoder_residual,      // [num tokens, hidden_units]
    LayerNormWeight<T> *attention_norm_weight,     // [hidden_units]
    float eps,
    bool isLast
) {
    const int num_tokens = decoder_out->shape[0];
    const int hidden_units = decoder_out->shape[1];
    const int vec_size = Vec<T>::size;
    const int num_threads = std::min<int>(hidden_units / vec_size, 1024);

    dim3 grid(num_tokens);
    dim3 block(num_threads);

    RMSNorm<T><<<grid, block>>>(
        decoder_out->data,
        decoder_residual->data,
        attention_norm_weight->gamma,
        eps,
        num_tokens,
        hidden_units
    );

    #ifdef PRINT_DATA
        print_data<<<1, 1>>>(decoder_out->data);
    #else
    #endif
}

template void launchRMSNorm(
    TensorWrapper<float> *decoder_out,        // [num tokens, hidden_units]
    TensorWrapper<float> *decoder_residual,
    LayerNormWeight<float> *attention_norm_weight,
    float eps, bool is_last
);

template void launchRMSNorm(
    TensorWrapper<half> *decoder_out,         // [num tokens, hidden_units]
    TensorWrapper<half> *decoder_residual,
    LayerNormWeight<half> *attention_norm_weight,
    float eps, bool is_last
);
