#include <stdio.h>
#include "src/kernels/includes/rmsnorm.h"

// Bugs:
// 1. Second warpReduceSum returns 0 due to blockDim.x < 32; blockDim.x / 32 = 0
// 2. Output buffer values are the same as before the call because we didn't successfully write into the output address
// 3. The first 32 values of the output buffer are correct, but the latter values are wrong because when using vec, the element count of a row is hidden_units / vec_size; we need to handle row stride carefully
// 4. Remember to add __syncthreads() in fp32/fp16 kernels; otherwise, we get incorrect results. Here, missing __syncthreads() leads to some results being 0

template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

// Note: When blockSize < 32, using blockDim.x / 32 to get the number of warps is incorrect; instead, use ceil.
template<typename T>
__device__ T blockReduceSum(T val) {
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int laneId = tid & 31;
    const int warpNum = (blockDim.x + 31) >> 5;

    static __shared__ T warpSum[32];
    val = warpReduceSum<T>(val);

    if (laneId == 0) {
        warpSum[warpId] = val;
    }
    __syncthreads();

    T sum = tid < warpNum ? warpSum[tid] : 0;
    sum = warpReduceSum<T>(sum); // Though 0th owns the sum, no need to shuffle sync
    return sum;
}

// This kernel is used at the beginning of every decoder layer and at the end of 32 decoder layers
// Allocate threads assuming head size can be divided by 4 and 2
// This kernel copies decoder_out to decoder_residual and then normalizes decoder_out; weights is gamma in RMSNorm
template <typename T>
__global__ void RMSNorm(
    T *decoderOut,          // [num tokens, q_hidden_units]
    T *decoderResidual,     // [num tokens, q_hidden_units]
    const T *weights,       // [hidden_units], aka gamma
    float eps,
    int numTokens,
    int hidden_units
) {
    const int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;

    float thread_sum = 0.0f;
    auto vecDout = reinterpret_cast<Vec_t *>(decoderOut + blockIdx.x * hidden_units);
    auto vecRsd = reinterpret_cast<Vec_t *>(decoderResidual + blockIdx.x * hidden_units);

    #pragma unroll
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t vecTemp = vecDout[idx];
        vecRsd[idx] = vecTemp;
        VectorizedOperator::add_assign(thread_sum, VectorizedOperator::mul(vecTemp, vecTemp));
    }

    thread_sum = blockReduceSum<float>(thread_sum);
    __shared__ float invMean;

    if (threadIdx.x == 0) {
        invMean = rsqrtf(thread_sum / hidden_units + eps);
    }
    __syncthreads();

    auto vecWeights = reinterpret_cast<Vec_t *>(const_cast<T *>(weights));

    #pragma unroll
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        vecDout[idx] = VectorizedOperator::mul(
            VectorizedOperator::mul(vecDout[idx], vecWeights[idx]), 
            ScalarCast2Vector::scalar_cast2_vector(invMean)
        );
    }
}

template<>
__global__ void RMSNorm(
    half *decoderOut,        // [num tokens, q_hidden_units]
    half *decoderResidual,   // [num tokens, q_hidden_units]
    const half *weights,     // [hidden_units]
    float eps,
    int numTokens,
    int hidden_units
) {
    const int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;

    auto vecDout = reinterpret_cast<Vec_t *>(decoderOut + blockDim.x * hidden_units);
    Vec_t *rsd = (decoderResidual != nullptr) ? reinterpret_cast<Vec_t *>(decoderResidual + blockDim.x * hidden_units) : nullptr;
    
    float thread_sum = 0.0f;

    #pragma unroll
    for (int i = threadIdx.x; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t out = vecDout[i]; // Note: offset should divide vec_size
        if (decoderResidual != nullptr) {
            rsd[i] = out;
        }
        thread_sum += __half2float(out.x) * __half2float(out.x);
        thread_sum += __half2float(out.y) * __half2float(out.y);
    }

    // Mean(x^2)
    float blockSum = blockReduceSum<float>(thread_sum);
    __shared__ float invFenmu;

    if (threadIdx.x == 0) {
        invFenmu = rsqrtf(blockSum / hidden_units + eps);
    }
    __syncthreads();

    // RMSNorm
    auto vecWeights = reinterpret_cast<Vec_t *>(const_cast<half *>(weights));

    for (int i = threadIdx.x; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t doutHalf2 = vecDout[i];
        vecDout[i].x = vecWeights[i].x * __float2half(__half2float(doutHalf2.x) * invFenmu);
        vecDout[i].y = vecWeights[i].y * __float2half(__half2float(doutHalf2.y) * invFenmu);
    }
}

template<typename T>
void launchRMSNorm(
    TensorWrapper<T> *decoderOut,           // [num tokens, hidden_units]
    TensorWrapper<T> *decoderResidual,      // [num tokens, hidden_units]
    LayerNormWeight<T> *attnNormWeight,     // [hidden_units]
    float eps,
    bool isLast
) {
    const int numTokens = decoderOut->shape[0];
    const int hidden_units = decoderOut->shape[1];
    const int vec_size = Vec<T>::size;
    const int numThreads = std::min<int>(hidden_units / vec_size, 1024);

    dim3 grid(numTokens);
    dim3 block(numThreads);

    RMSNorm<T><<<grid, block>>>(
        decoderOut->data,
        decoderResidual->data,
        attnNormWeight->gamma,
        eps,
        numTokens,
        hidden_units
    );

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(decoderOut->data);
#else
#endif
}

template void launchRMSNorm(
    TensorWrapper<float> *decoderOut,        // [num tokens, hidden_units]
    TensorWrapper<float> *decoderResidual,
    LayerNormWeight<float> *attnNormWeight,
    float eps,
    bool isLast
);

template void launchRMSNorm(
    TensorWrapper<half> *decoderOut,         // [num tokens, hidden_units]
    TensorWrapper<half> *decoderResidual,
    LayerNormWeight<half> *attnNormWeight,
    float eps,
    bool isLast
);
