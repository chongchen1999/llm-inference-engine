#include <iostream>
#include "src/kernels/includes/silu_and_mul.h"
// #include "src/utils/cuda_debug_utils.cuh"
#include "src/utils/macro.h"

template<typename T>
__device__ __forceinline__ T silu(const T &in) {
    // x * sigmoid(x)
    return static_cast<T>(static_cast<float>(in) / (1.0f + expf(static_cast<float>(-in))));
}

template<>
__device__ __forceinline__ half2 silu<half2>(const half2 &in) {
    return make_half2(__float2half(silu<float>(static_cast<float>(in.x))), __float2half(silu<float>(static_cast<float>(in.y))));
}

// Code logic: The first intermediate applies SiLU, and the result is multiplied by the second intermediate
template<typename T>
__global__ void siluAndMul(T *out, const T *input, const int intermedia_size) {
    const int batch_idx = blockIdx.x;

    #pragma unroll
    for (int idx = threadIdx.x; idx < intermedia_size; idx += blockDim.x) {
        const T x = input[batch_idx * 2 * intermedia_size + idx];
        const T y = input[batch_idx * 2 * intermedia_size + intermedia_size + idx];
        out[batch_idx * intermedia_size + idx] = silu<T>(x) * y;
    }
}

template<>
__global__ void siluAndMul<half>(half *out, const half *input, const int intermedia_size) {
    const int batch_idx = blockIdx.x;
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;

    #pragma unroll
    for (int idx = threadIdx.x * vec_size; idx < intermedia_size; idx += blockDim.x) {
        const Vec_t x = *reinterpret_cast<const Vec_t *>(&input[batch_idx * 2 * intermedia_size + idx]);
        const Vec_t y = *reinterpret_cast<const Vec_t *>(&input[batch_idx * 2 * intermedia_size + intermedia_size + idx]);
        *reinterpret_cast<Vec_t *>(&out[batch_idx * intermedia_size + idx]) = __hmul2(silu<Vec_t>(x), y);
    }
}

template<typename T>
void launchSiluAndMul(TensorWrapper<T> *input, TensorWrapper<T> *out) {
    int batch_size = input->shape[0];
    LLM_CHECK(input->shape[1] == 2);
    int intermedia_size = input->shape[2];
    dim3 grid(batch_size);
    dim3 block(256);
    siluAndMul<T><<<grid, block>>>(out->data, input->data, intermedia_size);
    
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(out->data);
#endif
}

// We must instantiate the template, otherwise, it will report a linking issue
template void launchSiluAndMul(TensorWrapper<float> *input, TensorWrapper<float> *output);
template void launchSiluAndMul(TensorWrapper<half> *input, TensorWrapper<half> *output);
