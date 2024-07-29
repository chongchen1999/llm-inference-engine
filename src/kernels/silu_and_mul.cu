#include <iostream>
#include "includes/silu_and_mul.h"
#include "../utils/cuda_debug_utils.cuh"
#include "../utils/macro.h"

template <typename T>
__device__ __forceinline__ T silu(const T &x) {
    // x * sigmoid(x)
    return static_cast<T>(static_cast<float>(x) / (1.0f + expf(static_cast<float>(-x))));
}

template <>
__device__ __forceinline__ half2 silu<half2>(const half2 &x) {
    return make_half2(
        __float2half(silu<float>(static_cast<float>(x.x))),
        __float2half(silu<float>(static_cast<float>(x.y)))
    );
}

/*
    Code logic: The first intermediate applies SiLU, and the result is multiplied by the second intermediate
    dim3 grid(batch_size);
    dim3 block(256);
*/
template <typename T>
__global__ void siluAndMul(
    T *out,
    const T *input, // [bs, 2, intermedia_size]
    const int intermedia_size
) {
    const int batch_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int base_offset = batch_id * 2 * intermedia_size;

    #pragma unroll
    for (int idx = tid; idx < intermedia_size; idx += blockDim.x) {
        const T gate = input[base_offset + idx];
        const T up = input[base_offset + intermedia_size + idx];
        out[batch_id * intermedia_size + idx] = silu<T>(gate) * up;
    }
}

template <>
__global__ void siluAndMul<half>(
    half *out,
    const half *input,
    const int intermedia_size
) {
    const int batch_id = blockIdx.x;
    const int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;

    #pragma unroll
    for (int idx = threadIdx.x * vec_size; idx < intermedia_size; idx += blockDim.x) {
        const Vec_t gate = *reinterpret_cast<const Vec_t *>(&input[batch_id * 2 * intermedia_size + idx]);
        const Vec_t up = *reinterpret_cast<const Vec_t *>(&input[batch_id * 2 * intermedia_size + intermedia_size + idx]);
        *reinterpret_cast<Vec_t *>(&out[batch_id * intermedia_size + idx]) = __hmul2(silu<Vec_t>(gate), up);
    }
}

template <typename T>
void launchSiluAndMul(
    TensorWrapper<T> *input, // [bs, 2, intermedia_size]
    TensorWrapper<T> *out
) {
    const int batch_size = input->shape[0];
    LLM_CHECK_WITH_INFO(input->shape[1] == 2, "The second dimension of the input tensor must be 2");
    const int intermedia_size = input->shape[2];
    
    dim3 grid(batch_size);
    dim3 block(256);
    
    siluAndMul<T><<<grid, block>>>(
        out->data, 
        input->data, 
        intermedia_size
    );
    
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(out->data);
#endif
}

// Instantiate the template to avoid linking issues
template void launchSiluAndMul(
    TensorWrapper<float> *input, 
    TensorWrapper<float> *output
);

template void launchSiluAndMul(
    TensorWrapper<half> *input, 
    TensorWrapper<half> *output
);
