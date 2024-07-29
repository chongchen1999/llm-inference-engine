#include <stdio.h>
#include "includes/add_residual.h"
// #include "src/utils/cuda_debug_utils.cuh"

// Note: This kernel is used at the end of FFN in every decoder layer

template <typename T>
__global__ void addResidual(
    T *residual,
    T *decoder_out,
    int num_tokens,
    int hidden_units
) {
    const int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;

    const int batch_id = blockIdx.x;
    const int tid = threadIdx.x;

    Vec_t *vec_decoder_out = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units);
    Vec_t *vec_residual = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units);

    #pragma unroll
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        VectorizedOperator<Vec_t>::add_assign(vec_decoder_out[i], vec_residual[i]);
    }
}

template <>
__global__ void addResidual(
    half *residual,
    half *decoder_out,
    int num_tokens,
    int hidden_units
) {
    const int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;

    const int batch_id = blockIdx.x;
    const int tid = threadIdx.x;

    Vec_t *vec_decoder_out = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units);
    Vec_t *vec_residual = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units);

    #pragma unroll
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        vec_decoder_out[i] = __hadd2(vec_decoder_out[i], vec_residual[i]);
    }
}

template <typename T>
void launchAddResidual(
    TensorWrapper<T> *residual,
    TensorWrapper<T> *decoder_out,
    bool is_print
) {
    const int batch_size = decoder_out->shape[0];
    const int hidden_units = decoder_out->shape[1];
    const int vec_size = Vec<T>::size;

    const dim3 grid(batch_size);
    const dim3 block(256);

    addResidual<T><<<grid, block>>>(
        residual->data,
        decoder_out->data,
        batch_size,
        hidden_units
    );

#ifdef PRINT_DATA
    if (is_print)
    {
        print_data<<<1, 1>>>(decoder_out->data);
    }
#endif
}

template void launchAddResidual(
    TensorWrapper<float> *residual,
    TensorWrapper<float> *decoder_out,
    bool is_print
);

template void launchAddResidual(
    TensorWrapper<half> *residual,
    TensorWrapper<half> *decoder_out,
    bool is_print
);
