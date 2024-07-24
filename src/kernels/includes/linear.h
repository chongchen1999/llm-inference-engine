#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <fstream>
#include "src/kernels/includes/cublas_utils.h"
#include "src/utils/tensor.h"
#include "src/weights/base_weights.h"
#include "src/utils/macro.h"

// TODO: When enabling int8/int4 weight only, add a new type parameter T2 to represent the weight type.
template <typename T>
void launchLinearGemm(
    TensorWrapper<T> *input, 
    BaseWeight<T> *weight, 
    TensorWrapper<T> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a = false, 
    bool trans_b = false
);

template <typename T>
void launchLinearStridedBatchGemm(
    TensorWrapper<T> *input1,
    TensorWrapper<T> *input2,
    TensorWrapper<T> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a = false, 
    bool trans_b = false
);
