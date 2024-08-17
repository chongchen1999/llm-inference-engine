#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../utils/tensor.h"
#include "../../utils/vectorize_utils.h"

template <typename T>
void launchFusedScaleMaskAndSoftmax(
    TensorWrapper<T> *qk,
    TensorWrapper<T> *mask,
    TensorWrapper<T> *attention_weights,
    float scale
);
