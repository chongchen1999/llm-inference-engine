#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../../weights/includes/base_weights.h"
#include "../../weights/includes/norm_weights.h"
#include "../../utils/tensor.h"
#include "../../utils/vectorize_utils.h"

template <typename T>
void launchFusedAddBiasResidualAndRMSNorm(
    TensorWrapper<T> *residual,
    TensorWrapper<T> *decoder_out,  // [num tokens, hidden_units]
    BaseWeight<T> *norm,
    T *scale,                      // RMSNorm weights
    float eps
);
