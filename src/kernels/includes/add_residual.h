#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"

template <typename T>
void launchAddResidual(
    TensorWrapper<T> *residual,       // [num tokens, hidden_units]
    TensorWrapper<T> *decoder_out,   // [num tokens, hidden_units]
    bool is_print = false            // Optional flag for printing
);
