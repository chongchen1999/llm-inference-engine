#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../../utils/tensor.h"

template <typename T>
void launchFusedTransposeAndRemovePadding(
    TensorWrapper<T> *padded_qkv_buf, 
    TensorWrapper<int> *padding_offset,
    TensorWrapper<T> *lineared_qkv
);
