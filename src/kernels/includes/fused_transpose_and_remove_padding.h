#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"

template <typename T>
void launchFusedTransposeAndRemovePadding(
    TensorWrapper<T> *qkv_buf_with_padding, 
    TensorWrapper<int> *padding_offset,
    TensorWrapper<T> *qkv_buf_without_padding
);
