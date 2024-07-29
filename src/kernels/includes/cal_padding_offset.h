#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../../utils/macro.h"
#include "../../utils/tensor.h"

/*
    for example:
    input_lengths: [4, 3, 5]
    cum_seqlens: [0, 4, 7, 12]
    padding_offset: [0, 0, 0, 0,
                     1, 1, 1,
                     3, 3, 3, 3, 3]
*/
void launchCalPaddingOffset(
    TensorWrapper<int> *padding_offset, 
    TensorWrapper<int> *cum_seqlens,
    TensorWrapper<int> *input_lengths
);