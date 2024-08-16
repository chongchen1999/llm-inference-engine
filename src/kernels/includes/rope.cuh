#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../../models/llama/llama_params.h"
#include "../../utils/tensor.h"
#include "../../weights/includes/base_weights.h"
#include "../../utils/vectorize_utils.h"

// Applies RoPE (Rotary Positional Encoding) to the QKV buffer.
template<typename T>
void launchRope(
    TensorWrapper<T> *qkv_buf,            // Input buffer containing QKV
    TensorWrapper<int> *step,             // Current step in the sequence
    LlamaAttentionStaticParams *static_params // Parameters for RoPE and other settings
);