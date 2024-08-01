#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../../models/llama/llama_params.h"
#include "../../utils/tensor.h"
#include "../../weights/includes/base_weights.h"
#include "../../utils/vectorize_utils.h"

// Applies bias to QKV tensors, performs transposition, and applies RoPE.
template<typename T>
void launchFusedQKVAddBiasAndTransposeAndRope(
    TensorWrapper<T> *q_buf,              // Output buffer for Q after RoPE and transposition
    TensorWrapper<T> *k_buf,              // Output buffer for K after RoPE and transposition
    TensorWrapper<T> *v_buf,              // Output buffer for V after transposition
    TensorWrapper<T> *QKV,                // Input buffer containing Q, K, V
    BaseWeight<T> *qkv,                   // Bias weights for QKV
    TensorWrapper<int> *padding_offset,   // Padding offsets for token sequences
    TensorWrapper<int> *history_length,   // History length for each batch
    TensorWrapper<int> *input_length,     // Actual input length of each sequence
    LlamaAttentionStaticParams *params    // Parameters for RoPE and other settings
);

// Applies RoPE (Rotary Positional Encoding) to the QKV buffer.
template<typename T>
void launchRope(
    TensorWrapper<T> *qkv_buf,            // Input buffer containing QKV
    TensorWrapper<int> *step,             // Current step in the sequence
    LlamaAttentionStaticParams *static_params // Parameters for RoPE and other settings
);
