#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../../utils/tensor.h"
#include "../../models/llama/llama_params.h"
#include "../../weights/base_weights.h"
#include "../../utils/vectorize_utils.h"

template <typename T>
void launchDecoderMaskedMultiHeadAttention(
    TensorWrapper<T> *qkv_buf,
    BaseWeight<T> *qkv,
    TensorWrapper<int> *layer_id,
    TensorWrapper<T> *k_cache,
    TensorWrapper<T> *v_cache,
    TensorWrapper<bool> *finished,
    TensorWrapper<int> *step,
    TensorWrapper<T> *mha_output,
    LlamaAttentionStaticParams *static_params
);
