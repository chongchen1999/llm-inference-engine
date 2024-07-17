#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/models/llama/llama_params.h"
#include "src/utils/tensor.h"
#include "src/weights/base_weights.h"
#include "src/utils/vectorize_utils.h"

template<typename T>
void launchAddFusedQKVBiasTransposeAndRope(TensorWrapper<T> *q_buf,
                                           TensorWrapper<T> *k_buf,
                                           TensorWrapper<T> *v_buf,
                                           TensorWrapper<T> *QKV,
                                           BaseWeight<T> &qkv, // output
                                           //Tensor *qkv_bias,
                                           TensorWrapper<int> *padding_offset,
                                           TensorWrapper<int> *history_length,
                                           TensorWrapper<int> *input_length,
                                           LlamaAttentionStaticParams &params);

template<typename T>
void launchRope(TensorWrapper<T> *qkv_buf,
                TensorWrapper<int> *step,
                LlamaAttentionStaticParams &static_params);
