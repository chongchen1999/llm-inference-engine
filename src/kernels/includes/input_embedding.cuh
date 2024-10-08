#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../../utils/tensor.h"
#include "../../weights/includes/embedding_weights.h"

template<typename T>
void launchInputEmbedding(
    TensorWrapper<int> *input_ids, // [num_tokens]
    TensorWrapper<T> *output, // [num_tokens, embed_dim]
    EmbeddingWeight<T> *embed_table
);