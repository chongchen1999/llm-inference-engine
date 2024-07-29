#include <stdio.h>
#include "includes/input_embedding.h"

template<typename T>
__global__ void embeddingFunctor(
    const int *input_ids, 
    T *output, 
    const T *embed_table,
    const int max_context_token_num,
    const int hidden_size
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    #pragma unroll
    for (int i = gid; i < max_context_token_num * hidden_size; i += stride) {
        const token_id = i / hidden_size;
        const int token_val = input_ids[token_id];
        const int feature_vector_idx = i - token_id * hidden_size; // aka i % hidden_size
        output[i] = embed_table[token_val * hidden_size + feature_vector_idx];
    }
}

template<typename T>
void launchInputEmbedding(
    TensorWrapper<int> *input_ids,    
    TensorWrapper<T> *output,       
    EmbeddingWeight<T> *embed_table) {
    const int block_size = 256;
    const int max_context_token_num = output->shape[0];
    const int hidden_size = output->shape[1];
    const int grid_size = 2048;
    
    LLM_CHECK_WITH_INFO(
        max_context_token_num == input_ids->shape[0], 
        "Input IDs 1st shape should equal 1st shape of output"
    );
    
    embeddingFunctor<T><<<grid_size, block_size>>>(
        input_ids->data,
        output->data,
        embed_table->data,
        max_context_token_num,
        hidden_size
    );
                                                 
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#endif
}

// Explicit template instantiation
template void launchInputEmbedding(
    TensorWrapper<int> *input_ids,    
    TensorWrapper<float> *output,       
    EmbeddingWeight<float> *embed_table
);

template void launchInputEmbedding(
    TensorWrapper<int> *input_ids,    
    TensorWrapper<half> *output,
    EmbeddingWeight<half> *embed_table
);
