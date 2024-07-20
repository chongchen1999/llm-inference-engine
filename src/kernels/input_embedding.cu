#include <stdio.h>
#include "src/kernels/includes/input_embedding.h"

template<typename T>
__global__ void embeddingFunctor(const int *input_ids, T *output, 
                                 const T *embed_table,
                                 const int max_context_token_num,
                                 const int hidden_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    #pragma unroll
    for (int i = gid; i < max_context_token_num * hidden_size; i += stride) {
        int token_val = input_ids[i / hidden_size];
        int feature_vector_idx = i % hidden_size;
        output[i] = embed_table[token_val * hidden_size + feature_vector_idx];
    }
}

template<typename T>
void launchInputEmbedding(TensorWrapper<int> *input_ids,    // INT [token num]
                          TensorWrapper<T> *output,       // T [token num, hidden_size] = [token num, 4096]
                          EmbeddingWeight<T> *embed_table) { // T [vocab_size, hidden_size]
    const int block_size = 256;
    const int max_context_token_num = output->shape[0]; // token num
    const int hidden_size = output->shape[1];
    const int grid_size = 2048;
    
    LLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0], 
                        "Input IDs 1st shape should equal 1st shape of output");
    
    embeddingFunctor<T><<<grid_size, block_size>>>(input_ids->data,
                                                   output->data,
                                                   embed_table->data,
                                                   max_context_token_num,
                                                   hidden_size);
                                                 
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#endif
}

// Explicit template instantiation
template void launchInputEmbedding(TensorWrapper<int> *input_ids,    
                                   TensorWrapper<float> *output,       
                                   EmbeddingWeight<float> *embed_table);
                                   
template void launchInputEmbedding(TensorWrapper<int> *input_ids,    
                                   TensorWrapper<half> *output,
                                   EmbeddingWeight<half> *embed_table);
