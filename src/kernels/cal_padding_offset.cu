#include "src/kernels/includes/cal_padding_offset.h"
// shape:
    //seq_lengths:[batch size]
    //cum_seqlens:[batch size + 1],first ele is 0
    //padding_offset:[batch size * max q len]
// note: the point is to calc padding offset and cum offset
// TODO: we first use serial algo, then can enhance to CUDA scan algo

__global__ void calPaddingOffset(int * const padding_offset, int * const cum_seqlens, 
                                 const int * const input_lengths,
                                 const int batch_size, const int max_q_len) {
    // printf("Hello!!!!");
    // printf("%d %d\n", batch_size, max_q_len);
    int tot_seqlen = 0;
    int tot_padding = 0;
    int idx = 0;
    for (int i = 0; i < batch_size; ++i) {
        // printf("%d %d\n", i, input_lengths[i]);
        int cur_seqlen = input_lengths[i];
        int cur_padding = max_q_len - cur_seqlen;

        cum_seqlens[i] = tot_seqlen;
        for (int j = 0; j < cur_seqlen; ++j) {
            padding_offset[idx++] = tot_padding;
        }
        tot_padding += cur_padding;
        tot_seqlen += cur_seqlen;
    }
    cum_seqlens[batch_size] = tot_seqlen;
}

void launchCalPaddingOffset(TensorWrapper<int> *padding_offset, 
                            TensorWrapper<int> *cum_seqlens,
                            TensorWrapper<int> *input_lengths) {
    const int batch_size = padding_offset->shape[0];                            
    const int max_q_len = padding_offset->shape[1]; 
    LLM_CHECK_WITH_INFO(batch_size == input_lengths->shape[0], 
                        "input lenghts numbers should equal to padding offset batch size dim!");                        
    LLM_CHECK_WITH_INFO(batch_size == cum_seqlens->shape[0] - 1, 
                        "cumulated seqlen should equal to padding offset batch size dim + 1!");
    // printf("ready!\n");
    // printf("%d %d\n", batch_size, max_q_len);
    calPaddingOffset<<<1, 1>>>(padding_offset->data, cum_seqlens->data, 
                               input_lengths->data, 
                               batch_size, max_q_len);
    // cudaDeviceSynchronize();
    // printf("done!\n");
}