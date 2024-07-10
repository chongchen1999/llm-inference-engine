#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include "src/kernels/cal_padding_offset.h"

// this kernel is only int type input and output, not fp32 or half
// we compare the kernel correctnesss by eyes and result print infos
// `./paddingoffset` to run
int main() {
    const int batch_size = 4;
    const int max_q_len = 5;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    int *h_seq_lens, *d_seq_lens;
    h_seq_lens = (int *)malloc(sizeof(int) * batch_size);
    cudaMalloc((void **) &d_seq_lens, sizeof(int) * batch_size);

    int *h_cum_seqlens, *d_cum_seqlens;
    h_cum_seqlens = (int *)malloc(sizeof(int) * (batch_size + 1));
    cudaMalloc((void **) &d_cum_seqlens, sizeof(int) * (batch_size + 1));
    
    int *h_padding_offset, *d_padding_offset;
    h_padding_offset = (int *)malloc(sizeof(int) * batch_size * max_q_len);
    cudaMalloc((void **) &d_padding_offset, sizeof(int) * batch_size * max_q_len);

    for(int i = 0; i < batch_size; i++) {
       h_seq_lens[i] = batch_size - (i * i % 3);
    }
    cudaMemcpy(d_seq_lens, h_seq_lens, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    DataType type_int = getTensorType<int>();
    auto padding_offset = new TensorWrapper<int> {Device::GPU, type_int, {batch_size, max_q_len}, d_padding_offset};
    auto cum_seqlens = new TensorWrapper<int> {Device::GPU, type_int, {batch_size + 1}, d_cum_seqlens};
    auto input_lengths = new TensorWrapper<int> {Device::GPU, type_int, {batch_size}, d_seq_lens};
    
    // printf("before launch cuda kernel!\n");
    launchCalPaddingOffset(padding_offset, cum_seqlens, input_lengths);
    // cudaDeviceSynchronize();
    // printf("after launch cuda kernel!\n");
    
    cudaMemcpy(h_padding_offset, d_padding_offset, sizeof(int) * batch_size * max_q_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cum_seqlens, d_cum_seqlens, sizeof(int) * (batch_size + 1), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_size; ++i) {
        printf("%d ", h_seq_lens[i]);
    }
    printf("\n");

    printf("padding offset:\n");
    for (int i = 0; i < batch_size * max_q_len; ++i) {
        printf("%d ", h_padding_offset[i]);
    }
    printf("\n");

    printf("cum_seqlens:\n");
    for (int i = 0; i < batch_size + 1; ++i) {
        printf("%d ", h_cum_seqlens[i]);
    }
    printf("\n");
    //expected result is:
    // padding_offset: 0,0,0,2,2,2,4,4,4,0.... shape = [batchsize, max_q_len]
    // cum_seqlens: 0,3,6,9. shape=[batchsize+1]
    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_seq_lens);
    free(h_padding_offset);
    free(h_cum_seqlens);
    cudaFree(d_seq_lens);
    cudaFree(d_padding_offset);
    cudaFree(d_cum_seqlens);
    return 0;
}