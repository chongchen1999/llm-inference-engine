#include <algorithm>    // std::fill_n
#include <iostream>     // snprintf
#include <math.h>       // expf, log
#include <stdlib.h>     // rand
#include <string>       // std::string
#include <vector>       // std::vector

#include <cuda.h>
#include "../../src/kernels/includes/topk.h"
#include "../../src/utils/tensor.h"

// Note:
// There is no top k CPU kernel implementation now.
// We compare the kernel correctness by visual inspection and result print infos.
// Use `./test_topk` to test fp32 GPU kernel.

int main() {
    const int batch_size = 1;
    const int vocab_size = 32000;
    const int beam_num = 2;
    const int K = 5;
    const int blocks_per_beam = 8;

    // Debug info, better to retain:
    // std::cout << "batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    
    const int probs_size = batch_size * vocab_size * beam_num;

    float *h_probs = static_cast<float *>(malloc(sizeof(float) * probs_size));
    float *d_probs;
    cudaMalloc(reinterpret_cast<void **>(&d_probs), sizeof(float) * probs_size);

    const int topK_val_buf_size = batch_size * beam_num * blocks_per_beam * K;
    const int topK_ids_buf_size = batch_size * beam_num * blocks_per_beam * K;
    const int final_topK_val_buf_size = batch_size * beam_num * K; // Sampling topK buf size

    int *d_tmp_topk_ids;
    cudaMalloc(reinterpret_cast<void **>(&d_tmp_topk_ids), sizeof(int) * topK_ids_buf_size);

    float *d_tmp_topk_vals;
    cudaMalloc(reinterpret_cast<void **>(&d_tmp_topk_vals), sizeof(float) * topK_val_buf_size);

    int *h_final_topk_ids = static_cast<int *>(malloc(sizeof(int) * final_topK_val_buf_size));
    int *d_final_topk_ids;
    cudaMalloc(reinterpret_cast<void **>(&d_final_topk_ids), sizeof(int) * final_topK_val_buf_size);

    float *h_final_topk_vals = static_cast<float *>(malloc(sizeof(float) * final_topK_val_buf_size));
    float *d_final_topk_vals;
    cudaMalloc(reinterpret_cast<void **>(&d_final_topk_vals), sizeof(float) * final_topK_val_buf_size);

    for (int i = 0; i < probs_size; ++i) {
        h_probs[i] = static_cast<float>(i);
    }
    
    cudaMemcpy(d_probs, h_probs, sizeof(float) * probs_size, cudaMemcpyHostToDevice);

    const DataType type_float = getTensorType<float>();
    const DataType type_int = getTensorType<int>();

    TensorWrapper<float> *probs_tensor = new TensorWrapper<float>(
        Device::GPU, type_float, {batch_size * beam_num, vocab_size}, d_probs
    );
    TensorWrapper<int> *tmp_topk_ids = new TensorWrapper<int>(
        Device::GPU, type_int, {batch_size, beam_num, blocks_per_beam, K}, d_tmp_topk_ids
    );
    TensorWrapper<float> *tmp_topk_vals = new TensorWrapper<float>(
        Device::GPU, type_float, {batch_size, beam_num, blocks_per_beam, K}, d_tmp_topk_vals
    );
    TensorWrapper<int> *final_topk_ids = new TensorWrapper<int>(
        Device::GPU, type_int, {batch_size * beam_num, K}, d_final_topk_ids
    );
    TensorWrapper<float> *final_topk_vals = new TensorWrapper<float>(
        Device::GPU, type_float, {batch_size * beam_num, K}, d_final_topk_vals
    );

    // Debug info, better to retain:
    // std::cout << "before launch kernel" << std::endl;
    launchTopKForBeamSearch(
        probs_tensor, tmp_topk_ids, tmp_topk_vals, final_topk_ids, final_topk_vals
    );

    // Note: Remember to memcpy from device to host and define the correct copy size (multiply by sizeof(dtype)), or it will cause a segmentation fault
    cudaMemcpy(h_final_topk_ids, d_final_topk_ids, sizeof(int) * final_topK_val_buf_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_topk_vals, d_final_topk_vals, sizeof(float) * final_topK_val_buf_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < final_topK_val_buf_size; ++i) {
        const int id = h_final_topk_ids[i];
        printf("topK id = %d\n", id);
        const float val = h_final_topk_vals[i];
        printf("topK val = %f\n", val);
    }

    // Debug info, better to retain:
    // std::cout << "before free" << std::endl;
    
    free(h_probs);
    free(h_final_topk_ids);
    free(h_final_topk_vals);
    cudaFree(d_probs);
    cudaFree(d_final_topk_ids);
    cudaFree(d_final_topk_vals);
    cudaFree(d_tmp_topk_ids);
    cudaFree(d_tmp_topk_vals);

    return 0;
}
