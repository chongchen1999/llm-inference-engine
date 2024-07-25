#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "src/layers/ffn.h"

int main(int argc, char **argv) {
    const int head_num = 4;
    const int head_size = 8;
    const int inter_size = 12;
    const int hidden_units = head_num * head_size;

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;

    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    auto *cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    auto *allocator = new CudaAllocator;

    LlamaAttentionDynamicParams attn_dyn_params;
    attn_dyn_params.num_tokens = 14;

    std::cout << "Start malloc/cudamalloc buffer" << std::endl;

    auto *h_ffn_input = static_cast<float *>(
        malloc(sizeof(float) * hidden_units * attn_dyn_params.num_tokens)
    );
    float *d_ffn_input;
    cudaMalloc(
        reinterpret_cast<void **>(&d_ffn_input), 
        sizeof(float) * hidden_units * attn_dyn_params.num_tokens
    );

    for (int i = 0; i < hidden_units * attn_dyn_params.num_tokens; ++i) {
        h_ffn_input[i] = static_cast<float>(i % 2 + 1);
    }

    auto *h_gate_up = static_cast<float *>(malloc(sizeof(float) * hidden_units * 2 * inter_size));
    float *d_gate_up;
    cudaMalloc(
        reinterpret_cast<void **>(&d_gate_up), 
        sizeof(float) * hidden_units * 2 * inter_size
    );

    for (int i = 0; i < hidden_units * 2 * inter_size; ++i) {
        h_gate_up[i] = static_cast<float>(i % 2 + 1);
    }

    auto *h_down = static_cast<float *>(malloc(sizeof(float) * hidden_units * inter_size));
    float *d_down;
    cudaMalloc(reinterpret_cast<void **>(&d_down), sizeof(float) * hidden_units * inter_size);

    for (int i = 0; i < hidden_units * inter_size; ++i) {
        h_down[i] = static_cast<float>(i % 2 + 1);
    }

    float *d_ffn_output;
    cudaMalloc(
        reinterpret_cast<void **>(&d_ffn_output), 
        sizeof(float) * attn_dyn_params.num_tokens * hidden_units
    );

    std::cout << "End malloc/cudamalloc buffer and start memcpy h2d" << std::endl;

    CHECK(
        cudaMemcpy(
            d_ffn_input, h_ffn_input,
            sizeof(float) * hidden_units * attn_dyn_params.num_tokens,
            cudaMemcpyHostToDevice
        )
    );

    CHECK(
        cudaMemcpy(
            d_gate_up, h_gate_up,
            sizeof(float) * hidden_units * 2 * inter_size,
            cudaMemcpyHostToDevice
        )
    );

    CHECK(
        cudaMemcpy(
            d_down, h_down,
            sizeof(float) * hidden_units * inter_size,
            cudaMemcpyHostToDevice
        )
    );

    const DataType type = getTensorType<float>(); 

    LlamaAttentionWeights<float> ffn_weights;
    ffn_weights.gateAndup.data = d_gate_up;
    ffn_weights.gateAndup.shape = {2 * inter_size, hidden_units};

    // ffn_weights.up.data = d_up;
    // ffn_weights.up.shape = {hidden_units, inter_size};

    ffn_weights.down.data = d_down;
    ffn_weights.down.shape = {hidden_units, inter_size};

    auto *ffn_input = new TensorWrapper<float>(
        Device::GPU, type,
        {attn_dyn_params.num_tokens, hidden_units},
        d_ffn_input
    );

    auto *ffn_output = new TensorWrapper<float>(
        Device::GPU, type,
        {attn_dyn_params.num_tokens, hidden_units},
        d_ffn_output
    );

    TensorMap ffn_inputs{
        {"ffn_input", ffn_input}
    };

    TensorMap ffn_outputs{
        {"ffn_output", ffn_output}
    };

    std::cout << "Initializing FFN layer" << std::endl;

    auto *ffn_layer = new LlamaFFNLayer<float>(
        head_num, 
        head_size, 
        inter_size,
        stream,
        cublas_wrapper,
        allocator
    );

    std::cout << "Start fwd" << std::endl;

    ffn_layer->forward(ffn_inputs, ffn_outputs, ffn_weights, attn_dyn_params);

    std::cout << "End fwd" << std::endl;

    free(h_ffn_input);
    free(h_gate_up);
    free(h_down);

    cudaFree(d_ffn_input);
    cudaFree(d_gate_up);
    cudaFree(d_down);
    cudaFree(d_ffn_output);

    return 0;
}
