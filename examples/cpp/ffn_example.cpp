#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../src/layers/includes/ffn.h"
#include "../../src/memory/memory_deleter.cuh"

int main(int argc, char **argv) {
    constexpr int head_num = 4;
    constexpr int head_size = 8;
    constexpr int intermediate_size = 12;
    constexpr int hidden_units = head_num * head_size;

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;

    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    auto cublas_wrapper = new CublasWrapper(cublas_handle, cublaslt_handle);
    auto allocator = new CudaAllocator();

    LlamaAttentionDynamicParams attention_dynamic_params;
    attention_dynamic_params.num_tokens = 14;

    std::cout << "Start malloc/cudamalloc buffer" << std::endl;

    float *h_ffn_input = new float[hidden_units * attention_dynamic_params.num_tokens];
    float *d_ffn_input;
    cudaMalloc(reinterpret_cast<void **>(&d_ffn_input), sizeof(float) * hidden_units * attention_dynamic_params.num_tokens);

    for (int i = 0; i < hidden_units * attention_dynamic_params.num_tokens; ++i) {
        h_ffn_input[i] = static_cast<float>(i % 2 + 1);
    }

    float *h_gate_up = new float[hidden_units * 2 * intermediate_size];
    float *d_gate_up;
    cudaMalloc(reinterpret_cast<void **>(&d_gate_up), sizeof(float) * hidden_units * 2 * intermediate_size);

    for (int i = 0; i < hidden_units * 2 * intermediate_size; ++i) {
        h_gate_up[i] = static_cast<float>(i % 2 + 1);
    }

    float *h_down = new float[hidden_units * intermediate_size];
    float *d_down;
    cudaMalloc(reinterpret_cast<void **>(&d_down), sizeof(float) * hidden_units * intermediate_size);

    for (int i = 0; i < hidden_units * intermediate_size; ++i) {
        h_down[i] = static_cast<float>(i % 2 + 1);
    }

    float *d_ffn_output;
    cudaMalloc(reinterpret_cast<void **>(&d_ffn_output), sizeof(float) * attention_dynamic_params.num_tokens * hidden_units);

    std::cout << "End malloc/cudamalloc buffer and start memcpy h2d" << std::endl;

    CHECK(cudaMemcpy(d_ffn_input, h_ffn_input, sizeof(float) * hidden_units * attention_dynamic_params.num_tokens, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_gate_up, h_gate_up, sizeof(float) * hidden_units * 2 * intermediate_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_down, h_down, sizeof(float) * hidden_units * intermediate_size, cudaMemcpyHostToDevice));

    const DataType type = getTensorType<float>();

    LlamaFFNWeights<float> ffn_weights;
    ffn_weights.gate_and_up.data = d_gate_up;
    ffn_weights.gate_and_up.shape = std::vector<int>{2 * intermediate_size, hidden_units};
    ffn_weights.gate_and_up.is_transposed = true;

    ffn_weights.down.data = d_down;
    ffn_weights.down.shape = {hidden_units, intermediate_size};
    ffn_weights.down.is_transposed = true;
    
    auto ffn_input = new TensorWrapper<float>(
        Device::GPU, type,
        std::vector<int>{attention_dynamic_params.num_tokens, hidden_units},
        d_ffn_input
    );

    auto ffn_output = new TensorWrapper<float>(
        Device::GPU, type,
        std::vector<int>{attention_dynamic_params.num_tokens, hidden_units},
        d_ffn_output
    );

    TensorMap ffn_inputs{
        {"ffn_input", ffn_input}
    };

    TensorMap ffn_outputs{
        {"ffn_output", ffn_output}
    };

    std::cout << "Initializing FFN layer" << std::endl;

    auto ffn_layer = new LlamaFFNLayer<float>(
        head_num, 
        head_size, 
        intermediate_size,
        stream,
        cublas_wrapper,
        allocator
    );

    std::cout << "Start fwd" << std::endl;

    ffn_layer->forward(
        &ffn_inputs, 
        &ffn_outputs, 
        &ffn_weights, 
        &attention_dynamic_params
    );

    std::cout << "End fwd" << std::endl;

    // Free allocated memory
    deallocate(ffn_input, "new");
    deallocate(ffn_output, "new");
    deallocate(ffn_layer, "new");
    deallocate(cublas_wrapper, "new");
    deallocate(allocator, "new");

    deallocate(h_ffn_input, "new[]");
    deallocate(h_gate_up, "new[]");
    deallocate(h_down, "new[]");

    deallocate(d_ffn_input, "cudaMalloc");
    deallocate(d_gate_up, "cudaMalloc");
    deallocate(d_down, "cudaMalloc");
    deallocate(d_ffn_output, "cudaMalloc");
    return 0;
}
