#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <cmath>       // expf, log
#include <cstdlib>     // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include "../../src/kernels/includes/silu_and_mul.h"

// () Note: not sure CPU implementation is absolutely right and the GPU kernel is right compared with HF.
// When you are implementing LLMs inference on CPU, you can reuse the CPU kernel and test its correctness.
// () Note:
// `./test_swiglu 1` to test half GPU kernel
// `./test_swiglu` to test fp32 GPU kernel

template <typename T>
void CPUSwiGLU(
    T *input,
    T *output,
    int batch_size,
    int intermedia_size
) {
    float silu_out = 0.0f;
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        for (int i = 0; i < intermedia_size; ++i) {
            const int offset_gate = batch_id * 2 * intermedia_size + i;
            const int offset_up = batch_id * 2 * intermedia_size + i + intermedia_size;
            const int out_offset = batch_id * intermedia_size + i;
            silu_out = static_cast<float>(input[offset_gate]) / (1.0f + expf(-static_cast<float>(input[offset_gate])));
            output[out_offset] = static_cast<T>(silu_out * static_cast<float>(input[offset_up]));
        }
    }
}

template <typename T>
bool checkResult(
    T *CPUoutput,
    T *GPUoutput,
    int output_size
) {
    for (int i = 0; i < output_size; ++i) {
        if (fabs(static_cast<float>(CPUoutput[i]) - static_cast<float>(GPUoutput[i])) > 1e-6) {
            printf("The %dth result is wrong, CPUoutput = %f, GPUoutput = %f\n", 
                   i, 
                   static_cast<float>(CPUoutput[i]), 
                   static_cast<float>(GPUoutput[i]));
            return false;
        }
    }
    return true;
}

template <typename T>
void test_swiglu(
    int batch_size,
    int intermedia_size,
    int input_size,
    int output_size
) {
    T *h_input = static_cast<T *>(malloc(sizeof(T) * input_size));
    T *d_input;
    cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(T) * input_size);
    
    T *h_output = static_cast<T *>(malloc(sizeof(T) * output_size));
    T *d_output;
    cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(T) * output_size);
    
    for (int i = 0; i < input_size; ++i) { // initialize host data
        h_input[i] = static_cast<T>(1);
    }
    
    cudaMemcpy(d_input, h_input, sizeof(T) * input_size, cudaMemcpyHostToDevice);
    
    DataType type = getTensorType<T>();
    auto *input_tensor = new TensorWrapper<T>(Device::GPU, type, {batch_size, 2, intermedia_size}, d_input);
    auto *output_tensor = new TensorWrapper<T>(Device::GPU, type, {batch_size, intermedia_size}, d_output);
    
    launchSiluAndMul(input_tensor, output_tensor);
    
    cudaMemcpy(h_output, d_output, sizeof(T) * output_size, cudaMemcpyDeviceToHost);
    
    T *CPU_output = static_cast<T *>(malloc(sizeof(T) * output_size));
    CPUSwiGLU(h_input, CPU_output, batch_size, intermedia_size);
    
    bool is_true = checkResult(CPU_output, h_output, output_size);
    if (is_true) {
        printf("Test passed\n");
    } else {
        printf("Test failed\n");
    }

    free(h_input);
    free(h_output);
    free(CPU_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char **argv) {
    constexpr int batch_size = 128;
    constexpr int intermedia_size = 11008;
    constexpr int input_size = batch_size * intermedia_size * 2;
    constexpr int output_size = batch_size * intermedia_size;
    
    if (argc > 1 && argv[1]) {
        test_swiglu<half>(batch_size, intermedia_size, input_size, output_size);
    } else {
        test_swiglu<float>(batch_size, intermedia_size, input_size, output_size);
    }
    
    return 0;
}
