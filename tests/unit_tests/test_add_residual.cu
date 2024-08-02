#include <algorithm>    // std::fill_n
#include <iostream>     // std::printf
#include <cmath>        // std::fabs
#include <cstdlib>      // std::malloc, std::free
#include <string>       // std::string
#include <vector>       // std::vector

#include "../../src/kernels/includes/add_residual.h"

void CPUresidual(
    float *h_residual, 
    float *h_decoder_out,  
    int hidden_units, 
    int num_tokens
) {
    for (int b = 0; b < num_tokens; ++b) {
        for (int i = 0; i < hidden_units; ++i) {
            h_decoder_out[b * hidden_units + i] += h_residual[b * hidden_units + i];
        }
    }
}

bool checkResult(
    const float *CPUoutput, 
    const float *GPUoutput, 
    int output_size
) {
    for (int i = 0; i < output_size; ++i) {
        if (std::fabs(CPUoutput[i] - GPUoutput[i]) > 1e-3) {
            std::printf("The %dth result is wrong, CPUoutput = %f, GPUoutput = %f\n", 
                        i, CPUoutput[i], GPUoutput[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int num_tokens = 16;
    const int hidden_units = 4096;
    const int total_size = num_tokens * hidden_units;

    float *h_residual = static_cast<float *>(std::malloc(sizeof(float) * total_size));
    float *d_residual;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_residual), sizeof(float) * total_size));

    for (int i = 0; i < total_size; ++i) { 
        h_residual[i] = static_cast<float>(i % 2 + 1);
    }

    float *h_decoder_out = static_cast<float *>(std::malloc(sizeof(float) * total_size));
    float *decoder_out = static_cast<float *>(std::malloc(sizeof(float) * total_size));
    float *d_decoder_out;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_decoder_out), sizeof(float) * total_size));

    for (int i = 0; i < total_size; ++i) { 
        h_decoder_out[i] = static_cast<float>(i % 2 + 1);
    }

    CHECK(cudaMemcpy(d_residual, h_residual, sizeof(float) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, sizeof(float) * total_size, cudaMemcpyHostToDevice));

    const DataType type_float = getTensorType<float>();
    TensorWrapper<float> *decoder_out_tensor = new TensorWrapper<float>(
        Device::GPU, 
        type_float,
        { num_tokens, hidden_units }, 
        d_decoder_out
    );
    TensorWrapper<float> *residual_tensor = new TensorWrapper<float>(
        Device::GPU, 
        type_float,
        { num_tokens, hidden_units }, 
        d_residual
    );

    std::cout << "Before launching kernel" << std::endl;
    launchAddResidual(residual_tensor, decoder_out_tensor);
    std::cout << "After launching kernel" << std::endl;

    std::cout << "CUDA memcpy device to host" << std::endl;
    CHECK(cudaMemcpy(decoder_out, d_decoder_out, sizeof(float) * total_size, cudaMemcpyDeviceToHost));

    float *CPUout = static_cast<float *>(std::malloc(sizeof(float) * total_size));
    for (int i = 0; i < total_size; ++i) {
        CPUout[i] = static_cast<float>(i % 2 + 1);
    }

    CPUresidual(h_residual, CPUout, hidden_units, num_tokens);
    bool is_right = checkResult(CPUout, decoder_out, total_size);

    std::cout << "Before freeing resources" << std::endl;
    if (is_right) {
        std::cout << "addResidual kernel passed" << std::endl;
    }

    std::free(h_residual);
    std::free(h_decoder_out);
    std::free(CPUout);
    std::free(decoder_out);
    cudaFree(d_residual);
    cudaFree(d_decoder_out);

    return 0;
}
