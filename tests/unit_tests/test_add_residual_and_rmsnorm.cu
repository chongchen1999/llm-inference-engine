#include <algorithm>   // std::fill_n
#include <iostream>    // printf
#include <math.h>      // expf, log
#include <stdlib.h>    // malloc, free
#include <string>      // std::string
#include <vector>      // std::vector

#include "../../src/kernels/includes/add_residual_and_rmsnorm.h"
#include <stdio.h>

#define CHECK(call)                                   \
do {                                                  \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess) {                  \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

template <typename T>
void CPUFusedResidualAndRMSNorm(
    T *h_residual,
    T *h_decoder_out,
    T *h_bias,
    T *h_scale,
    float eps,
    int hidden_units,
    int num_tokens
) {
    for (int b = 0; b < num_tokens; ++b) {
        T inv_fenmu = 0.0f;
        T mean = 0.0f;
        T input = 0.0f;

        for (int i = 0; i < hidden_units; ++i) {
            input = h_decoder_out[b * hidden_units + i] +
                    h_residual[b * hidden_units + i] + h_bias[i];
        }

        T sum = 0.0f;
        for (int i = 0; i < hidden_units; ++i) {
            sum += input * input;
        }

        mean = sum / hidden_units;
        inv_fenmu = rsqrt(mean + eps);

        for (int i = 0; i < hidden_units; ++i) {
            h_decoder_out[b * hidden_units + i] = h_decoder_out[b * hidden_units + i] * inv_fenmu * h_scale[i];
        }
    }
}

template <typename T>
bool checkResult(T *CPUoutput, T *GPUoutput, int output_size) {
    for (int i = 0; i < output_size; ++i) {
        if (fabs(CPUoutput[i] - GPUoutput[i]) > 1e-3) {
            printf("The %dth result is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int num_tokens = 2048;
    const int hidden_units = 128;
    const int total_size = num_tokens * hidden_units;
    float eps = 0.5f;

    float *h_residual = (float *)malloc(sizeof(float) * total_size);
    float *d_residual;
    cudaMalloc((void **)&d_residual, sizeof(float) * total_size);
    std::fill_n(h_residual, total_size, 0.0f);

    float *h_decoder_out = (float *)malloc(sizeof(float) * total_size);
    float *decoder_out = (float *)malloc(sizeof(float) * total_size);
    float *d_decoder_out;
    cudaMalloc((void **)&d_decoder_out, sizeof(float) * total_size);
    std::fill_n(h_decoder_out, total_size, 1.0f);

    float *h_bias = (float *)malloc(sizeof(float) * hidden_units);
    float *d_bias;
    cudaMalloc((void **)&d_bias, sizeof(float) * hidden_units);
    std::fill_n(h_bias, hidden_units, 0.0f);

    float *h_scale = (float *)malloc(sizeof(float) * hidden_units);
    float *d_scale;
    cudaMalloc((void **)&d_scale, sizeof(float) * hidden_units);
    std::fill_n(h_scale, hidden_units, 1.0f);

    CHECK(cudaMemcpy(d_residual, h_residual, sizeof(float) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, sizeof(float) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, h_bias, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scale, h_scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));

    DataType type_float = getTensorType<float>();
    TensorWrapper<float> *decoder_out_tensor = new TensorWrapper<float>(
        Device::GPU, type_float, {num_tokens, hidden_units}, d_decoder_out
    );
    TensorWrapper<float> *residual_tensor = new TensorWrapper<float>(
        Device::GPU, type_float, {num_tokens, hidden_units}, d_residual
    );
    BaseWeight<float> norm;
    LayerNormWeight<float> scale;
    scale.gamma = d_scale;

    std::cout << "Before launch kernel" << std::endl;
    launchFusedAddBiasResidualAndRMSNorm(
        residual_tensor,
        decoder_out_tensor,
        &norm,
        d_scale,
        eps
    );
    std::cout << "After launch kernel" << std::endl;

    std::cout << "CUDA memcpy device to host" << std::endl;
    CHECK(cudaMemcpy(decoder_out, d_decoder_out, sizeof(float) * total_size, cudaMemcpyDeviceToHost));

    float *CPUout = (float *)malloc(sizeof(float) * total_size);
    std::fill_n(CPUout, total_size, 1.0f);
    CPUFusedResidualAndRMSNorm(
        h_residual,
        CPUout,
        h_bias,
        h_scale,
        eps,
        hidden_units,
        num_tokens
    );

    bool is_right = checkResult(CPUout, decoder_out, total_size);
    if (is_right) {
        std::cout << "Fused add residual and RMSNorm passed" << std::endl;
    } else {
        std::cout << "Fused add residual and RMSNorm failed" << std::endl;
    }

    std::cout << "Before free" << std::endl;
    free(h_residual);
    free(h_decoder_out);
    free(h_bias);
    free(h_scale);
    free(CPUout);
    free(decoder_out);
    cudaFree(d_residual);
    cudaFree(d_decoder_out);
    cudaFree(d_bias);
    cudaFree(d_scale);

    return 0;
}
