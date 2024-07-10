#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include "src/kernels/rmsnorm.h"

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        std::cerr << "CUDA Error:\n"                  \
                  << "    File: " << __FILE__ << "\n" \
                  << "    Line: " << __LINE__ << "\n" \
                  << "    Error code: " << error_code << "\n" \
                  << "    Error text: " << cudaGetErrorString(error_code) << "\n"; \
        std::exit(1);                                 \
    }                                                 \
} while (0)

void CPUfusedresidandRMSNorm(float *h_decoder_out, float *h_weights, 
                             float eps, int hidden_units, int num_tokens) {
    for (int i = 0; i < num_tokens; ++i) {
        float inv_fenmu = 0.0f;
        float mean = 0.0f;
        float sum = 0.0f;
        for (int j = 0; j < hidden_units; ++j) {
            float temp = h_decoder_out[i * hidden_units + j];
            sum += temp * temp;
        }
        mean = sum / hidden_units;
        inv_fenmu = rsqrt(mean + eps);
        
        for (int j = 0; j < hidden_units; ++j) {
            h_decoder_out[j * hidden_units + j] = h_decoder_out[i * hidden_units + j] * inv_fenmu * h_weights[j];
        }
    }
}

template<typename T>
bool CheckResult(float *CPU_output, T *GPU_output, int output_size) {
    for (int i = 0; i < output_size; ++i) {
        float fp32GPUoutput = static_cast<float>(GPU_output[i]);
        if (std::fabs(CPU_output[i] - fp32GPUoutput) > 1e-6) {
            std::cerr << "The " << i << "th result is wrong, CPUoutput = " << CPU_output[i] 
                      << ", GPUoutput = " << fp32GPUoutput << "\n";
            return false;
        }
    }
    return true;
}

template<typename T>
void runTest() {
    const int num_tokens = 64;
    const int hidden_units = 4096;
    const int total_size = num_tokens * hidden_units;
    float eps = 1e-6;

    std::vector<T> h_decoder_out(total_size);
    std::vector<T> decoder_out(total_size);
    T *d_decoder_out;
    T *d_decoder_rsd;
    T *d_weights;

    if constexpr (std::is_same_v<T, half>) {
        std::fill(h_decoder_out.begin(), h_decoder_out.end(), half(1.0f));
    } else {
        for (int i = 0; i < total_size; ++i) {
            h_decoder_out[i] = static_cast<T>(i % 2 + 1);
        }
    }

    CHECK(cudaMalloc(&d_decoder_out, sizeof(T) * total_size));
    CHECK(cudaMalloc(&d_decoder_rsd, sizeof(T) * total_size));
    CHECK(cudaMalloc(&d_weights, sizeof(T) * hidden_units));

    std::vector<T> h_weights(hidden_units);
    if constexpr (std::is_same_v<T, half>) {
        std::fill(h_weights.begin(), h_weights.end(), half(1.0f));
    } else {
        for (int i = 0; i < hidden_units; ++i) {
            h_weights[i] = static_cast<T>(i % 2 + 1);
        }
    }

    CHECK(cudaMemcpy(d_decoder_out, h_decoder_out.data(), sizeof(T) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weights, h_weights.data(), sizeof(T) * hidden_units, cudaMemcpyHostToDevice));

    DataType type = getTensorType<T>();
    TensorWrapper<T> decoder_out_tensor(Device::GPU, type, {num_tokens, hidden_units}, d_decoder_out);
    TensorWrapper<T> decoder_rsd(Device::GPU, type, {num_tokens, hidden_units}, d_decoder_rsd);

    LayerNormWeight<T> scale;
    scale.gamma = d_weights;

    std::cout << "Before launch kernel\n";
    launchRMSNorm(&decoder_out_tensor, &decoder_rsd, scale, eps);
    std::cout << "After launch kernel\n";
    std::cout << "CUDA memcpy device to host\n";

    CHECK(cudaMemcpy(decoder_out.data(), d_decoder_out, sizeof(T) * total_size, cudaMemcpyDeviceToHost));

    std::vector<float> CPUout(total_size);
    if constexpr (std::is_same_v<T, half>) {
        std::fill(CPUout.begin(), CPUout.end(), 1.0f);
    } else {
        for (int i = 0; i < total_size; ++i) {
            CPUout[i] = static_cast<float>(i % 2 + 1);
        }
    }

    std::vector<float> cpu_scale(hidden_units);
    if constexpr (!std::is_same_v<T, half>) {
        for (int i = 0; i < hidden_units; ++i) {
            cpu_scale[i] = static_cast<float>(i % 2 + 1);
        }
    } else {
        std::fill(cpu_scale.begin(), cpu_scale.end(), 1.0f);
    }

    CPUfusedresidandRMSNorm(CPUout.data(), cpu_scale.data(), eps, hidden_units, num_tokens);
    bool is_right = CheckResult<T>(CPUout.data(), decoder_out.data(), total_size);

    if (is_right) {
        std::cout << "RMSNorm passed\n";
    }

    cudaFree(d_decoder_out);
    cudaFree(d_weights);
}

int main(int argc, char *argv[]) {
    bool use_fp16 = argc > 1 && std::string(argv[1]) == "1";
    if (use_fp16) {
        runTest<half>();
    } else {
        runTest<float>();
    }
    return 0;
}