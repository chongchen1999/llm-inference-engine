#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <cmath>       // expf, log
#include <cstdlib>     // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <cstdio>
#include <fstream>
#include <ctime>
#include <memory>      // std::unique_ptr

#include "../../src/utils/macro.h"
#include "../../src/kernels/includes/linear.h"
#include "../../src/weights/includes/base_weights.h"

// CPU linear operation function
void CPUlinear(
    float *input,
    float *weight,
    float *output,
    int MM,
    int KK,
    int NN
) {
    for (int i = 0; i < MM; ++i) {
        for (int k = 0; k < KK; ++k) {
            float temp = input[i * KK + k];
            for (int j = 0; j < NN; ++j) {
                output[i * NN + j] += temp * weight[j * KK + k];
            }
        }
    }
}

// Check results function
bool checkResult(
    float *CPUoutput,
    float *GPUoutput,
    int output_size
) {
    for (int i = 0; i < output_size; ++i) {
        if (i < 5) {
            printf("%dth res, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
        }
        if (fabs(CPUoutput[i] - GPUoutput[i]) > 1e-3) {
            printf("The %dth result is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    srand(233);

    const int seqlen = 64;
    const int hidden_units = 4096;
    int hidden_units_2 = hidden_units * hidden_units;
    int output_size = seqlen * hidden_units;

    // Use smart pointers for CPU memory
    auto host_weights = std::make_unique<float[]>(hidden_units_2);
    auto host_input = std::make_unique<float[]>(seqlen * hidden_units);
    auto host_output = std::make_unique<float[]>(output_size);
    auto CPUout = std::make_unique<float[]>(output_size);

    // Raw pointers for GPU memory
    float *device_weights;
    float *device_input;
    float *device_output;

    cudaMalloc((void **)&device_weights, sizeof(float) * hidden_units_2);
    cudaMalloc((void **)&device_input, sizeof(float) * seqlen * hidden_units);
    cudaMalloc((void **)&device_output, sizeof(float) * output_size);

    for (int i = 0; i < hidden_units_2; ++i) {
        host_weights[i] = static_cast<float>(rand() % 3);  // Pattern: 1, 2, 1, 2...
    }

    for (int i = 0; i < seqlen * hidden_units; ++i) {
        host_input[i] = static_cast<float>(rand() % 3);
    }

    ::memset(CPUout.get(), 0, sizeof(float) * output_size);

    CHECK(cudaMemcpy(device_input, host_input.get(), sizeof(float) * seqlen * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_weights, host_weights.get(), sizeof(float) * hidden_units_2, cudaMemcpyHostToDevice));

    printf("ok");

    DataType type = getTensorType<float>();
    WeightType wtype = getWeightType<float>();

    auto in = std::make_unique<TensorWrapper<float>>(Device::GPU, type, std::vector<int>{seqlen, hidden_units}, device_input);
    BaseWeight<float> weight;
    weight.shape = {hidden_units, hidden_units};
    weight.data = device_weights;
    weight.type = wtype;

    auto out = std::make_unique<TensorWrapper<float>>(Device::GPU, type, std::vector<int>{seqlen, hidden_units}, device_output);

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);

    auto cublas_wrapper = std::make_unique<CublasWrapper>(cublas_handle, cublaslt_handle);
    cublas_wrapper->setFP32GemmConfig();

    std::cout << "Before launch kernel" << std::endl;
    launchLinearGemm(in.get(), &weight, out.get(), cublas_wrapper.get(), false, true);
    std::cout << "After launch kernel" << std::endl;

    std::cout << "CUDA memcpy device to host" << std::endl;
    CHECK(cudaMemcpy(host_output.get(), device_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost));

    CPUlinear(
        host_input.get(),
        host_weights.get(),
        CPUout.get(),
        seqlen,
        hidden_units,
        hidden_units
    );

    bool is_right = checkResult(CPUout.get(), host_output.get(), output_size);
    if (is_right) {
        std::cout << "Linear passed" << std::endl;
    }

    cudaFree(device_input);
    cudaFree(device_weights);
    cudaFree(device_output);

    return 0;
}
