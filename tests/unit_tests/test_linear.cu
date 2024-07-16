#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <cmath>       // expf, log
#include <cstdlib>     // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <cstdio>
#include <fstream>
#include <ctime>
#include "src/utils/macro.h"
#include "src/kernels/includes/linear.h"
#include "src/weights/base_weights.h"

void CPUlinear(float *input, float *weight, float *output,
               int MM, int KK, int NN) {
    for (int i = 0; i < MM; ++i) {
        for (int k = 0; k < KK; ++k) {
            float temp = input[i * KK + k];
            for (int j = 0; j < NN; ++j) {
                output[i * NN + j] +=  temp * weight[j * KK + k];
            }
        }
    }
}

bool checkResult(float *CPUoutput, float *GPUoutput, int output_size) {
    for (int i = 0; i < output_size; ++i) {
        if (i < 5) {
            printf("%dth res, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
        }
        if (fabs(CPUoutput[i] - GPUoutput[i]) > 1e-3) {
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    srand(233);
    const int seqlen = 64;
    const int hidden_units = 4096;
    int hidden_units_2 = 0;
    int output_size = 0;

    hidden_units_2 = hidden_units * hidden_units;
    output_size = seqlen * hidden_units;
    
    float *host_weights, *device_weights;
    host_weights = (float *)malloc(sizeof(float) * hidden_units_2);
    cudaMalloc((void **)&device_weights, sizeof(float) * hidden_units_2);
    for (int i = 0; i < hidden_units_2; ++i) {
        host_weights[i] = (float)(rand() % 3);  // pattern: 1, 2, 1, 2...
    }

    float *host_input = (float *)malloc(sizeof(float) * seqlen * hidden_units);
    float *device_input;
    cudaMalloc((void **)&device_input, sizeof(float) * seqlen * hidden_units);
    for (int i = 0; i < seqlen * hidden_units; ++i) {
        host_input[i] = (float)(rand() % 3);
    }

    float *host_output = (float *)malloc(sizeof(float) * output_size);
    float *device_output;
    cudaMalloc((void **)&device_output, sizeof(float) * output_size);

    CHECK(cudaMemcpy(device_input, host_input, sizeof(float) * hidden_units * seqlen, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_weights, host_weights, sizeof(float) * hidden_units_2, cudaMemcpyHostToDevice));

    DataType type = getTensorType<float>();
    WeightType wtype = getWeightType<float>();
    TensorWrapper<float> *in = new TensorWrapper<float>(Device::GPU, type, {seqlen, hidden_units}, device_input);
    //print_mat(host_input, seqlen, hidden_units);

    BaseWeight<float> weight;
    weight.shape = {hidden_units, hidden_units};
    weight.data = device_weights;
    weight.type = wtype;

    TensorWrapper<float> *out;
    out = new TensorWrapper<float>(Device::GPU, type, {seqlen, hidden_units}, device_output);

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    cublasWrapper *cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    cublas_wrapper->setFP32GemmConfig();

    std::cout << "before launch kernel" << std::endl;
    launchLinearGemm(in, weight, out, cublas_wrapper, false, true);
    std::cout << "after launch kernel" << std::endl;

    std::cout << "cuda memcpy device to host" << std::endl;
    CHECK(cudaMemcpy(host_output, device_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost));

    float *CPUout = (float *)malloc(sizeof(float) * output_size);
    memset(CPUout, 0, sizeof(float) * output_size);
    CPUlinear(host_input, host_weights, CPUout, seqlen, hidden_units, hidden_units);

    bool is_right = checkResult(CPUout, host_output, output_size);
    if (is_right) {
        std::cout << "linear passed" << std::endl;
    }
    free(host_input);
    free(host_weights);
    free(host_output);
    free(CPUout);
    cudaFree(device_input);
    cudaFree(device_weights);
    cudaFree(device_output);
    return 0;
}