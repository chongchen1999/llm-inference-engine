#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <fstream>
#include "src/utils/tensor.h"
#include "src/weights/base_weights.h"
#include "src/utils/macro.h"

// Note: Overloaded functions for saving intermediate output tensor to debug
// Because LLMs have many layers, overloaded functions are provided to specify layer ID 
// and print specific layer output tensor for debugging.
// After saving the tensor into the specified file, you can compare results in 
// tests/unitests/test_data_compare.cu by specifying the file path.

template<typename T>
void saveTensor(
    TensorWrapper<T> *input,
    const std::string &filename
) {
    int Bm = 0;
    int Bk = 0;

    if (input->shape.size() == 4) {
        Bm = input->shape[0] * input->shape[1];
        Bk = input->shape[3] * input->shape[2];
    } else if (input->shape.size() == 3) {
        Bm = input->shape[0];
        Bk = input->shape[1] * input->shape[2];
    } else if (input->shape.size() == 2) {
        Bm = input->shape[0];
        Bk = input->shape[1];
    }

    T *icpu = static_cast<T *>(malloc(sizeof(T) * Bm * Bk));
    cudaMemcpy(icpu, input->data, sizeof(T) * Bm * Bk, cudaMemcpyDeviceToHost);

    std::ofstream file;
    std::cout << "Saving intermediate tensor in " << filename << "\n";
    file.open("/home/data/" + filename, std::ofstream::binary);
    file.write(reinterpret_cast<const char *>(icpu), sizeof(T) * Bm * Bk);
    file.close();

    free(icpu);
}

template<typename T>
void saveTensor(
    TensorWrapper<T> *input,
    const std::string &filename,
    TensorWrapper<int> *layer_id
) {
    const int id = layer_id->getVal();
    if (id > 2) {
        return;
    }

    int Bm = 0;
    int Bk = 0;

    if (input->shape.size() == 4) {
        Bm = input->shape[0] * input->shape[1];
        Bk = input->shape[3] * input->shape[2];
    } else if (input->shape.size() == 3) {
        Bm = input->shape[0];
        Bk = input->shape[1] * input->shape[2];
    } else if (input->shape.size() == 2) {
        Bm = input->shape[0];
        Bk = input->shape[1];
    }

    T *icpu = static_cast<T *>(malloc(sizeof(T) * Bm * Bk));
    cudaMemcpy(icpu, input->data, sizeof(T) * Bm * Bk, cudaMemcpyDeviceToHost);

    std::ofstream file;
    std::cout << "Saving intermediate tensor in " << filename << "\n";
    file.open("/home/data/" + std::to_string(id) + "_" + filename, std::ofstream::binary);
    file.write(reinterpret_cast<const char *>(icpu), sizeof(T) * Bm * Bk);
    file.close();

    free(icpu);
}

template<typename T>
void saveTensor(
    TensorWrapper<T> *input,
    const std::string &filename,
    int layer_id
) {
    if (layer_id > 2) {
        return;
    }

    int Bm = 0;
    int Bk = 0;

    if (input->shape.size() == 4) {
        Bm = input->shape[0] * input->shape[1];
        Bk = input->shape[3] * input->shape[2];
    } else if (input->shape.size() == 3) {
        Bm = input->shape[0];
        Bk = input->shape[1] * input->shape[2];
    } else if (input->shape.size() == 2) {
        Bm = input->shape[0];
        Bk = input->shape[1];
    }

    T *icpu = static_cast<T *>(malloc(sizeof(T) * Bm * Bk));
    cudaMemcpy(icpu, input->data, sizeof(T) * Bm * Bk, cudaMemcpyDeviceToHost);

    std::ofstream file;
    std::cout << "Saving intermediate tensor in " << filename << "\n";
    file.open("/home/data/" + std::to_string(layer_id) + "_" + filename, std::ofstream::binary);
    file.write(reinterpret_cast<const char *>(icpu), sizeof(T) * Bm * Bk);
    file.close();

    free(icpu);
    return 0;
}
