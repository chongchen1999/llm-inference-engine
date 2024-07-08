#include <algorithm>   // std::fill_n
#include <iostream>    // std::cout, std::endl, snprintf
#include <cmath>       // expf, log
#include <cstdlib>     // rand, malloc, free
#include <string>      // std::string
#include <vector>      // std::vector
#include <random>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "src/kernels/input_embedding.h"

// Macros for checking CUDA errors
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

// CPU implementation of embedding
void cpuEmbedding(const int* input_ids, float* output, float* embed_table,
                  const int max_context_token_num, const int hidden_size, const int vocab_size) {
    for (int i = 0; i < max_context_token_num; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            output[j + i * hidden_size] = embed_table[j + input_ids[i] * hidden_size];
        }
    }
}

// Function to check results between CPU and GPU
bool checkResults(float* h_output, float* d_output, const int output_size) {
    float* d_output_cpu = (float*) malloc(output_size * sizeof(float)); // prepare for cpu check
    CHECK(cudaMemcpy(d_output_cpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    const float EPS = 1e-5;
    for (int i = 0; i < output_size; ++i) {
        if (fabs(d_output_cpu[i] - h_output[i]) / (fabs(h_output[i]) + EPS) > EPS) {
            std::cout << "DEV : ";
            for (int j = std::max(0, i - 10); j < std::min(output_size, i + 10); ++j) {
                std::cout << d_output_cpu[j] << " ";
            }
            std::cout << std::endl;

            std::cout << "CPU : ";
            for (int j = std::max(0, i - 10); j < std::min(output_size, i + 10); ++j) {
                std::cout << h_output[j] << " ";
            }
            std::cout << std::endl;
            free(d_output_cpu);
            return false;
        }
    }
    free(d_output_cpu);
    return true;
}

template <typename T>
void allocateMemoryAndInitialize(T*& h_table, T*& h_output, int table_size, int output_size) {
    h_table = (T*) malloc(table_size * sizeof(T));
    h_output = (T*) malloc(output_size * sizeof(T));
}

template <typename T>
void initializeHostMemory(int* h_input, int max_context_token_num, int vocab_size, int table_size, int hidden_size, T* h_table) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_int(0, vocab_size - 1);

    for (int i = 0; i < max_context_token_num; ++i) {
        h_input[i] = dis_int(gen);
        printf("h_input[%d] = %d\n", i, h_input[i]);
    }
    for (int i = 0; i < table_size; ++i) {
        h_table[i] = static_cast<T>(i / hidden_size);
    }
}

template <typename T>
void copyMemoryToDevice(int* h_input, int* d_input, T* h_table, T* d_table, int input_size, int table_size) {
    CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_table, h_table, table_size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void launchEmbedding(int* d_input, T* d_table, T* d_output, int max_context_token_num, int hidden_size, int output_size) {
    DataType type = getTensorType<T>();
    DataType type_int = getTensorType<int>();

    TensorWrapper<int>* input_ids = new TensorWrapper<int>(Device::GPU, type_int, {max_context_token_num}, d_input);
    TensorWrapper<T>* output = new TensorWrapper<T>(Device::GPU, type, {max_context_token_num, hidden_size}, d_output);
    EmbeddingWeight<T> emb_table;
    emb_table.data = d_table;

    launchInputEmbedding(input_ids, output, &emb_table);

    T* h_output = (T*) malloc(output_size * sizeof(T));
    CHECK(cudaMemcpy(h_output, output->data, output_size * sizeof(T), cudaMemcpyDeviceToHost));

    std::cout << "printf h_output for check" << std::endl;
    for (int i = 0; i < max_context_token_num; ++i) {
        std::cout << static_cast<float>(h_output[i * hidden_size]) << std::endl;
    }

    free(h_output);
}

template <typename T>
void process(int* h_input, int input_size, int table_size, int output_size, 
    int max_context_token_num, int hidden_size, int vocab_size) {
    T* h_table;
    T* h_output;

    allocateMemoryAndInitialize(h_table, h_output, table_size, output_size);

    initializeHostMemory(h_input, max_context_token_num, vocab_size, table_size, hidden_size, h_table);

    int* d_input;
    T *d_table, *d_output;

    cudaMalloc((void**)&d_input, input_size * sizeof(int));
    cudaMalloc((void**)&d_table, table_size * sizeof(T));
    cudaMalloc((void**)&d_output, output_size * sizeof(T));

    copyMemoryToDevice(h_input, d_input, h_table, d_table, input_size, table_size);

    launchEmbedding(d_input, d_table, d_output, max_context_token_num, hidden_size, output_size);

    cudaFree(d_output);
    cudaFree(d_table);
    cudaFree(d_input);
    free(h_output);
    free(h_table);
}

int main(int argc, char *argv[]) {
    const int max_context_token_num = 64;
    const int hidden_size = 4096;
    const int vocab_size = 30000;
    const int input_size = max_context_token_num;
    const int table_size = vocab_size * hidden_size;
    const int output_size = max_context_token_num * hidden_size;

    int* h_input = (int*) malloc(input_size * sizeof(int));
    if (argc > 1) {
        process<float>(h_input, input_size, table_size, output_size, max_context_token_num, hidden_size, vocab_size);
    } else {
        process<half>(h_input, input_size, table_size, output_size, max_context_token_num, hidden_size, vocab_size);
    }
    free(h_input);
    return 0;
}