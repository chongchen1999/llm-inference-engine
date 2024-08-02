#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Function to check CUDA calls and provide detailed error messages
inline void checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error:\n"
                  << "    File:       " << file << '\n'
                  << "    Line:       " << line << '\n'
                  << "    Error code: " << result << '\n'
                  << "    Error text: " << cudaGetErrorString(result) << '\n';
        exit(1);
    }
}

#define CHECK(call) checkCuda((call), __FILE__, __LINE__)

// Function to get CUDA error string
static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorString(error);
}

// Function to get cuBLAS error string
static const char* _cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:           return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:   return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:      return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:     return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:     return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:     return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:  return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:    return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:     return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:     return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                              return "<unknown>";
    }
}

// Template function to check CUDA or cuBLAS results
template<typename T>
void check(T result, const char* func, const char* file, int line) {
    if (result != 0) {
        throw std::runtime_error(
            "[TM][ERROR] CUDA runtime error: " + std::string(_cudaGetErrorEnum(result)) + 
            " " + file + ":" + std::to_string(line) + " \n"
        );
    }
}

#define CHECK_CUBLAS(val) check((val), #val, __FILE__, __LINE__)

// Function to synchronize device and check for errors
inline void syncAndCheck(const char* file, int line) {
    cudaDeviceSynchronize();
    cudaError_t result = cudaGetLastError();
    if (result) {
        throw std::runtime_error(
            "[TM][ERROR] CUDA runtime error: " + std::string(_cudaGetErrorEnum(result)) + 
            " " + file + ":" + std::to_string(line) + " \n"
        );
    }
}

#define DeviceSyncAndCheckCudaError() syncAndCheck(__FILE__, __LINE__)

// Function to throw runtime errors with file and line information
[[noreturn]] inline void throwRuntimeError(const char* file, int line, const std::string& info = "") {
    throw std::runtime_error(
        "[oneLLM][ERROR] " + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n"
    );
}

// Function to assert and throw errors with additional information
inline void llmAssert(bool result, const char* file, int line, const std::string& info = "") {
    if (!result) {
        throwRuntimeError(file, line, info);
    }
}

#define LLM_CHECK(val) llmAssert(val, __FILE__, __LINE__)

inline void llmCheckWithInfo(bool val, const std::string& info, const char* file, int line) {
    if (!val) {
        llmAssert(val, file, line, info);
    }
}

#define LLM_CHECK_WITH_INFO(val, info) llmCheckWithInfo((val), (info), __FILE__, __LINE__)
