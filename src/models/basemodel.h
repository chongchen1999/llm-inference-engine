#pragma once

#include <string>
#include <functional>
#include <vector>
#include "src/utils/tensor.h"
#include "src/models/common_params.h"
#include "src/memory/allocator/base_allocator.h"
#include "src/kernels/includes/cublas_utils.h"

// Callback function for printing the generated content of each conversation round
using CallBack = std::function<void(int index, const char *generateContent)>;

class BaseModel {
public:
    std::string modelName;
    
    // Common data members required by all model subclasses
    cudaStream_t stream;
    cublasWrapper *cublasWrapper;
    BaseAllocator *allocator;
    cudaDeviceProp *cudaDeviceProp;

    BaseModel(
        cudaStream_t stream,
        cublasWrapper *cublasWrapper,
        BaseAllocator *allocator,
        cudaDeviceProp *cudaDeviceProp = nullptr
    ) :
        stream(stream),
        cublasWrapper(cublasWrapper),
        allocator(allocator),
        cudaDeviceProp(cudaDeviceProp) {}

    // Pure virtual functions that each model subclass must implement
    virtual void loadTokenizer(const std::string &file) = 0;
    virtual void loadWeights(const std::string &file) = 0;
    virtual void loadWeightsFromDummy() = 0;

    // Pure virtual functions for defining input, history, and response APIs
    // Generate the current round's prompt based on historical information and current input
    virtual std::vector<std::string> makeInput(
        const std::string &history,
        int round,
        const std::string &input
    ) const = 0;

    // Update history string based on the current round's response
    virtual std::string makeHistory(
        const std::string &history,
        int round,
        const std::string &input,
        const std::string &output
    ) const = 0;

    // Interface for returning response content
    virtual std::string response(
        const std::vector<std::string> &input,
        CallBack printRes
    ) = 0;
};
