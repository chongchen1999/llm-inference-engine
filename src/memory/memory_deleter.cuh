#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

template <typename T>
void deallocate(T *ptr, const std::string &alloc_type) {
    if (ptr == nullptr) {
        return;
    }

    if (alloc_type == "new") {
        delete ptr;
    } else if (alloc_type == "new[]") {
        delete[] ptr;
    } else if (alloc_type == "cudaMalloc") {
        cudaFree(ptr);
    } else if (alloc_type == "malloc") {
        free(ptr);
    } else {
        std::cerr << "Unknown allocation type for deallocation." << std::endl;
    }
    ptr = nullptr;
}