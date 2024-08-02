#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

class BaseAllocator {
public:
    BaseAllocator() = default;
    virtual ~BaseAllocator() = default;

    // Unified interface for all derived allocators to allocate buffers
    template<typename T>
    void malloc(T **ptr, size_t size, bool is_host) {
        // unifyMalloc(reinterpret_cast<void **>(ptr), size, is_host);
        if (is_host) {
            *ptr = reinterpret_cast<T *>(::malloc(size));
        } else {
            cudaMalloc(reinterpret_cast<void **>(ptr), size);
        }
    }

    virtual void unifyMalloc(void **ptr, size_t size, bool is_host = false) = 0;

    template<typename T>
    void free(T *ptr, bool is_host = false) {
        // unifyFree(static_cast<void *>(ptr), is_host);
        if (!ptr) {
            return;
        }
        if (is_host) {
            ::free(static_cast<void *>(ptr));
        } else {
            cudaFree(static_cast<void *>(ptr));
        }
    }

    virtual void unifyFree(void *ptr, bool is_host = false) = 0;
};
