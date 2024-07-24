#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

class BaseAllocator {
public:
    BaseAllocator() = default;
    virtual ~BaseAllocator() = default;

    // Unified interface for all derived allocators to allocate buffers
    template<typename T>
    void malloc(T **ptr, size_t size, bool is_host) {
        *ptr = static_cast<T *>(unifyMalloc(static_cast<void *>(*ptr), size, is_host));
    }

    virtual void *unifyMalloc(void *ptr, size_t size, bool is_host = false) = 0;

    template<typename T>
    void free(T *ptr, bool is_host = false) {
        unifyFree(static_cast<void *>(ptr), is_host);
    }

    virtual void unifyFree(void *ptr, bool is_host = false) = 0;
};
