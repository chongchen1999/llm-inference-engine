#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

class BaseAllocator {
public:
    virtual ~BaseAllocator() = default;

    // Unified interface for all derived allocators to allocate buffers
    template<typename T>
    T *Malloc(T *ptr, size_t size, bool is_host) {
        return static_cast<T *>(UnifyMalloc(static_cast<void *>(ptr), size, is_host));
    }

    virtual void *UnifyMalloc(void *ptr, size_t size, bool is_host = false) = 0;

    template<typename T>
    void Free(T *ptr, bool is_host = false) {
        UnifyFree(static_cast<void *>(ptr), is_host);
    }

    virtual void UnifyFree(void *ptr, bool is_host = false) = 0;
};
