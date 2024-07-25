#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <iostream>
#include "src/utils/macro.h"

template<typename T>
void GPUMalloc(T **ptr, size_t size);

template<typename T>
void GPUFree(T *ptr);

template <
    typename OutputType, 
    typename FileType, 
    const bool enabled = std::is_same<OutputType, FileType>::value
>
struct loadWeightFromBin {
    static void loadFromFileToDevice(
        OutputType *ptr,
        const std::vector<size_t> *const shape,
        const std::string * const filename
    );
};  // Template specialization (prototype)
