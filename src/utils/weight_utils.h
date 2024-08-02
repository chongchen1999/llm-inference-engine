#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <iostream>
#include "macro.h"

template<typename T>
void GPUMalloc(T **ptr, size_t size);

template<typename T>
void GPUFree(T *ptr);

template <
    typename OutputType, 
    typename FileType, 
    const bool enabled = std::is_same<OutputType, FileType>::value
>
class loadWeightFromBin {
public:
    static void loadFromFileToDevice(
        OutputType *ptr,
        const std::vector<int> &shape,
        const std::string &filename
    );
};  // Template specialization (prototype)
