#include "weight_utils.h"

template<typename OutputType, typename InputType>
__device__ __forceinline__ OutputType typeCast(InputType val) {
    return val;
}

template<>
__device__ __forceinline__ float typeCast(half val) {
    return __half2float(val);
}

template<>
__device__ __forceinline__ half typeCast(float val) {
    return __float2half(val);
}

template<typename T>
void GPUMalloc(T **ptr, size_t size) {
    CHECK(cudaMalloc(reinterpret_cast<void **>(ptr), sizeof(T) * size));
}

template void GPUMalloc(float **ptr, size_t size);
template void GPUMalloc(half **ptr, size_t size);

template<typename T>
void GPUFree(T *ptr) {
    if (ptr != nullptr) {
        CHECK(cudaFree(ptr));
        ptr = nullptr;
    }
}

template void GPUFree(float *ptr);
template void GPUFree(half *ptr);

template<typename T>
void cudaHostToDeviceCopy(T *dst, const T *src, size_t size) {
    CHECK(cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template void cudaHostToDeviceCopy(float *dst, const float *src, size_t size);
template void cudaHostToDeviceCopy(half *dst, const half *src, size_t size);

template<typename InputType, typename OutputType>
__global__ void typeConversion(OutputType *dst, const InputType *src, int size) {
    const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    const int total_thread_nums = blockDim.x * gridDim.x;
    
    for (int index = gtid; index < size; index += total_thread_nums) {
        dst[index] = typeCast<OutputType>(src[index]);
    }
}

template<typename InputType, typename OutputType>
void cudaTypeConversion(OutputType *dst, const InputType *src, int size) {
    dim3 grid(256);
    dim3 block(256);
    typeConversion<InputType, OutputType><<<grid, block>>>(dst, src, size);
}

template void cudaTypeConversion(float *dst, const half *src, int size);
template void cudaTypeConversion(half *dst, const float *src, int size);

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

template<typename T>
std::vector<T> loadWeightFromBinHelper(
    const std::vector<size_t> &shape, 
    const std::string &filename
) {
    // Check if shape has fewer than two dimensions
    if (shape.size() > 2) {
        std::cerr << "[ERROR] shape should have less than two dims\n";
        return std::vector<T>();
    }

    // Ensure shape is not empty
    if (shape.empty()) {
        std::cerr << "[ERROR] shape is empty\n";
        return std::vector<T>();
    }

    size_t dim0 = shape[0];
    size_t dim1 = shape.size() == 2 ? shape[1] : 1;
    size_t size = dim0 * dim1;

    if (size == 0) {
        std::cout << "Shape is zero, skipping loading weight from file: " << filename << std::endl;
        return std::vector<T>();
    }

    std::vector<T> host_array(size);
    std::ifstream input_file(filename, std::ios::in | std::ios::binary);

    if (!input_file.is_open()) {
        std::cerr << "File " << filename << " cannot be opened, loading model fails!" << std::endl;
        return std::vector<T>();
    }

    size_t loaded_data_size = sizeof(T) * size;
    input_file.seekg(0, std::ios::end);
    std::streamsize file_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);

    std::cout << "Expected to read " << loaded_data_size << " bytes from " << filename << std::endl;

    if (file_size < loaded_data_size) {
        std::cerr << "File " << filename << " is too small, expected " << loaded_data_size << " bytes but got " << file_size << " bytes" << std::endl;
        return std::vector<T>();
    }

    input_file.read(reinterpret_cast<char *>(host_array.data()), loaded_data_size);

    if (!input_file) {
        std::cerr << "Error reading from file " << filename << std::endl;
        return std::vector<T>();
    }

    input_file.close();
    return host_array;
}

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

template<typename T>
std::vector<T> loadWeightFromBinHelper(
    const std::vector<int> &shape, 
    const std::string &filename
) {
    // Check if shape has fewer than two dimensions
    if (shape.size() > 2) {
        std::cerr << "[ERROR] shape should have less than two dims\n";
        return std::vector<T>();
    }

    // Ensure shape is not empty
    if (shape.empty()) {
        std::cerr << "[ERROR] shape is empty\n";
        return std::vector<T>();
    }

    int dim0 = shape[0];
    int dim1 = shape.size() == 2 ? shape[1] : 1;
    int size = dim0 * dim1;

    if (size <= 0) {
        std::cout << "Shape is zero or negative, skipping loading weight from file: " << filename << std::endl;
        return std::vector<T>();
    }

    std::vector<T> host_array(size);
    std::ifstream input_file(filename, std::ios::in | std::ios::binary);

    if (!input_file.is_open()) {
        std::cerr << "File " << filename << " cannot be opened, loading model fails!" << std::endl;
        return std::vector<T>();
    }

    std::streamsize loaded_data_size = sizeof(T) * size;
    input_file.seekg(0, std::ios::end);
    std::streamsize file_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);

    std::cout << "Expected to read " << loaded_data_size << " bytes from " << filename << std::endl;

    if (file_size < loaded_data_size) {
        std::cerr << "File " << filename << " is too small, expected " << loaded_data_size << " bytes but got " << file_size << " bytes" << std::endl;
        return std::vector<T>();
    }

    input_file.read(reinterpret_cast<char *>(host_array.data()), loaded_data_size);

    if (!input_file) {
        std::cerr << "Error reading from file " << filename << std::endl;
        return std::vector<T>();
    }

    input_file.close();
    return host_array;
}

template <typename OutputType, typename FileType>
class loadWeightFromBin<OutputType, FileType, true> {
public:
    static void loadFromFileToDevice(
        OutputType *ptr, 
        const std::vector<int> &shape, 
        const std::string &filename
    ) {
        std::vector<FileType> host_array = loadWeightFromBinHelper<FileType>(shape, filename);
        if (host_array.empty()) {
            return;
        }
        cudaHostToDeviceCopy(ptr, host_array.data(), host_array.size());
    }
};

template <typename OutputType, typename FileType>
class loadWeightFromBin<OutputType, FileType, false> {
public:
    static void loadFromFileToDevice(
        OutputType *ptr, 
        const std::vector<int> &shape, 
        const std::string &filename
    ) {
        std::vector<FileType> host_array = loadWeightFromBinHelper<FileType>(shape, filename);
        if (host_array.empty()) {
            return;
        }

        FileType *temp_ptr;
        GPUMalloc(&temp_ptr, host_array.size());
        cudaHostToDeviceCopy(temp_ptr, host_array.data(), host_array.size());
        cudaTypeConversion(ptr, temp_ptr, host_array.size());
        GPUFree(temp_ptr);
    }
};

template class loadWeightFromBin<float, float, true>;
template class loadWeightFromBin<half, half, true>;
template class loadWeightFromBin<float, half, false>;
template class loadWeightFromBin<half, float, false>;
