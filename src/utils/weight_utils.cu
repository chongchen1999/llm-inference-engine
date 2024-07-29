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
    LLM_CHECK_WITH_INFO(
        size >= static_cast<size_t>(0), 
        "Ask cudaMalloc size " + std::to_string(size) + " < 0, which is invalid."
    );
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

template<typename T>
std::vector<T> loadWeightFromBinHelper(
    const std::vector<size_t> *const shape, 
    const std::string *const filename
) {
    if (shape->size() > 2) {
        std::printf("[ERROR] shape should have less than two dims\n");
        return std::vector<T>();
    }

    size_t dim0 = (*shape)[0];
    size_t dim1 = shape->size() == 2 ? (*shape)[1] : 1;
    size_t size = dim0 * dim1;

    if (size == 0) {
        std::cout << "shape is zero, skip loading weight from file: " << *filename << std::endl;
        return std::vector<T>();
    }

    std::vector<T> host_array(size);
    std::ifstream input_file(*filename, std::ios::in | std::ios::binary);

    if (!input_file.is_open()) {
        std::cout << "File " << *filename << " cannot open, loading model fails!" << std::endl;
        return std::vector<T>();
    }

    size_t loaded_data_size = sizeof(T) * size;
    input_file.seekg(0, std::ios::end);
    std::cout << "Read " << loaded_data_size << " bytes from " << *filename << std::endl;
    input_file.seekg(0, std::ios::beg);
    input_file.read(reinterpret_cast<char *>(host_array.data()), loaded_data_size);

    size_t in_get_size = input_file.gcount();
    if (in_get_size != loaded_data_size) {
        return std::vector<T>();
    }

    input_file.close();
    return host_array;
}

template <typename OutputType, typename FileType>
struct loadWeightFromBin<OutputType, FileType, true> {
    static void loadFromFileToDevice(
        OutputType *ptr, 
        const std::vector<size_t> *const shape, 
        const std::string *const filename
    ) {
        std::vector<FileType> host_array = loadWeightFromBinHelper<FileType>(shape, filename);
        if (host_array.empty()) {
            return;
        }

        cudaHostToDeviceCopy(ptr, host_array.data(), host_array.size());
    }
};

template <typename OutputType, typename FileType>
struct loadWeightFromBin<OutputType, FileType, false> {
    static void loadFromFileToDevice(
        OutputType *ptr, 
        const std::vector<size_t> *const shape, 
        const std::string *const filename
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

template struct loadWeightFromBin<float, float, true>;
template struct loadWeightFromBin<half, half, true>;
template struct loadWeightFromBin<float, half, false>;
template struct loadWeightFromBin<half, float, false>;
