#include "macro.h"

void initializeCuda() {
    // Initialize CUDA device
    CHECK(cudaSetDevice(0));
    std::cout << "CUDA device initialized successfully." << std::endl;
}

void allocateMemory(float** d_A, int size) {
    // Allocate memory on GPU
    CHECK(cudaMalloc((void **)d_A, size * sizeof(float)));
    std::cout << "Memory allocated successfully on GPU." << std::endl;
}

void freeMemory(float* d_A) {
    // Free memory on GPU
    CHECK(cudaFree(d_A));
    std::cout << "Memory freed successfully on GPU." << std::endl;
}

void cublasExample() {
    // Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    std::cout << "cuBLAS initialized successfully." << std::endl;

    // Example: Allocate and set up vectors on the device
    float *d_A;
    int size = 10;

    allocateMemory(&d_A, size);

    // Fill the vector with values
    float h_A[size];
    for (int i = 0; i < size; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    CHECK_CUBLAS(cublasSetVector(size, sizeof(float), h_A, 1, d_A, 1));

    // Perform some cuBLAS operations here (e.g., scaling a vector)
    float alpha = 2.0f;
    CHECK_CUBLAS(cublasSscal(handle, size, &alpha, d_A, 1));
    std::cout << "cuBLAS operation performed successfully." << std::endl;

    // Synchronize and check for any errors
    DeviceSyncAndCheckCudaError();

    // Clean up
    freeMemory(d_A);
    CHECK_CUBLAS(cublasDestroy(handle));
}

int main() {
    try {
        initializeCuda();
        cublasExample();
        std::cout << "All operations completed successfully." << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
