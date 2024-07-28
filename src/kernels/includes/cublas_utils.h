#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include "src/utils/macro.h"

// 1. cuBLAS API: must allocate the required matrices in GPU memory space, 
// fill them with data, call the sequence of desired cuBLAS functions, and then upload the results back to the host.
// 2. cuBLASXt API: have the data on the Host
// 3. cuBLASLt API: lightweight library dedicated to GEMM with a new flexible API. 
// Adds flexibility in matrix data layouts, input types, compute types, and also in choosing algorithmic implementations 
// and heuristics through parameter programmability.

class CublasWrapper {
private:
    cublasHandle_t cublasHandle_;
    cublasLtHandle_t cublasLtHandle_;

    cudaDataType_t Atype_;
    cudaDataType_t Btype_;
    cudaDataType_t Ctype_;
    cudaDataType_t computeType_;

public:
    CublasWrapper(
        cublasHandle_t cublasHandle,
        cublasLtHandle_t cublasLtHandle
    );

    ~CublasWrapper();

    void setFP32GemmConfig();
    void setFP16GemmConfig();

    // For proj matmul
    void gemm(
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const void *A,
        int lda,
        const void *B,
        int ldb,
        void *C,
        int ldc,
        float alpha,
        float beta
    );

    // For qk*v and q*k
    void stridedBatchedGemm(
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const void *A,
        int lda,
        int64_t strideA,
        const void *B,
        int ldb,
        int64_t strideB,
        void *C,
        int ldc,
        int64_t strideC,
        int batchCount,
        float alpha,
        float beta
    );
};
