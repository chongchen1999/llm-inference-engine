#include "src/kernels/includes/cublas_utils.h"
#include <iostream>

cublasWrapper::cublasWrapper(
    cublasHandle_t cublas_handle,
    cublasLtHandle_t cublaslt_handle
) :
    cublas_handle_(cublas_handle),
    cublaslt_handle_(cublaslt_handle) {}

cublasWrapper::~cublasWrapper() {}

void cublasWrapper::setFP32GemmConfig() {
    Atype_ = CUDA_R_32F;
    Btype_ = CUDA_R_32F;
    Ctype_ = CUDA_R_32F;
    computeType_ = CUDA_R_32F;
}

void cublasWrapper::setFP16GemmConfig() {
    Atype_ = CUDA_R_16F;
    Btype_ = CUDA_R_16F;
    Ctype_ = CUDA_R_16F;
    computeType_ = CUDA_R_32F;
}

void cublasWrapper::gemm(
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
    float alpha = 1.0f,
    float beta = 0.0f
) {
    bool is_fp16_computeType = (computeType_ == CUDA_R_16F);
    
    half alpha_half = static_cast<half>(alpha);
    half beta_half = static_cast<half>(beta);
    
    const void *alpha_ptr = is_fp16_computeType
        ? static_cast<const void*>(&alpha_half)
        : static_cast<const void*>(&alpha);
        
    const void *beta_ptr = is_fp16_computeType
        ? static_cast<const void*>(&beta_half)
        : static_cast<const void*>(&beta);

    CHECK_CUBLAS(
        cublasGemmEx(
            cublas_handle_,
            transa,
            transb,
            m,
            n,
            k,
            alpha_ptr,
            A,
            Atype_,
            lda,
            B,
            Btype_,
            ldb,
            beta_ptr,
            C,
            Ctype_,
            ldc,
            computeType_,
            CUBLAS_GEMM_DEFAULT
        )
    );
}

void cublasWrapper::stridedBatchedGemm(
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
    float alpha = 1.0f,
    float beta = 0.0f
) {
    bool is_fp16_computeType = (computeType_ == CUDA_R_16F);
    
    half alpha_half = static_cast<half>(alpha);
    half beta_half = static_cast<half>(beta);
    
    const void *alpha_ptr = is_fp16_computeType
        ? static_cast<const void*>(&alpha_half)
        : static_cast<const void*>(&alpha);
        
    const void *beta_ptr = is_fp16_computeType
        ? static_cast<const void*>(&beta_half)
        : static_cast<const void*>(&beta);

    CHECK_CUBLAS(
        cublasGemmStridedBatchedEx(
            cublas_handle_,
            transa,
            transb,
            m,
            n,
            k,
            alpha_ptr,
            A,
            Atype_,
            lda,
            strideA,
            B,
            Btype_,
            ldb,
            strideB,
            beta_ptr,
            C,
            Ctype_,
            ldc,
            strideC,
            batchCount,
            computeType_,
            CUBLAS_GEMM_DEFAULT
        )
    );
}
