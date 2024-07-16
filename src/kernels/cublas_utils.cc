#include "src/kernels/includes/cublas_utils.h"
#include <iostream>

// notes: cublas gemm and strided batch gemm function calls are fairly fixed
cublasWrapper::cublasWrapper(cublasHandle_t cublas_handle,
                             cublasLtHandle_t cublaslt_handle)
    : cublas_handle_(cublas_handle), cublaslt_handle_(cublaslt_handle) {}

cublasWrapper::~cublasWrapper() {}

// Invoked in model example main function after initializing cublas wrapper
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

// FP32 GEMM and FP16 GEMM,
void cublasWrapper::gemm(cublasOperation_t transa,
                         cublasOperation_t transb,
                         const int         m,
                         const int         n,
                         const int         k,
                         const void       *A,
                         const int         lda,
                         const void       *B,
                         const int         ldb,
                         void             *C,
                         const int         ldc,
                         float             alpha = 1.0f,
                         float             beta = 0.0f) {
    bool is_fp16_computeType = computeType_ == CUDA_R_16F;
    half alpha_half = static_cast<half>(alpha);
    half beta_half = static_cast<half>(beta);
    const void *alpha_ptr = is_fp16_computeType ? reinterpret_cast<const void *>(&alpha_half) 
                                                : reinterpret_cast<const void *>(&alpha);
    const void *beta_ptr = is_fp16_computeType ? reinterpret_cast<const void *>(&beta_half) 
                                               : reinterpret_cast<const void *>(&beta);

    /*printf("transa: %d, transb: %d, m: %d, n: %d, k: %d\n", transa, transb, m, n, k);
    printf("lda: %d, ldb: %d, ldc: %d\n", lda, ldb, ldc);
    printf("alpha: %f, beta: %f\n", alpha, beta);*/

    // printf("below is 2d matmul:\n");
    // print_gpu_mat(reinterpret_cast<float *>(const_cast<void *>(A)), m, k);
    // print_gpu_mat(reinterpret_cast<float *>(const_cast<void *>(B)), k, n);
    // print_gpu_mat(reinterpret_cast<float *>(const_cast<void *>(C)), m, n);

    CHECK_CUBLAS(cublasGemmEx(cublas_handle_,
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
                              CUBLAS_GEMM_DEFAULT));
    
    // printf("after matmul:\n");
    // print_gpu_mat(reinterpret_cast<float *>(const_cast<void *>(C)), m, n);
}

void cublasWrapper::stridedBatchedGemm(cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       const int         m,
                                       const int         n,
                                       const int         k,
                                       const void       *A,
                                       const int         lda,
                                       const int64_t     strideA,
                                       const void       *B,
                                       const int         ldb,
                                       const int64_t     strideB,
                                       void             *C,
                                       const int         ldc,
                                       const int64_t     strideC,
                                       const int         batchCount,
                                       float             alpha = 1.0f,
                                       float             beta = 0.0f) {
    bool is_fp16_computeType = computeType_ == CUDA_R_16F;
    half alpha_half = static_cast<half>(alpha);
    half beta_half = static_cast<half>(beta);
    const void *alpha_ptr = is_fp16_computeType ? reinterpret_cast<const void *>(&alpha_half) 
                                                : reinterpret_cast<const void *>(&alpha);
    const void *beta_ptr = is_fp16_computeType ? reinterpret_cast<const void *>(&beta_half) 
                                               : reinterpret_cast<const void *>(&beta);
    
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(cublas_handle_,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            alpha_ptr,
                                            A,
                                            Atype_,
                                            lda,
                                            strideA, // stride of batch dim, ex. A = [1, 2, 3, 4], then strideA = 3 * 4
                                            B,
                                            Btype_,
                                            ldb,
                                            strideB,
                                            beta_ptr,
                                            C,
                                            Ctype_,
                                            ldc,
                                            strideC,
                                            batchCount, // ex. A = [1, 2, 3, 4], then batchCount = 1 * 2
                                            computeType_,
                                            CUBLAS_GEMM_DEFAULT));
}