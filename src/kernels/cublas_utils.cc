#include "src/kernels/cublas_utils.h"
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

// FP32 GEMM and FP16 GEMM
// from row-major(C++ style) to col-major(cuBLAS style)
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


    int opAm = transa ? k : m;
    int opAn = transa ? m : k;
    int opBm = transb ? n : k;
    int opBn = transb ? k : n;
    int Cm = transa ? m : opAm;
    int Cn = transb ? n : opBn;

    CHECK_CUBLAS(cublasGemmEx(cublas_handle_,
                              transa,
                              transb,
                              Cm,
                              Cn,
                              opAn,
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

    int opAm = transa ? k : m;
    int opAn = transa ? m : k;
    int opBm = transb ? n : k;
    int opBn = transb ? k : n;
    int Cm = transa ? m : opAm;
    int Cn = transb ? n : opBn;
    
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(cublas_handle_,
                                            transa,
                                            transb,
                                            Cm,
                                            Cn,
                                            opAn,
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