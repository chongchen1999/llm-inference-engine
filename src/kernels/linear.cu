#include <iostream>
#include <fstream>
#include "src/kernels/includes/linear.h"

/*
All matmul cases:
ctx qkv linear: [num_tokens, qhiddenunits] * [qhiddenunits, hiddenunits] = {num_tokens, qkv_head_num, head_size}
ctx attn output linear: {num_tokens, head_num, head_size} * {qhiddenunits, qhiddenunits} = {num_tokens, qhiddenunits}
self qkv linear: [bs, qhiddenunits] * [qhiddenunits, hiddenunits] = {bs, qkv_head_num, head_size}
self attn output linear: {batch_size, qhiddenunits} * [qhiddenunits, qhiddenunits] = [bs, qhiddenunits]
lmhead linear: [bs, qhiddenunits] * [vocab size, qhiddenunits], need transpose B
gate: [bs/token nums, qhiddenunits] * [qhiddenunits, intersize] = [bs/token nums, intersize]
up: [bs/token nums, qhiddenunits] * [qhiddenunits, intersize] = [bs/token nums, intersize]
fusedGateUpGemm: [bs/token nums, qhiddenunits] * [qhiddenunits, 2 * intersize] = [bs/token nums, 2 * intersize]
down: [bs/token nums, intersize] * [qhiddenunits, intersize] = [bs/token nums, qhiddenunits]
*/

//Note: cuBLAS is column-major.

// compute y = x * AT, since cublas is col-major, we actually compute yT = A * xT
template <typename T>
void launchLinearGemm(TensorWrapper<T> *input,
                      BaseWeight<T> &weight,
                      TensorWrapper<T> *output,
                      cublasWrapper *cublas_wrapper,
                      bool trans_a, bool trans_b) {
    int Am = input->shape[0];
    int An = input->shape[1];
    int Bm = weight.shape[0];
    int Bn = weight.shape[1];
    int Cm = output->shape[0];
    int Cn = output->shape[1];
    // printf("shape A: %d %d\n", Am, An);
    // printf("shape B: %d %d\n", Bm, Bn);
    // printf("shape C: %d %d\n", Cm, Cn);
    // puts("");
    // for ctx attn and self attn qkv linear, assume [bs/token nums, qkv head num, head size]
    // for gate & up linear, assume weight.shape=[hidden, 2*intersize], output.shape=[bs, 2, intersize]
    // for ctx attn output linear
    if (input->shape.size() == 3) {
        An = input->shape[1] * input->shape[2];
    }
    if (output->shape.size() == 3) {
        Cn = output->shape[1] * output->shape[2];
    }

    int opAm = trans_a ? An : Am;
    int opAn = trans_a ? Am : An;
    int opBm = trans_b ? Bn : Bm;
    int opBn = trans_b ? Bm : Bn;

    LLM_CHECK_WITH_INFO(opAn == opBm, "2nd dim of weight MUST = 1st dim of input");
    LLM_CHECK_WITH_INFO(opAm == Cm && opBn == Cn, "output shape should be equal to weight shape");

    // we neend compute C = opA * opB, actually compute: CT = (opB)T * (opA)T
    int lda = opAn; // leading dim for (opA)T in col-major
    int ldb = opBn;
    int ldc = Cn;

    // Two transpose make no transpose
    auto transA = trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto transB = trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;

    // we need C = opA * opB, 
    // feed CT = opBT * opAT to cublas,
    // cublas retuen CT in col-major, that is C in row-major
    cublas_wrapper->gemm(transA,
                         transB,
                         Cn,
                         Cm,               
                         opAn,
                         weight.data,
                         ldb,         
                         input->data,  
                         lda,          
                         output->data, 
                         ldc,        
                         1.0f,
                         0.0f);

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#else
#endif
}

template <typename T>
void launchLinearStridedBatchGemm(TensorWrapper<T> *input1,
                                  TensorWrapper<T> *input2,
                                  TensorWrapper<T> *output,
                                  cublasWrapper *cublas_wrapper,
                                  bool trans_a, bool trans_b) {
    // B.T A.T = C.T
    // input1 and input2 shape: [bs, head_num, seqlen, head_size]
    int Am = input1->shape[2];
    int An = input1->shape[3];
    int Bm = input2->shape[2];
    int Bn = input2->shape[3];
    int Cm = output->shape[2];
    int Cn = output->shape[3];

    int opAm = trans_a ? An : Am;
    int opAn = trans_a ? Am : An;
    int opBm = trans_b ? Bn : Bm;
    int opBn = trans_b ? Bm : Bn;

    LLM_CHECK_WITH_INFO(opAn == opBm, "2nd dim of weight MUST = 1st dim of input");
    LLM_CHECK_WITH_INFO(opAm == Cm && opBn == Cn, "output shape should be equal to weight shape");

    // we neend compute C = opA * opB, actually compute: CT = (opB)T * (opA)T
    int lda = opAn; // leading dim for (opA)T in col-major
    int ldb = opBn;
    int ldc = Cn;

    int64_t strideA = opAm * opAn;
    int64_t strideB = opBm * opBn;
    int64_t strideC = Cm * Cn;

    // TODO: check 4th dim of input = 3rd dim of weight
    // TODO: check batchCount of two matrix is equal
    int batchCount = input1->shape[0] * input1->shape[1];

    // Two transpose make no transpose
    auto transA = trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto transB = trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;

    cublas_wrapper->stridedBatchedGemm(transA,
                                       transB,
                                       Cn,           // m
                                       Cm,           // n
                                       opAn,           // k
                                       input2->data,
                                       ldb,
                                       strideB,
                                       input1->data,
                                       lda,
                                       strideA,
                                       output->data,
                                       ldc,
                                       strideC,
                                       batchCount,
                                       1.0f,
                                       0.0f);

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#else
#endif
}

template void launchLinearGemm(TensorWrapper<float> *input,
                               BaseWeight<float> &weight,
                               TensorWrapper<float> *output,
                               cublasWrapper *cublas_wrapper,
                               bool trans_a, bool trans_b);

template void launchLinearGemm(TensorWrapper<half> *input,
                               BaseWeight<half> &weight,
                               TensorWrapper<half> *output,
                               cublasWrapper *cublas_wrapper,
                               bool trans_a, bool trans_b);

template void launchLinearStridedBatchGemm(TensorWrapper<float> *input1,
                                           TensorWrapper<float> *input2,
                                           TensorWrapper<float> *output,
                                           cublasWrapper *cublas_wrapper,
                                           bool trans_a, bool trans_b);

template void launchLinearStridedBatchGemm(TensorWrapper<half> *input1,
                                           TensorWrapper<half> *input2,
                                           TensorWrapper<half> *output,
                                           cublasWrapper *cublas_wrapper,
                                           bool trans_a, bool trans_b);
