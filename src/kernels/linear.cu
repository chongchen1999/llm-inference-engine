#include <iostream>
#include <fstream>
#include "src/kernels/includes/linear.h"

/*
All matmul cases:
ctx qkv linear: [num_tokens, qhiddenunits] * [qhiddenunits, hiddenunits] = {num_tokens, qkv_head_num, head_size}
ctx attn output linear: {num_tokens, head_num, head_size} * {qhiddenunits, qhiddenunits} = {num_tokens, qhiddenunits}
self qkv linear: [bs, qhiddenunits] * [qhiddenunits, hiddenunits] = {bs, qkv_head_num, head_size}
self attn output linear: [batch_size, qhiddenunits] * [qhiddenunits, qhiddenunits] = [bs, qhiddenunits]
lmhead linear: [bs, qhiddenunits] * [vocab size, qhiddenunits], need transpose B
gate: [bs/token nums, qhiddenunits] * [qhiddenunits, intersize] = [bs/token nums, intersize]
up: [bs/token nums, qhiddenunits] * [qhiddenunits, intersize] = [bs/token nums, intersize]
fusedGateUpGemm: [bs/token nums, qhiddenunits] * [qhiddenunits, 2 * intersize] = [bs/token nums, 2 * intersize]
down: [bs/token nums, intersize] * [qhiddenunits, intersize] = [bs/token nums, qhiddenunits]
*/

// Note: cuBLAS is column-major.

// Compute y = x * AT, since cuBLAS is col-major, we actually compute yT = A * xT
template <typename T>
void launchLinearGemm(
    TensorWrapper<T> *input,
    BaseWeight<T> *weight,
    TensorWrapper<T> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
) {
    const int Am = input->shape[0];
    const int An = input->shape[1];
    const int Bm = weight->shape[0];
    const int Bn = weight->shape[1];
    const int Cm = output->shape[0];
    const int Cn = output->shape[1];

    // For ctx attn and self attn qkv linear, assume [bs/token nums, qkv head num, head size]
    // For gate & up linear, assume weight.shape=[hidden, 2*intersize], output.shape=[bs, 2, intersize]
    // For ctx attn output linear
    if (input->shape.size() == 3) {
        An = input->shape[1] * input->shape[2];
    }

    if (output->shape.size() == 3) {
        Cn = output->shape[1] * output->shape[2];
    }

    const int opAm = trans_a ? An : Am;
    const int opAn = trans_a ? Am : An;
    const int opBm = trans_b ? Bn : Bm;
    const int opBn = trans_b ? Bm : Bn;

    LLM_CHECK_WITH_INFO(opAn == opBm, "2nd dim of weight MUST = 1st dim of input");
    LLM_CHECK_WITH_INFO(opAm == Cm && opBn == Cn, "output shape should be equal to weight shape");

    // We need to compute C = opA * opB, actually compute: CT = (opB)T * (opA)T
    const int lda = opAn; // leading dim for (opA)T in col-major
    const int ldb = opBn;
    const int ldc = Cn;

    // Two transposes make no transpose
    const auto transA = trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto transB = trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;

    // We need C = opA * opB, 
    // feed CT = opBT * opAT to cuBLAS,
    // cuBLAS returns CT in col-major, that is C in row-major
    cublas_wrapper->gemm(
        transA,
        transB,
        Cn,
        Cm,
        opAn,
        weight->data,
        ldb,
        input->data,
        lda,
        output->data,
        ldc,
        1.0f,
        0.0f
    );

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#else
#endif
}

template <typename T>
void launchLinearStridedBatchGemm(
    TensorWrapper<T> *input1,
    TensorWrapper<T> *input2,
    TensorWrapper<T> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
) {
    // B.T A.T = C.T
    // input1 and input2 shape: [bs, head_num, seqlen, head_size]
    const int Am = input1->shape[2];
    const int An = input1->shape[3];
    const int Bm = input2->shape[2];
    const int Bn = input2->shape[3];
    const int Cm = output->shape[2];
    const int Cn = output->shape[3];

    const int opAm = trans_a ? An : Am;
    const int opAn = trans_a ? Am : An;
    const int opBm = trans_b ? Bn : Bm;
    const int opBn = trans_b ? Bm : Bn;

    LLM_CHECK_WITH_INFO(opAn == opBm, "2nd dim of weight MUST = 1st dim of input");
    LLM_CHECK_WITH_INFO(opAm == Cm && opBn == Cn, "output shape should be equal to weight shape");

    // We need to compute C = opA * opB, actually compute: CT = (opB)T * (opA)T
    const int lda = opAn; // leading dim for (opA)T in col-major
    const int ldb = opBn;
    const int ldc = Cn;

    const int64_t strideA = opAm * opAn;
    const int64_t strideB = opBm * opBn;
    const int64_t strideC = Cm * Cn;

    // TODO: check 4th dim of input = 3rd dim of weight
    // TODO: check batchCount of two matrices is equal
    const int batchCount = input1->shape[0] * input1->shape[1];

    // Two transposes make no transpose
    const auto transA = trans_a ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto transB = trans_b ? CUBLAS_OP_N : CUBLAS_OP_T;

    cublas_wrapper->stridedBatchedGemm(
        transA,
        transB,
        Cn,           // m
        Cm,           // n
        opAn,         // k
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
        0.0f
    );

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#else
#endif
}

// Explicit template instantiation
template void launchLinearGemm(
    TensorWrapper<float> *input,
    BaseWeight<float> *weight,
    TensorWrapper<float> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
);

template void launchLinearGemm(
    TensorWrapper<half> *input,
    BaseWeight<half> *weight,
    TensorWrapper<half> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
);

template void launchLinearStridedBatchGemm(
    TensorWrapper<float> *input1,
    TensorWrapper<float> *input2,
    TensorWrapper<float> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
);

template void launchLinearStridedBatchGemm(
    TensorWrapper<half> *input1,
    TensorWrapper<half> *input2,
    TensorWrapper<half> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
);
