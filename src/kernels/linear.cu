#include <iostream>
#include <fstream>
#include <algorithm>
#include "includes/linear.h"
#include "../utils/output_utils.h"

/*
All matrix multiplication cases:
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
    CublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
) {
    std::cout << "launchLinearGemm!" << std::endl << std::endl;

    print_tensor<T>(input);
    print_weight<T>(weight);
    print_tensor<T>(output);

    std::cout << "trans_a: " << trans_a << std::endl;
    std::cout << "trans_b: " << trans_b << std::endl;

    int Am = input->shape[0];
    int An = input->shape[1];
    int Bm = weight->shape[0];
    int Bn = weight->shape[1];
    int Cm = output->shape[0];
    int Cn = output->shape[1];

    // For ctx attn and self attn qkv linear, assume [bs/token nums, qkv head num, head size]
    // For gate & up linear, assume weight.shape=[hidden, 2*intersize], output.shape=[bs, 2, intersize]
    // For ctx attn output linear
    if (input->shape.size() == 3) {
        An = input->shape[1] * input->shape[2];
    }

    if (output->shape.size() == 3) {
        Cn = output->shape[1] * output->shape[2];
    }

    std::cout << "ready for cublas0!" << std::endl;

    int opAm = Am;
    int opAn = An;
    int opBm = Bm;
    int opBn = Bn;
    if (trans_a) {
        std::swap(opAm, opAn);
    }
    if (trans_b) {
        std::swap(opBm, opBn);
    }

    int lda = opAn; //col-major + transpose
    int ldb = opBn; //col-major + transpose
    int ldc = Cn; //col-major

    LLM_CHECK_WITH_INFO(opAn == opBm, "2nd dim of weight MUST = 1st dim of input");
    LLM_CHECK_WITH_INFO(opAm == Cm && opBn == Cn, "output shape should be equal to weight shape");

    std::cout << "ready for cublas!" << std::endl;

    cublas_wrapper->gemm(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
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
    CublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
) {
    // B.T A.T = C.T
    // input1 and input2 shape: [bs, head_num, seqlen, head_size]
    int Am = input1->shape[2];
    int An = input1->shape[3];
    int Bm = input2->shape[2];
    int Bn = input2->shape[3];
    int Cm = output->shape[2];
    int Cn = output->shape[3];

    int opAm = Am;
    int opAn = An;
    int opBm = Bm;
    int opBn = Bn;
    if (trans_a) {
        std::swap(opAm, opAn);
    }
    if (trans_b) {
        std::swap(opBm, opBn);
    }

    int lda = opAn; //col-major + transpose
    int ldb = opBn; //col-major + transpose
    int ldc = Cn; //col-major

    LLM_CHECK_WITH_INFO(opAn == opBm, "2nd dim of weight MUST = 1st dim of input");
    LLM_CHECK_WITH_INFO(opAm == Cm && opBn == Cn, "output shape should be equal to weight shape");

    long long int strideA = opAm * opAn;
    long long int strideB = opBm * opBn;
    long long int strideC = Cm * Cn;

    // TODO: check 4th dim of input = 3rd dim of weight
    // TODO: check batch_count of two matrices is equal
    int batch_count = input1->shape[0] * input1->shape[1];
    LLM_CHECK_WITH_INFO(batch_count == input2->shape[0] * input2->shape[1], "dim 0 and dim 1 wrong!");

    cublas_wrapper->stridedBatchedGemm(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
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
        batch_count,
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
    CublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
);

template void launchLinearGemm(
    TensorWrapper<half> *input,
    BaseWeight<half> *weight,
    TensorWrapper<half> *output,
    CublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
);

template void launchLinearStridedBatchGemm(
    TensorWrapper<float> *input1,
    TensorWrapper<float> *input2,
    TensorWrapper<float> *output,
    CublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
);

template void launchLinearStridedBatchGemm(
    TensorWrapper<half> *input1,
    TensorWrapper<half> *input2,
    TensorWrapper<half> *output,
    CublasWrapper *cublas_wrapper,
    bool trans_a,
    bool trans_b
);
