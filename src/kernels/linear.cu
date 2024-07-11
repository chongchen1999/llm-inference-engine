#include <iostream>
#include <fstream>
#include "src/kernels/linear.h"

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

    // col-major
    int lda = Am;
    int ldb = Bm;
    int ldc = Cm;

    // for lmhead linear and ffn all linears
    auto transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    // row-major to cublas_wrapper, and trans to col-major when feed to cublas
    cublas_wrapper->gemm(transA,
                         transB,
                         Cm, // m
                         Cn,                // n, when load real weight, lmhead weight is same as pre embedding, which shape = [vocab, hidden], so here should transpose b
                         An,
                         weight.data,  // A, cur_input_len is for context decoder lmhead
                         lda,          // lda
                         input->data,  // B
                         ldb,          // ldb
                         output->data, // C
                         ldc,          // ldc
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

    // col-major
    int lda = Am;
    int ldb = Bm;
    int ldc = Cm;

    int64_t strideA = opAm * opAn;
    int64_t strideB = opBm * opBn;
    int64_t strideC = Cm * Cn;

    // TODO: check 4th dim of input = 3rd dim of weight
    // TODO: check batchCount of two matrix is equal
    int batchCount = input1->shape[0] * input1->shape[1];

    auto transA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto transB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublas_wrapper->stridedBatchedGemm(transA,
                                       transB,
                                       Cm,           // m
                                       Cn,           // n
                                       An,           // k
                                       input1->data,
                                       lda,
                                       strideA,
                                       input2->data,
                                       ldb,
                                       strideB,
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
