#include <iostream>
#include "../../src/kernels/includes/sampling.h"
#include "../../src/utils/macro.h"

// Note: There is no CPU implementation of this kernel
// We compare the kernel correctness by eye and result print info
// Use `./test_sampling` to test fp32 GPU kernel
// Use `./test_sampling 1` to test fp16 GPU kernel

template <typename T>
void launch_sampling(
    int batch_size,
    int K,
    int step,
    int vocab_size,
    int end_id
) {
    int *h_topkid = static_cast<int *>(malloc(sizeof(int) * batch_size * K));
    int *d_topkid;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_topkid), sizeof(int) * batch_size * K));

    T *h_topkval = static_cast<T *>(malloc(sizeof(T) * batch_size * K));
    T *d_topkval;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_topkval), sizeof(T) * batch_size * K));

    int *h_outid = static_cast<int *>(malloc(sizeof(int) * batch_size));
    int *d_outid;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_outid), sizeof(int) * batch_size));

    int *h_cuseqlen = static_cast<int *>(malloc(sizeof(int) * batch_size));
    int *d_cuseqlen;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_cuseqlen), sizeof(int) * batch_size));

    bool *h_finished = static_cast<bool *>(malloc(sizeof(bool) * batch_size));
    bool *d_finished;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_finished), sizeof(bool) * batch_size));

    for (int i = 0; i < batch_size; ++i) {
        h_finished[i] = false;
        h_cuseqlen[i] = 4;
    }

    for (int i = 0; i < batch_size * K; ++i) {
        h_topkid[i] = i;
        h_topkval[i] = static_cast<T>(K - 1 - (i % K));
    }

    CHECK(cudaMemcpy(d_topkval, h_topkval, sizeof(T) * batch_size * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_topkid, h_topkid, sizeof(int) * batch_size * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cuseqlen, h_cuseqlen, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_finished, h_finished, sizeof(bool) * batch_size, cudaMemcpyHostToDevice));

    const DataType type = getTensorType<T>();
    auto *topk_val = new TensorWrapper<T>(Device::GPU, type, {batch_size, K}, d_topkval);

    const DataType type_int = getTensorType<int>();
    const DataType type_bool = getTensorType<bool>();

    auto *topk_id = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, K}, d_topkid);
    auto *cuseqlen = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_cuseqlen);
    auto *finished = new TensorWrapper<bool>(Device::GPU, type_bool, {batch_size}, d_finished);
    auto *output_id = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_outid);

    MapStringToInt intParams;
    intParams.insert({"step", step});
    intParams.insert({"vocab_size", vocab_size});
    intParams.insert({"end_id", end_id});

    std::cout << "Before launching sampling kernel" << std::endl;
    launchSampling<T>(topk_id, topk_val, cuseqlen, finished, output_id, &intParams);
    std::cout << "After launching sampling kernel" << std::endl;

    std::cout << "Copying data from device to host" << std::endl;
    CHECK(cudaMemcpy(h_outid, output_id->data, sizeof(int) * batch_size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < batch_size; ++i) {
        std::cout << "Sequence " << i + 1 << ": " << h_outid[i] << std::endl;
    }

    free(h_topkid);
    free(h_topkval);
    free(h_cuseqlen);
    free(h_finished);
    free(h_outid);

    CHECK(cudaFree(d_topkid));
    CHECK(cudaFree(d_topkval));
    CHECK(cudaFree(d_cuseqlen));
    CHECK(cudaFree(d_finished));
    CHECK(cudaFree(d_outid));

    delete topk_val;
    delete topk_id;
    delete cuseqlen;
    delete finished;
    delete output_id;
}

int main(int argc, char *argv[]) {
    constexpr int batch_size = 3;
    constexpr int K = 3;
    constexpr int vocab_size = 1000;
    constexpr int step = 6;
    constexpr int end_id = 10;

    if (argc > 1) {
        launch_sampling<half>(batch_size, K, step, vocab_size, end_id);
    } else {
        launch_sampling<float>(batch_size, K, step, vocab_size, end_id);
    }

    return 0;
}
