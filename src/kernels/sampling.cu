#include <iostream>
#include "includes/sampling.cuh"

// mini-softmax + curand_sample
// Input: [bs, K] from topK output
// Output: [bs]
// Note: beamsearch does not involve sampling, so bsxbm = bs

/*
    dim3 grid(batch_size);
    dim3 block(K);
*/

template <typename T>
__global__ void samplingKernel(
    int *const topk_id,
    T *const topk_val, // [bs, K] from topK
    int *const output_id, // [bs]
    int *const seq_len, // cumulated seq len, [bs]
    bool *const is_finished, // [bs]
    const int K,
    const int rand_num, // step
    const int end_id, // fixed value when initializing llama model
    const int vocab_size
) {
    const int batch_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int base_offset = batch_id * K;
    const int offset = base_offset + tid;

    T max_val = topk_val[base_offset]; // max val is the top of the buffer, because topK
    topk_val[offset] = static_cast<T>(expf(static_cast<float>(topk_val[offset]) - static_cast<float>(max_val)));

    __shared__ float threshold;
    __shared__ float sum;

    if (tid == 0) {
        sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            sum += static_cast<float>(topk_val[base_offset + i]);
        }

        curandState_t state;
        curand_init(
            static_cast<unsigned long long>(rand_num),
            static_cast<unsigned long long>(batch_id),
            static_cast<unsigned long long>(0),
            &state
        );

        threshold = static_cast<float>(curand_uniform(&state)) * sum; // for a block, [0.0, sum)
        int chosen_val = topk_id[base_offset] % vocab_size;

        #pragma unroll
        for (int i = 0; i < K; ++i) {
            threshold -= static_cast<float>(topk_val[base_offset + i]);
            if (threshold < 0.0f) {
                chosen_val = topk_id[base_offset + i] % vocab_size;
                break;
            }
        }

        output_id[batch_id] = chosen_val;

        if (!is_finished[batch_id]) {
            ++seq_len[batch_id];
        }
        is_finished[batch_id] = output_id[batch_id] == end_id;
    }
}

template <typename T>
void launchSampling(
    TensorWrapper<int> *topk_id, // [bs, K]
    TensorWrapper<T> *topk_val, // [bs, K]
    TensorWrapper<int> *seq_len, // [bs]
    TensorWrapper<bool> *is_finished, // [bs]
    TensorWrapper<int> *output_id, // [bs]
    MapStringToInt *params
) {
    const int batch_size = topk_id->shape[0];
    const int K = topk_id->shape[1];
    const int vocab_size = (*params)["vocab_size"];
    const int step = (*params)["step"];
    const int end_id = (*params)["end_id"];

    dim3 grid(batch_size);
    dim3 block(K);

    samplingKernel<<<grid, block>>>(
        topk_id->data,
        topk_val->data,
        output_id->data,
        seq_len->data,
        is_finished->data,
        K,
        step,
        end_id,
        vocab_size
    );
}

// Explicit instantiation for float and half types
template void launchSampling(
    TensorWrapper<int> *topk_id,
    TensorWrapper<float> *topk_val,
    TensorWrapper<int> *seq_len,
    TensorWrapper<bool> *is_finished,
    TensorWrapper<int> *output_id,
    MapStringToInt *params
);

template void launchSampling(
    TensorWrapper<int> *topk_id,
    TensorWrapper<half> *topk_val,
    TensorWrapper<int> *seq_len,
    TensorWrapper<bool> *is_finished,
    TensorWrapper<int> *output_id,
    MapStringToInt *params
);
