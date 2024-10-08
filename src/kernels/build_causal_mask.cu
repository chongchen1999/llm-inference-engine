#include "includes/build_causal_mask.cuh"

// mask shape = [bs, max_q_len, max_k_len]
template <typename T>
__global__ void buildCausalMasks(
    T *mask,                  // [bs, max_q_len, max_k_len]
    const int *q_lens,        // Input lengths, shape=[batch size]
    const int *k_lens,        // Context lengths, shape=[batch size]
    const int max_q_len, 
    const int max_k_len
) {
    const int tid = threadIdx.x;
    const int q_len = q_lens[blockIdx.x];
    const int k_len = k_lens[blockIdx.x];
    mask += blockIdx.x * max_q_len * max_k_len;

    for (int offset = tid; offset < max_q_len * max_k_len; offset += blockDim.x) {
        const int q = offset / max_k_len;
        const int k = offset - q * max_k_len; // aka offset % max_k_len;
        bool is_one = (q < q_len) && (k < k_len) && (k <=  q + (k_len - q_len));
        mask[offset] = static_cast<T>(is_one);
    }
}

template <typename T>
void launchBuildCausalMasks(
    TensorWrapper<T> *mask, 
    TensorWrapper<int> *q_lens, 
    TensorWrapper<int> *k_lens
) {
    const int batch_size = mask->shape[0];
    const int max_q_len = mask->shape[1];
    const int max_k_len = mask->shape[2];

    buildCausalMasks<T><<<batch_size, 256>>>(
        mask->data, 
        q_lens->data, 
        k_lens->data, 
        max_q_len, 
        max_k_len
    );
}

template void launchBuildCausalMasks(
    TensorWrapper<float> *mask, 
    TensorWrapper<int> *q_lens, 
    TensorWrapper<int> *k_lens
);

template void launchBuildCausalMasks(
    TensorWrapper<half> *mask, 
    TensorWrapper<int> *q_lens, 
    TensorWrapper<int> *k_lens
);
