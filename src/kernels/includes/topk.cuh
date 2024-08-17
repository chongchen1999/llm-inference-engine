#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../../utils/tensor.h"

template<typename T, int beam_size>
struct TopK {
    T val[beam_size];
    int id[beam_size];

    __device__ void init() {
        for (int i = 0; i < beam_size; ++i) {
            id[i] = -1;
            val[i] = -1e-20;
        }
    }

    template <typename U>
    __device__ __forceinline__ void swap(U &x, U &y) {
        U temp = x;
        x = y;
        y = temp;
    }

    __device__ void insertQueue(const T &data, const int &data_id) {
        const float v = static_cast<float>(val[beam_size - 1]);
        if (id[beam_size - 1] == -1 || v < static_cast<float>(data)) {
            id[beam_size - 1] = data_id;
            val[beam_size - 1] = data;
        }

        for (int i = beam_size - 2; i >= 0; --i) {
            if (id[i] == -1 || val[i + 1] > val[i]) {
                swap<T>(val[i + 1], val[i]);
                swap<int>(id[i + 1], id[i]);
            }
        }
    }
};

template <typename T>
void launchTopKForBeamSearch(
    TensorWrapper<T> *probs,
    TensorWrapper<int> *topk_ids,
    TensorWrapper<T> *topk_vals,
    TensorWrapper<int> *final_topk_ids,
    TensorWrapper<T> *final_topk_vals
);
