#pragma once

#include "tensor.h"
#include <cstring>
#include <vector>
#include "../weights/base_weights.h"

void print_tensor(Tensor *tensor) {
    printf("number of dimensions: %d\n", tensor->shape.size());
    for (auto &dim : tensor->shape) {
        printf("%d ", dim);
    }
    printf("\n");
}

template <typename T>
void print_tensor(TensorWrapper<T> *tensor) {
    printf("number of dimensions: %d\n", tensor->shape.size());
    for (auto &dim : tensor->shape) {
        printf("%d ", dim);
    }
    printf("\n");
}

template <typename T>
void print_weight(BaseWeight<T> *weight) {
    printf("number of dimensions: %d\n", weight->shape.size());
    for (auto &dim : weight->shape) {
        printf("%d ", dim);
    }
    printf("\n");
}