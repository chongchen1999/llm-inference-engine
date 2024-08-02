#pragma once

#include <iostream>
#include "tensor.h"
#include <cstring>
#include <vector>
#include "../weights/includes/base_weights.h"

void print_tensor(const Tensor *tensor) {
    std::cout << "number of dimensions: " << tensor->shape.size() << std::endl;
    for (const auto &dim : tensor->shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void print_tensor(const TensorWrapper<T> *tensor) {
    std::cout << "number of dimensions: " << tensor->shape.size() << std::endl;
    for (const auto &dim : tensor->shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void print_weight(const BaseWeight<T> *weight) {
    std::cout << "number of dimensions: " << weight->shape.size() << std::endl;
    for (const auto &dim : weight->shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
}
