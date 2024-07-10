#pragma once

#include <cuda_fp16.h>
#include <vector>
#include <cstdint>

enum class WeightType {
    FP32_W,
    FP16_W,
    INT8_W,
    UNSUPPORTED_W
};

template <typename T>
inline WeightType getWeightType() {
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return WeightType::FP32_W;
    } else if (std::is_same<T, __half>::value || std::is_same<T, const __half>::value) {
        return WeightType::FP16_W;
    } else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return WeightType::INT8_W;
    } else {
        return WeightType::UNSUPPORTED_W;
    }
}

template<typename T>
class BaseWeight {
public:
    WeightType type;
    std::vector<int> shape;
    T* data;
    T* bias;
};