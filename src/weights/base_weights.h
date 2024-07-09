#pragma once
#include <vector>
#include <cstdint>
#include <cuda_fp16.h>

enum class WeightType {
    FP32_W,
    FP16_W,
    INT8_W,
    UNSUPPORTED_W
};

template <typename T>
constexpr WeightType getWeightType() {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, const float>) {
        return WeightType::FP32_W;
    } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, const half>) {
        return WeightType::FP16_W;
    } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, const int8_t>) {
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