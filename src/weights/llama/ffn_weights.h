#pragma once
#include "src/weights/base_weights.h"
template<typename T>
struct LlamaFFNWeights {
    BaseWeight<T> gate;
    BaseWeight<T> up;
    BaseWeight<T> down;
    BaseWeight<T> gate_and_up;
};