#pragma once

#include "../base_weights.h"

template<typename T>
struct LlamaAttentionWeights {
    BaseWeight<T> q;
    BaseWeight<T> k;
    BaseWeight<T> v;
    BaseWeight<T> qkv;
    BaseWeight<T> output;
};