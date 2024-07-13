#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T_OUT, typename T_IN>
__device__ __forceinline__ T_OUT scalar_cast_vec(T_IN val) {
    return val;
}

template<>
__device__ __forceinline__ auto scalar_cast_vec<half2, float>(float val) -> half2 {
    return __float2half2_rn(val);
}

template<>
__device__ __forceinline__ auto scalar_cast_vec<float4, float>(float val) -> float4 {
    return make_float4(val, val, val, val);
}

template<>
__device__ __forceinline__ auto scalar_cast_vec<float2, float>(float val) -> float2 {
    return make_float2(val, val);
}

template<>
__device__ __forceinline__ auto scalar_cast_vec<half2, half>(half val) -> half2 {
    return half2{val, val};
}

template<typename T>
struct Vec {
    using Type = T;
    static constexpr int size = 0;
};

template<>
struct Vec<half> {
    using Type = half2; 
    static constexpr int size = 2;
};

template<>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};

struct TwoFloat2 {
    float2 x;
    float2 y;
};