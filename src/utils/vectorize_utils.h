#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace ScalarCast2Vector {
    template<typename OutputType, typename InputType>
    __device__ __forceinline__ OutputType scalarCastToVector(const InputType val) {
        return val;
    }

    template<>
    __device__ __forceinline__ half2 scalarCastToVector<half2, float>(const float val) {
        return __float2half2_rn(val);
    }

    template<>
    __device__ __forceinline__ float4 scalarCastToVector<float4, float>(const float val) {
        return make_float4(val, val, val, val);
    }

    template<>
    __device__ __forceinline__ float2 scalarCastToVector<float2, float>(const float val) {
        return make_float2(val, val);
    }

    template<>
    __device__ __forceinline__ half2 scalarCastToVector<half2, half>(const half val) {
        half2 res;
        res.x = val;
        res.y = val;
        return res;
    }
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

template <typename T>
struct VectorizedOperator {
    __device__ __forceinline__ static void add_assign(T &dst, const T &src);
    __device__ __forceinline__ static T add(const T &x, const T &y);
    __device__ __forceinline__ static void mul_assign(T &dst, const T &src);
    __device__ __forceinline__ static T mul(const T &x, const T &y);
};

template<>
struct VectorizedOperator<float4> {
    __device__ __forceinline__ static void add_assign(float4 &dst, const float4 &src) {
        dst.x += src.x;
        dst.y += src.y;
        dst.z += src.z;
        dst.w += src.w;
    }

    __device__ __forceinline__ static float4 add(const float4 &x, const float4 &y) {
        float4 res;
        res.x = x.x + y.x;
        res.y = x.y + y.y;
        res.z = x.z + y.z;
        res.w = x.w + y.w;
        return res;
    }

    __device__ __forceinline__ static void mul_assign(float4 &dst, const float4 &src) {
        dst.x *= src.x;
        dst.y *= src.y;
        dst.z *= src.z;
        dst.w *= src.w;
    }

    __device__ __forceinline__ static float4 mul(const float4 &x, const float4 &y) {
        float4 res;
        res.x = x.x * y.x;
        res.y = x.y * y.y;
        res.z = x.z * y.z;
        res.w = x.w * y.w;
        return res;
    }
};

template<>
struct VectorizedOperator<half2> {
    __device__ __forceinline__ static void add_assign(half2 &dst, const half2 &src) {
        dst.x += src.x;
        dst.y += src.y;
    }

    __device__ __forceinline__ static half2 add(const half2 &x, const half2 &y) {
        half2 res;
        res.x = x.x + y.x;
        res.y = x.y + y.y;
        return res;
    }

    __device__ __forceinline__ static void mul_assign(half2 &dst, const half2 &src) {
        dst.x *= src.x;
        dst.y *= src.y;
    }

    __device__ __forceinline__ static half2 mul(const half2 &x, const half2 &y) {
        half2 res;
        res.x = x.x * y.x;
        res.y = x.y * y.y;
        return res;
    }
};
