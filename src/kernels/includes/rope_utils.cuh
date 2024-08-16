#pragma once

#include <cmath>
#include <cstdio>

__device__ __forceinline__ float2 getRopeFreqCis(
    const int &zid, const int &rotary_embedding_dim,
    const float &base, const float &m
) {
    const float inv_freq = m / powf(base, static_cast<float>(zid) / rotary_embedding_dim);
    return {cosf(inv_freq), sinf(inv_freq)};
}

__device__ __forceinline__ float2 getRopeRes(
    const float &x0, const float &x1,
    const float &cos_, const float &sin_
) {
    return {x0 * cos_ - x1 * sin_, x1 * cos_ + x0 * sin_};
}