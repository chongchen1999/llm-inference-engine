#include <stdio.h>
#include "src/kernels/rmsnorm.h"
//bugs1: 2nd warpreducesum returns 0, because blockDim.x < 32, blockDim.x / 32=0
//bugs2: output buffer valuse is the same as ones before call, thats because we didn't successfully write into the output address
//bugs3: output buffer's 1st 32 values are right, the latter is wrong, because when we use vec, the ele nums of a row is hiddenunits/vecsize, we should note the row stride to move the ptr carefully
//bugs4: remeber add __syncthreads() in fp32/fp16 kernel, or we cant get the right res, ep, here we didnt add it, we get some res equal to 0 
template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for(int i = 16; i > 0; i >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}
//note:!!!when blocksize < 32, use blockDim.x/32 to get warp nums is wrong, we should instead ceil it
template<typename T>
__device__ T blockReduceSum(T val) {
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane_id = tid % 32;
    const int warp_num = (blockDim.x + 31) / 32;
    static __shared__ T warp_sum[32];
    val = warpReduceSum<T>(val);
    if (lane_id == 0) {
        warp_sum[wid] = val;
    }
    __syncthreads();

    T sum = tid < warp_num ? warp_sum[tid] : 0;
    sum = warpReduceSum<T>(sum); //though 0th own the sum, but dont need to shfl sync
    return sum;
}
// 1.this kernel is used at the begin of every decoder layer and the end of 32 decoder layers
// 2.I allocate threads number by assuming head size can be divided by 4 and 2
template <typename T>
__global__ void RMSNorm(T *decoder_out, // [num tokens, q_hidden_units]
                        T *decoder_residual, // [num tokens, q_hidden_units]
                        T *weights, // [hidden_units]
                        float eps, int num_tokens, int hidden_units) {
    const int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    float thread_sum = 0.0f;
    auto vec_dout = reinterpret_cast<Vec_t *>(decoder_out + blockIdx.x * hidden_units);
    auto vec_rsd = reinterpret_cast<Vec_t *>(decoder_residual + blockIdx.x * hidden_units);

    #pragma unroll
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t vec = vec_dout[idx];
        vec_rsd[idx] = vec;
        thread_sum += vec.x * vec.x;
        thread_sum += vec.y * vec.y;
        thread_sum += vec.z * vec.z;
        thread_sum += vec.w * vec.w;
    }

    thread_sum = blockReduceSum<float>(thread_sum);
    __shared__ float inv_mean;
    if (threadIdx.x == 0) {
        inv_mean = rsqrtf(thread_sum / hidden_units + eps);
    }
    __syncthreads();
    
    auto vec_weights = reinterpret_cast<Vec_t *>(weights);

    #pragma unroll
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t cur_vec_dout = vec_dout[idx];// note the offset should divide vec size
        Vec_t cur_vec_weight = vec_weights[idx];
        Vec_t normed_vec_dout = {
            cur_vec_dout.x * inv_mean * cur_vec_weight.x,
            cur_vec_dout.y * inv_mean * cur_vec_weight.y,
            cur_vec_dout.z * inv_mean * cur_vec_weight.z,
            cur_vec_dout.w * inv_mean * cur_vec_weight.w
        };
        vec_dout[idx] = normed_vec_dout;
    }
}

template<>
__global__ void RMSNorm(half *decoder_out, // [num tokens, q_hidden_units]
                        half *decoder_residual, // [num tokens, q_hidden_units]
                        half *weights, //[hidden_units]
                        float eps, int num_tokens, int hidden_units) {
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    auto vec_dout = reinterpret_cast<Vec_t *>(decoder_out + blockDim.x * hidden_units);
    Vec_t *rsd;
    if (decoder_residual != nullptr) {
        rsd = reinterpret_cast<Vec_t*>(decoder_residual + blockDim.x * hidden_units);
    }
    float thread_sum = 0.0f;

    #pragma unroll
    for(int i = threadIdx.x; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t out = vec_dout[i];// note the offset should divide vec size
        if (decoder_residual != nullptr) {
            rsd[i] = out;
        }
        thread_sum += __half2float(out.x) * __half2float(out.x);
        thread_sum += __half2float(out.y) * __half2float(out.y);
    } //x^2
    
    // mean(x^2)
    float blocksum = blockReduceSum<float>(thread_sum);
    __shared__ float inv_fenmu;
    if (threadIdx.x == 0) {
        inv_fenmu = rsqrtf(float(blocksum / hidden_units) + eps);
    }
    __syncthreads();
    // rmsnorm
    auto vec_weights = reinterpret_cast<Vec_t *>(weights);
    for (int i = threadIdx.x; i < hidden_units / vec_size; i += blockDim.x) {
        Vec_t dout_half2 =vec_dout[i];
        vec_dout[i].x = vec_weights[i].x * __float2half(__half2float(dout_half2.x) * inv_fenmu);
        vec_dout[i].y = vec_weights[i].y * __float2half(__half2float(dout_half2.y) * inv_fenmu);
    }
}

template<typename T>
void launchRMSNorm(TensorWrapper<T> *decoder_out, // [num tokens, hidden_units]
                   TensorWrapper<T> *decoder_residual, // [num tokens, hidden_units]
                   LayerNormWeight<T> &attn_norm_weight, // [hidden_units]
                   float eps, bool is_last) {
    int num_tokens = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int vec_size = Vec<T>::size;
    int num_threads = std::min<int>(hidden_units / vec_size, 1024);
    T *rsd = decoder_residual->data;
    dim3 grid(num_tokens);
    dim3 block(num_threads);
    RMSNorm<T><<<grid, block>>>(decoder_out->data, rsd,
                                attn_norm_weight.gamma,
                                eps, num_tokens, hidden_units);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(decoder_out->data);
#else
#endif
}

template void launchRMSNorm(TensorWrapper<float> *decoder_out, // [num tokens, hidden_units]
                            TensorWrapper<float> *decoder_residual,
                            LayerNormWeight<float> &attn_norm_weight,
                            float eps, bool is_last);
template void launchRMSNorm(TensorWrapper<half> *decoder_out, // [num tokens, hidden_units]
                            TensorWrapper<half> *decoder_residual,
                            LayerNormWeight<half> &attn_norm_weight,
                            float eps, bool is_last);