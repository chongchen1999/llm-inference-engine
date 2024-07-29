# src/kernels/CMakeLists.txt
# Define a function to simplify adding libraries with properties
function(add_cuda_library name)
    add_library(${name} STATIC ${ARGN})
    set_property(TARGET ${name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()

# Add libraries for each kernel
add_cuda_library(add_residual_and_rmsnorm add_residual_and_rmsnorm.cu)
add_cuda_library(add_residual add_residual.cu)
add_cuda_library(build_causal_mask build_causal_mask.cu)
add_cuda_library(cal_padding_offset cal_padding_offset.cu)
add_cuda_library(concat_past_kv concat_past_kv.cu)
add_cuda_library(decoder_self_attention decoder_self_attention.cu)
add_cuda_library(input_embedding input_embedding.cu)
add_cuda_library(qkv_bias_and_rope qkv_bias_and_rope.cu)
add_cuda_library(repeat_kv repeat_kv.cu)
add_cuda_library(rmsnorm rmsnorm.cu)
add_cuda_library(sampling sampling.cu)
add_cuda_library(scale_and_mask_and_softmax scale_and_mask_and_softmax.cu)
add_cuda_library(silu_and_mul silu_and_mul.cu)
add_cuda_library(topk topk.cu)
add_cuda_library(transpose_and_remove_padding transpose_and_remove_padding.cu)

add_library(cublasWrapper STATIC cublas_utils.cpp)
set_property(TARGET cublasWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cublasWrapper CUDA::cublas CUDA::cuda_driver)

add_cuda_library(linear linear.cu)
target_link_libraries(linear PUBLIC -lcudart -lcublas cublasWrapper)