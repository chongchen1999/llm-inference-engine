cmake_minimum_required(VERSION 3.8)

add_library(llama_context_attention STATIC context_attention.cpp)
set_property(TARGET llama_context_attention PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET llama_context_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(
    llama_context_attention PUBLIC
    -lcudart
    -lcudadevrt
    qkv_bias_and_rope
    concat_past_kv
    cublasWrapper
    linear
    transpose_and_remove_padding
    repeat_kv
    scale_and_mask_and_softmax
)

add_library(llama_self_attention STATIC self_attention.cpp)
set_property(TARGET llama_self_attention PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET llama_self_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(
    llama_self_attention PUBLIC
    -lcudart
    -lcudadevrt
    qkv_bias_and_rope
    concat_past_kv
    cublasWrapper
    linear
    transpose_and_remove_padding
    repeat_kv
    scale_and_mask_and_softmax
    decoder_self_attention
)

add_library(llama_ffn STATIC ffn.cpp)
set_property(TARGET llama_ffn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET llama_ffn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(
    llama_ffn PUBLIC
    -lcudart
    -lcudadevrt
    cublasWrapper
    qkv_bias_and_rope
    concat_past_kv
    linear
    silu_and_mul
    transpose_and_remove_padding
    repeat_kv
    scale_and_mask_and_softmax
    decoder_self_attention
)