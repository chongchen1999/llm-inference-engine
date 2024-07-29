# Define libraries
set(CUDA_LIBS -lcudart -lcudadevrt)

# Define executables and their corresponding libraries
add_executable(test_add_residual_and_rmsnorm test_add_residual_and_rmsnorm.cu)
target_link_libraries(test_add_residual_and_rmsnorm PUBLIC ${CUDA_LIBS} add_residual_and_rmsnorm)

add_executable(test_add_residual test_add_residual.cu)
target_link_libraries(test_add_residual PUBLIC ${CUDA_LIBS} add_residual)

add_executable(test_build_causal_mask test_build_causal_mask.cu)
target_link_libraries(test_build_causal_mask PUBLIC ${CUDA_LIBS} build_causal_mask)

add_executable(test_cal_padding_offset test_cal_padding_offset.cu)
target_link_libraries(test_cal_padding_offset PUBLIC ${CUDA_LIBS} cal_padding_offset)

add_executable(test_concat_past_kv test_concat_past_kv.cu)
target_link_libraries(test_concat_past_kv PUBLIC ${CUDA_LIBS} concat_past_kv)

add_executable(test_decoder_self_attention test_decoder_self_attention.cu)
target_link_libraries(test_decoder_self_attention PUBLIC ${CUDA_LIBS} decoder_self_attention)

add_executable(test_input_embedding test_input_embedding.cu)
target_link_libraries(test_input_embedding PUBLIC ${CUDA_LIBS} input_embedding)

add_executable(test_linear test_linear.cu)
target_link_libraries(test_linear PUBLIC ${CUDA_LIBS} linear)

add_executable(test_qkv_bias_and_rope test_qkv_bias_and_rope.cu)
target_link_libraries(test_qkv_bias_and_rope PUBLIC ${CUDA_LIBS} qkv_bias_and_rope)

add_executable(test_repeat_kv test_repeat_kv.cu)
target_link_libraries(test_repeat_kv PUBLIC ${CUDA_LIBS} repeat_kv)

add_executable(test_rmsnorm test_rmsnorm.cu)
target_link_libraries(test_rmsnorm PUBLIC ${CUDA_LIBS} rmsnorm)

add_executable(test_sampling test_sampling.cu)
target_link_libraries(test_sampling PUBLIC ${CUDA_LIBS} sampling)

add_executable(test_scale_and_mask_and_softmax test_scale_and_mask_and_softmax.cu)
target_link_libraries(test_scale_and_mask_and_softmax PUBLIC ${CUDA_LIBS} scale_and_mask_and_softmax)

add_executable(test_silu_and_mul test_silu_and_mul.cu)
target_link_libraries(test_silu_and_mul PUBLIC ${CUDA_LIBS} silu_and_mul)

add_executable(test_topk test_topk.cu)
target_link_libraries(test_topk PUBLIC ${CUDA_LIBS} topk)

add_executable(test_transpose_and_remove_padding test_transpose_and_remove_padding.cu)
target_link_libraries(test_transpose_and_remove_padding PUBLIC ${CUDA_LIBS} transpose_and_remove_padding)
