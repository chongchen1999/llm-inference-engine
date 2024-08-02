add_library(weight_utils STATIC weight_utils.cu)
set_property(TARGET weight_utils PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET weight_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET weight_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)