find_package(CUDA REQUIRED)

add_library(layer_weights STATIC layer_weights.cpp)
target_include_directories(layer_weights PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(layer_weights PUBLIC -lcudart weightutils)
set_property(TARGET layer_weights PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET layer_weights PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_library(llama_weights STATIC llama_weights.cpp)
target_include_directories(llama_weights PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(llama_weights PUBLIC -lcudart weightutils)
set_property(TARGET llama_weights PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llama_weights PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)