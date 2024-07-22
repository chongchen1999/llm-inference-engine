cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)

# Set the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
)

# Add source files
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/kernels/fused_transpose_and_remove_padding.cu
    # ${CMAKE_SOURCE_DIR}/src/kernels/cublas_utils.cc
    # Add other source files here, e.g., cublasWrapper.cpp
)

# Add test source files
set(TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/tests/unit_tests/test_fused_transpose_and_remove_padding.cu
)

# Add the executable for the tests
add_executable(test ${SOURCES} ${TEST_SOURCES})

# Find CUDA toolkit including cuBLAS
find_package(CUDAToolkit REQUIRED)

# Check if CUDA Toolkit was found
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found.")
endif()

# Link CUDA and cuBLAS libraries
target_link_libraries(test CUDA::cublas CUDA::cuda_driver)

# Optionally, you can also specify other libraries if needed
# target_link_libraries(test other_libraries)

# If your project uses additional CUDA libraries, you might need to link them as well
# For example:
# find_package(CUDA REQUIRED)
# target_link_libraries(test ${CUDA_LIBRARIES})
