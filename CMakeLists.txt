cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)

# Set the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the root directory as an include directory
include_directories(${CMAKE_SOURCE_DIR})

# Add the source files
set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/kernels/cal_padding_offset.cu
)

# Add the test sources
set(TEST_SOURCES
    ${CMAKE_SOURCE_DIR}/tests/unit_tests/test_cal_padding_offset.cu
)

# Add the executable for the test
add_executable(test ${SOURCES} ${TEST_SOURCES})

# Link CUDA libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(test ${CUDAToolkit_LIBRARIES})
