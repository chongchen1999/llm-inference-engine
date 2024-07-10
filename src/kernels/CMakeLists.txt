# Specify source files
set(SOURCE_FILES
    input_embedding.cu
)

# Create a library from the source files
add_library(MyProjectLib ${SOURCE_FILES})

# Include directories
target_include_directories(MyProjectLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Set the CUDA architecture
set_target_properties(MyProjectLib PROPERTIES
    CUDA_ARCHITECTURES 86
)

# Link against cuBLAS
target_link_libraries(MyProjectLib PUBLIC cublas)
