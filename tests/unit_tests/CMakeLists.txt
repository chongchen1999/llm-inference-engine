# Specify the test source files
set(TEST_FILES
    test_input_embedding.cu
)

# Create an executable for the unit tests
add_executable(TestInputEmbedding ${TEST_FILES})

# Link the main library and any required libraries (e.g., GTest)
target_link_libraries(TestInputEmbedding PRIVATE MyProjectLib cublas)

# Set the CUDA architecture for the test executable
set_target_properties(TestInputEmbedding PROPERTIES
    CUDA_ARCHITECTURES 86
)

# Add the test to CTest
add_test(NAME TestInputEmbedding COMMAND TestInputEmbedding)
