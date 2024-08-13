# CUDA-based LLM Inference Engine

This repository contains a high-performance Large Language Model (LLM) inference engine built using CUDA, specifically designed for efficient inference on local devices. The project showcases how to implement and optimize the inference of the Llama2 model with GPU acceleration, enhancing performance and responsiveness.

## Problems

- **CUDA Kernel Verification**: All the CUDA kernels have been verified to work correctly through unit tests.
- **Single Transformer Layer**: The model can correctly run a single transformer layer.
- **Memory Limitation**: When attempting to load all weights of 32 transformer layers, the available memory is insufficient, even with the 7B model.
- **Quantization Plan**: There are plans to explore quantization methods to address the memory limitation.

## Features

- **CUDA Kernels**: Essential kernels have been implemented to form the core of the LLM inference engine, including:
  - Word Embedding
  - Padding Offset Calculation
  - Causal Mask Matrix Construction
  - RMSNorm
  - Tensor Linear Transformation
  - KV-cache Construction & Broadcasting
  - Masked Attention Score Calculation
  - Residual Addition
  - Rotary Position Embedding
  - SwiGLU Activation
  - Sampling

- **Kernel Fusion**: Memory access and computation have been optimized by fusing operators to reduce memory fetches and allocations, improving overall memory efficiency. Examples include:
  - Fusing residual/bias addition with RMSNorm in attention and feedforward layers.
  - Combining GEMM, scaling, and Softmax into a single fused operator for attention score calculation.
  - Merging bias addition with positional encoding during QKV linear transformations.

- **Memory Allocation**: The memory allocation strategy has been enhanced by merging allocations for QKV activations and other linear transformations, reducing the number of CUDA kernel launches and increasing parallelism.

- **Performance Improvement**: The project includes efforts to understand the workings of LLMs and apply advanced programming techniques, such as template programming and object-oriented programming, to build large-scale projects using CMake.

## Future Work

- **Quantization**: Plan to address the issue of limited GPU memory on local devices by quantizing the Llama2-7B model to 16-bit.
- **Dynamic Quantization**: Further plans include dynamically quantizing the model using PyTorch and developing 4-bit operators to reduce memory usage and accelerate inference.
