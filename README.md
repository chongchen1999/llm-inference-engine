# CUDA-based LLM Inference Engine

This repository contains a high-performance LLM (Large Language Model) inference engine built using CUDA, specifically designed for efficient inference on local devices. The project showcases how to implement and optimize the inference of the Llama2 model with GPU acceleration, enhancing performance and responsiveness.

## Problems
    **All the CUDA kernels have been verified to work correctly through unit tests.**
    **This model can run one single transformer layer correctly.**
    **When try to load all weights of 32 transformer layers, the memory is not enough, even with 7b model.**
    **I will try some quantization method later.**

## Features

- **CUDA Kernels**: Implemented essential kernels that form the core of the LLM inference engine, including:
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

- **Kernel Fusion**: Optimized memory access and computation by fusing operators to reduce memory fetches and allocations, improving overall memory efficiency. Examples include:
  - Fusing residual/bias addition with RMSNorm in attention and feedforward layers.
  - Combining GEMM, scaling, and Softmax into a single fused operator for attention score calculation.
  - Merging bias addition with positional encoding during QKV linear transformations.

- **Memory Allocation**: Enhanced memory allocation strategy by merging allocations for QKV activations and other linear transformations, reducing the number of CUDA kernel launches and increasing parallelism.

- **Performance Improvement**: The project includes efforts to understand the workings of large language models (LLMs) and apply advanced programming techniques, such as template programming and object-oriented programming, to build large-scale projects using CMake.

- **Future Work**: Plan to address the issue of limited GPU memory on local devices by quantizing the Llama2-7b model to 16-bit. Further plans include dynamically quantizing the model using PyTorch and developing 4-bit operators to reduce memory usage and accelerate inference.
