Here’s a rewritten and organized version of the allocator descriptions to improve clarity and readability:

---

### Allocator Design Goals

**Objective:**  
- Reduce the frequency of `malloc` and `free` calls
- Minimize memory fragmentation

### Our Allocator

1. **Block**: 
   - Attributes: `ptr`, `size`, `is_allocated`

2. **Blockpool**: 
   - Structure: `map<int, vector<Block>>`

3. **UnifyMalloc**: 
   - **Pools**:
     - `bigBlockPool` for large buffers
     - `smallBlockPool` for small buffers
   - **Allocation Process**:
     - Search for the buffer in the appropriate pool (`block vector`)
     - If not found, use `cudaMalloc`

4. **UnifyFree**:
   - **Deallocation Process**:
     - Check if the pointer is in the `blockPool`:
       - Set `is_allocated` to `false`
       - Do not return to the OS
     - If not found in `blockPool`, use `cudaFree`
   - **Memory Fragmentation**:
     - Implement strategies to clear memory fragments

### PyTorch Allocator

1. **Block**:
   - Attributes: `ptr`, `size`, `is_allocated`, `prev`, `next`
   - Structure: Doubly linked list

2. **Blockpool**:
   - Structure: `set<Block>`

3. **Malloc Process**:
   - Search for the buffer in the `Blockpool` (`set`)
   - If not found:
     - Attempt to merge free blocks into a larger block
     - If merging is unsuccessful:
       - Release free blocks and try to find a suitable block
     - If no suitable block is found:
       - Split a large block to clean up memory fragments
     - If still not successful, use `cudaMalloc`

### TensorFlow BFC Allocator

1. **Block**:
   - Attributes: Similar to PyTorch, but with a focus on Buddy Allocator.

2. **Buddy Allocator**:
   - Allocates and manages memory in a buddy system to reduce fragmentation.
   - **Details**: Specific mechanisms for splitting and merging blocks.

3. **Other Allocators**:
   - The TensorFlow BFC allocator uses variations of the buddy allocator and additional strategies for memory management.

---

This structured approach outlines the key features and strategies of each allocator, emphasizing the process and techniques used to handle memory allocation and deallocation efficiently.