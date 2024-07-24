# Memory/GPU Memory Allocator

## Goals
- Reduce the frequency of malloc and free calls
- Minimize memory fragmentation

## Our Allocator
- **Block**: `ptr, size, is_allocated`
- **Blockpool**: `map<int, vector<block>>`
- **UnifyMalloc**: 
  - `bigblockpoolforbigbuf, smallblockpoolforsmallbuf`
  - Find buffer from pool (block vector)
  - If not found, `cudamalloc`
- **UnifyFree**:
  - Clear memory fragment
  - If pointer found in blockpool, set `is_allocated=false`, do not return to OS
  - If not found, `cudaFree`

## PyTorch Allocator
- **Block**: `ptr, size, is_allocated, prev, next`, doubly linked list
- **Blockpool**: `set<block>`
- **Malloc**:
  - Find buffer from pool (set)
  - If not found, try to merge free blocks into a big block
  - If not found, try to release free blocks, then search if requirements are met
  - If not found, try to split big block to clean up memory fragmentation
  - If not found, `cudamalloc`

## Other Allocators
- **TF BFC Allocator**
- **Buddy Allocator**
- ...
