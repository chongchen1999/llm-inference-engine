#pragma once

#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include "src/memory/allocator/base_allocator.h"
#include "src/utils/macro.h"

// I use Bytes to printf buffer size msg, because sometimes I allocate <1KB buffer,
// which causes the display to show 0KB
struct CudaBigBlock {
    void *start_pointer;
    size_t size;
    bool is_allocated;

    CudaBigBlock() = default;

    CudaBigBlock(void *start_pointer, size_t size, bool is_allocated) :
        start_pointer(start_pointer),
        size(size),
        is_allocated(is_allocated) {}
};

struct CudaSmallBlock {
    void *start_pointer;
    size_t size;
    bool is_allocated;

    CudaSmallBlock() = default;

    CudaSmallBlock(void *start_pointer, size_t size, bool is_allocated) :
        start_pointer(start_pointer),
        size(size),
        is_allocated(is_allocated) {}
};

class CudaAllocator : public BaseAllocator {
private:
    // map (key, value) is (device_id, block)
    std::map<int, std::vector<CudaSmallBlock>> cuda_small_blocks;
    std::map<int, std::vector<CudaBigBlock>> cuda_big_blocks;
    std::map<int, size_t> free_size;
    size_t totalAllocatedSize = 0;
    int device_id;

public:
    CudaAllocator() {
        cudaGetDevice(&device_id);
    }

    ~CudaAllocator() {
        for (auto &it : cuda_small_blocks) {
            auto &cudaBlocks = it.second; // vector
            for (size_t i = 0; i < cudaBlocks.size(); ++i) {
                cudaFree(cudaBlocks[i].start_pointer);
            }
            auto &big_blocks = cuda_big_blocks[it.first];
            for (size_t i = 0; i < big_blocks.size(); ++i) {
                cudaFree(big_blocks[i].start_pointer);
            }
        }
    }

    void *unifyMalloc(void *ptr, size_t size, bool is_host) {
        // align to 4 bytes
        size = ((size + 31) >> 5) << 5;

        // 1. Host malloc
        if (is_host) {
            ptr = std::malloc(size);
            memset(ptr, 0, size);
            return ptr;
        }

        // 2. Big buffer: Check for free big blocks
        const int threashold = 1 << 20;
        if (size > threashold) {
            auto &big_blocks = cuda_big_blocks[device_id];
            int block_id = -1;
            for (size_t i = 0; i < big_blocks.size(); ++i) {
                auto &cur_block = big_blocks[i];
                if (
                    cur_block.size >= size && 
                    !cur_block.is_allocated &&
                    cur_block.size - size < threashold
                ) {
                    if (block_id == -1 || big_blocks[block_id].size > cur_block.size) {
                        block_id = i;
                    }
                }
            }

            if (block_id != -1) {
                big_blocks[block_id].is_allocated = true;
                return big_blocks[block_id].start_pointer;
            }

            // Allocate new big block
            void *newBuffer;
            cudaMalloc(&newBuffer, size);
            totalAllocatedSize += size;
            big_blocks.push_back(CudaBigBlock(newBuffer, size, true));
            return newBuffer;
        }

        // 3. Small buffer: Check for free small blocks
        auto &smallBlocks = cuda_small_blocks[device_id];
        for (size_t i = 0; i < smallBlocks.size(); ++i) {
            if (smallBlocks[i].size >= size && !smallBlocks[i].is_allocated) {
                smallBlocks[i].is_allocated = true;
                free_size[device_id] += smallBlocks[i].size;
                return smallBlocks[i].start_pointer;
            }
        }

        // 4. Allocate new small block
        void *newBuffer = nullptr;
        CHECK(cudaMalloc(&newBuffer, size));
        CHECK(cudaMemset(newBuffer, 0, size));
        smallBlocks.push_back(CudaSmallBlock(newBuffer, size, true));
        return newBuffer;
    }

    void UnifyFree(void *ptr, bool is_host) {
        if (ptr == nullptr) {
            return;
        }

        // 1. Host free
        if (is_host) {
            free(ptr);
            return;
        }

        // 2. Clean up fragments: If total small buffer size > 1GB, free unallocated small blocks
        for (auto &it : cuda_small_blocks) {
            if (free_size[it.first] > 1024 * 1024 * 1024) {
                auto &cudaBlocks = it.second;
                std::vector<CudaSmallBlock> temp;
                for (size_t i = 0; i < cudaBlocks.size(); ++i) {
                    if (!cudaBlocks[i].is_allocated) {
                        cudaSetDevice(it.first);
                        cudaFree(cudaBlocks[i].start_pointer);
                    } else {
                        temp.push_back(cudaBlocks[i]);
                    }
                }
                cudaBlocks.clear();
                it.second = std::move(temp);
                free_size[it.first] = 0;
            }
        }

        // 3. free buffer and update state
        for (auto &it : cuda_small_blocks) {
            auto &cudaBlocks = it.second;
            for (size_t i = 0; i < cudaBlocks.size(); ++i) {
                if (cudaBlocks[i].start_pointer == ptr) {
                    free_size[it.first] += cudaBlocks[i].size;
                    cudaBlocks[i].is_allocated = false;
                    return;
                }
            }

            auto &big_blocks = cuda_big_blocks[it.first];
            for (size_t i = 0; i < big_blocks.size(); ++i) {
                if (big_blocks[i].start_pointer == ptr) {
                    big_blocks[i].is_allocated = false;
                    return;
                }
            }
        }

        cudaFree(ptr);
    }
};
