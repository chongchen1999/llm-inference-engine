#pragma once

#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include "base_allocator.h"
#include "../../utils/macro.h"

// I use Bytes to printf buffer size msg, because sometimes I allocate <1KB buffer,
// which causes the display to show 0KB
struct CudaBlock {
    void *start_pointer;
    const size_t size;
    bool is_allocated;

    CudaBlock() = default;

    CudaBlock(void *start_pointer, size_t size, bool is_allocated)
        : start_pointer(start_pointer), size(size), is_allocated(is_allocated) {}
};

class CudaAllocator : public BaseAllocator {
private:
    // map: (key, value) = (device_id, blocks)
    std::map<int, std::vector<CudaBlock>> device_to_small_blocks;
    std::map<int, std::vector<CudaBlock>> device_to_big_blocks;
    std::map<int, size_t> device_to_total_free_small_block_size;

    int device_id;

public:
    CudaAllocator() {
        cudaGetDevice(&device_id);
    }

    ~CudaAllocator() {
        for (auto &[device_id, blocks] : device_to_small_blocks) {
            // Set the device context
            cudaSetDevice(device_id);

            // Free each block in the device
            for (auto &block : blocks) {
                if (block.is_allocated && !block.start_pointer) {
                    cudaFree(block.start_pointer);
                }
            }
        }

        for (auto &[device_id, blocks] : device_to_big_blocks) {
            // Set the device context
            cudaSetDevice(device_id);

            // Free each block in the device
            for (auto &block : blocks) {
                if (block.is_allocated && !block.start_pointer) {
                    cudaFree(block.start_pointer);
                }
            }
        }
    }

    void setDevice(int device_id) {
        this->device_id = device_id;
        cudaSetDevice(device_id);
    }

    void updateDevice() {
        cudaGetDevice(&device_id);
    }

    void allocateByCudaMalloc(
        void **ptr, 
        size_t size, 
        std::vector<CudaBlock> *blocks
    ) {
        CHECK(cudaMalloc(ptr, size));
        blocks->push_back(CudaBlock(*ptr, size, true));
    }

    void unifyMalloc(void **ptr, size_t size, bool is_host) {
        // align to 16 bytes
        size = ((size + 15) >> 4) << 4;

        // Host malloc
        if (is_host) {
            *ptr = std::malloc(size);
            return;
        }

        const size_t threshold = static_cast<size_t>(1 << 20); // 1MB

        // Allocate a big block if size > 1MB, use best fit strategy
        if (size > threshold) {
            auto &big_blocks = device_to_big_blocks[device_id];
            CudaBlock *best_block = nullptr;
            for (auto &cur_block : big_blocks) {
                if (cur_block.size >= size && 
                    !cur_block.is_allocated &&
                    cur_block.size - size < threshold &&
                    (!best_block || best_block->size > cur_block.size)) {
                    best_block = &cur_block;
                }
            }
            
            // Found free big block
            if (best_block) {
                best_block->is_allocated = true;
                *ptr = best_block->start_pointer;
                return;
            }

            // Allocate new big block
            allocateByCudaMalloc(ptr, size, &big_blocks);
            return;
        }

        // Allocate a small block if size <= 1MB, use first fit strategy
        auto &small_blocks = device_to_small_blocks[device_id];
        for (auto &cur_block : small_blocks) {
            if (cur_block.size >= size && !cur_block.is_allocated) {
                cur_block.is_allocated = true;
                device_to_total_free_small_block_size[device_id] -= cur_block.size;
                *ptr = cur_block.start_pointer;
                return;
            }
        }

        // Allocate new small block
        allocateByCudaMalloc(ptr, size, &small_blocks);
    }

    void unifyFree(void *ptr, bool is_host) {
        if (!ptr) {
            return;
        }

        // Host free
        if (is_host) {
            std::free(ptr);
            return;
        }

        cudaGetDevice(&device_id);

        // Check big blocks
        for (auto &cur_block : device_to_big_blocks[device_id]) {
            if (cur_block.start_pointer == ptr) {
                cur_block.is_allocated = false;
                return;
            }
        }

        // Check small blocks
        for (auto &cur_block : device_to_small_blocks[device_id]) {
            if (cur_block.start_pointer == ptr) {
                device_to_total_free_small_block_size[device_id] += cur_block.size;
                cur_block.is_allocated = false;
                return;
            }
        }

        // Clean up fragments: If total small buffer size > 1GB, free unallocated small blocks
        const size_t threshold = static_cast<size_t>(1 << 30); // 1GB

        if (device_to_total_free_small_block_size[device_id] > threshold) {
            std::vector<CudaBlock> temp;
            auto &small_blocks = device_to_small_blocks[device_id];
            for (auto &cur_block : small_blocks) {
                if (!cur_block.is_allocated) {
                    cudaFree(cur_block.start_pointer);
                } else {
                    temp.push_back(cur_block);
                }
            }
            small_blocks.clear();
            small_blocks = std::move(temp);
            device_to_total_free_small_block_size[device_id] = 0;
        }

        // If not in big blocks and small blocks, free by cudaFree
        cudaFree(ptr);
    }
};
