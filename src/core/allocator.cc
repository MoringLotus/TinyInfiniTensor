#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        if (freeBlocks.empty())
        {
            used += size;
            peak = std::max(peak, used);
            freeBlocks.emplace(0, size); // Allocate from the start
            return 0;
        }
        for(auto it = freeBlocks.begin(); it != freeBlocks.end(); it ++){
            auto [addr, blockSize] = *it;
            if(blockSize >= size){
                size_t upper_addr = freeBlocks.upper_bound(addr)->first;
                size_t gap = upper_addr - (addr + blockSize);
                if(gap >= size){
                    used += size;
                    freeBlocks[addr + blockSize] = gap; // Update the free block after allocation
                    return addr + blockSize;
                }
            }
        }
        used += size;
        peak = std::max(peak, used);
        size_t lastAddr = freeBlocks.rbegin()->first + freeBlocks.rbegin()->second;
        freeBlocks.emplace(lastAddr, size); // Allocate
        return lastAddr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 =================================== 
        used -= size;
        auto next = freeBlocks.upper_bound(addr);
        if(next != freeBlocks.end() && addr + size == next -> first){ // 再次确保是否物理相邻
            // Merge with next block
            size += next->second;
            freeBlocks.erase(next);
        }
        auto prev = freeBlocks.lower_bound(addr);
        if(prev != freeBlocks.begin() && prev -> first + prev->second == addr){ // 再次确保是否物理相邻
            // Merge with previous block
            size += prev->second;
            addr = prev->first; // Update address to the start of the merged block
            freeBlocks.erase(prev);
        }
        freeBlocks.emplace(addr, size); // Store the freed block
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
