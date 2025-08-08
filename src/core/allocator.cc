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
            freeBlocks[0] = 4096; // Initially, all memory is free
        }
        for(auto it = freeBlocks.begin(); it != freeBlocks.end(); it ++){
            auto [addr, blockSize] = *it;
            if(blockSize >= size){ //blockSize 是可用空间
                if(blockSize > size){
                    // Split the block if it's larger than requested size
                    freeBlocks[addr + size] = blockSize - size;
                }
                freeBlocks.erase(it);
                used += size;
                peak = std::max(peak, used);
                return it->first;
            }
        }
        
        return 0;



        // if (this->freeBlocks.empty())
        //     this->freeBlocks[0] = 1024;
        // for (auto it = this->freeBlocks.begin(); it != this->freeBlocks.end(); ++it)
        // {
        //     if (it->second >= size)
        //     {
        //         if (it->second > size)
        //             this->freeBlocks[it->first + size] = it->second - size;
        //         auto ans = it->first;
        //         this->freeBlocks.erase(it);
        //         this->used += size;
        //         this->peak = (this->peak >= this->used) ? this->peak : this->used;
        //         return ans;
        //     }
        // }
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 =================================== 
        freeBlocks[addr] = size;
        auto it = freeBlocks.find(addr);
        auto nextIt = std::next(it);
        if (nextIt != freeBlocks.end() && it->first + it->second == nextIt->first)
        {
            it->second += nextIt->second;
            freeBlocks.erase(nextIt);
        }
        auto prevIt = std::prev(it);
        if (it != freeBlocks.begin() && prevIt->first + prevIt->second == it->first)
        {
            prevIt->second += it->second;
            freeBlocks.erase(it);
        }
        used = used - size;
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
