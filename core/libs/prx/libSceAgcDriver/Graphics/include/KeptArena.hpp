#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_KEPTARENA_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_KEPTARENA_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

namespace AgcDriver::Graphics {

// Storage for the objects a recorded draw keeps until its batch completed (Draw.cpp's Kept, one per
// draw), carved in order out of blocks by the recording thread and given back block by block. The
// objects die on the release thread, so a heap block one of them occupied was last written there:
// handed out again to the recording thread, taking it off the free list missed the cache on every
// draw (2% of the worker on S3K). Carving reads nothing. Each allocation keeps its block's address
// just before it, and a block returns to the heap once every object in it was released (from any
// thread) and the arena moved on from it, so an object may outlive its batch or the arena.
class KeptArena {
public:
    static constexpr std::size_t BlockBytes = 64 * 1024;

    KeptArena() = default;
    KeptArena(const KeptArena&) = delete;
    KeptArena& operator=(const KeptArena&) = delete;
    ~KeptArena() { retire(); }

    // `bytes` aligned to `alignment` (at most 16) from the current block, or a new one when it is
    // full. The arena's users serialize these calls (the GPU mutex, for the recorder's).
    void* Allocate(std::size_t bytes, std::size_t alignment) {
        Require(alignment <= 16 && bytes <= BlockBytes - HeaderBytes - 32, "kept object does not fit an arena block");
        const auto align = std::max(alignment, alignof(Block*));
        auto at = (used + sizeof(Block*) + align - 1) & ~(align - 1);
        if (block == nullptr || at + bytes > BlockBytes) {
            retire();
            block = new (::operator new(BlockBytes, std::align_val_t{alignof(Block)})) Block;
            liveBlocks.fetch_add(1, std::memory_order_relaxed);
            allocations = 0;
            at = (HeaderBytes + sizeof(Block*) + align - 1) & ~(align - 1);
        }
        auto* base = reinterpret_cast<std::byte*>(block);
        std::memcpy(base + at - sizeof(Block*), &block, sizeof(Block*));
        used = at + bytes;
        ++allocations;
        return base + at;
    }

    // Gives back one allocation of any arena, from any thread.
    static void Release(void* pointer) noexcept {
        Block* owner;
        std::memcpy(&owner, static_cast<std::byte*>(pointer) - sizeof(Block*), sizeof(Block*));
        if (owner->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) giveBack(owner);
    }

    // Blocks taken from the heap and not given back, over every arena (tests).
    static std::size_t LiveBlocks() { return liveBlocks.load(std::memory_order_relaxed); }

    // For std::allocate_shared: the object and its control block in one allocation of the arena.
    template<typename T>
    struct Allocator {
        using value_type = T;
        KeptArena* arena;
        explicit Allocator(KeptArena& arena) : arena(&arena) {}
        template<typename U>
        Allocator(const Allocator<U>& other) : arena(other.arena) {}
        T* allocate(std::size_t count) { return static_cast<T*>(arena->Allocate(count * sizeof(T), alignof(T))); }
        // The arena is not touched: the object may outlive it.
        void deallocate(T* pointer, std::size_t) noexcept { Release(pointer); }
        template<typename U>
        bool operator==(const Allocator<U>& other) const { return arena == other.arena; }
    };

private:
    // A block's header, alone on its cache line: the releasing threads' count stays off the lines
    // the recording thread fills. `remaining` holds Bias less the releases while the arena carves
    // the block; leaving it subtracts what was never allocated, so the count reaches zero exactly
    // when the last object was released and the block was left, whichever comes last.
    struct alignas(64) Block {
        std::atomic<std::uint64_t> remaining{Bias};
    };
    static constexpr std::uint64_t Bias = std::uint64_t{1} << 62;
    static constexpr std::size_t HeaderBytes = sizeof(Block);

    static void giveBack(Block* owner) noexcept {
        owner->~Block();
        ::operator delete(owner, std::align_val_t{alignof(Block)});
        liveBlocks.fetch_sub(1, std::memory_order_relaxed);
    }

    void retire() noexcept {
        if (block == nullptr) return;
        if (block->remaining.fetch_sub(Bias - allocations, std::memory_order_acq_rel) == Bias - allocations) giveBack(block);
        block = nullptr;
    }

    static inline std::atomic<std::size_t> liveBlocks{0};
    Block* block = nullptr;
    std::size_t used = 0;
    std::uint64_t allocations = 0;
};

}

#endif
