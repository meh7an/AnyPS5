#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

using AgcDriver::Graphics::Require;
namespace GuestMemory = AgcDriver::GuestMemory;

constexpr std::size_t Unit = 65536;
constexpr std::size_t BlockBytes = 8 * Unit;
constexpr std::array<std::uint32_t, 4> Zero{0u, 0u, 0u, 0u};
constexpr std::array<std::uint32_t, 4> Other{0xababababu, 0xababababu, 0xababababu, 0xababababu};

void* AllocateWatched(std::size_t bytes, std::size_t alignment) {
    if (!GuestMemory::WriteWatched()) return nullptr;
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, alignment);
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) throw std::runtime_error("fill elision: cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + alignment) munmap(reinterpret_cast<void*>(aligned + bytes), begin + alignment - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    return block;
}

void ReleaseWatched(void* block, std::size_t bytes) {
#ifdef _WIN32
    GuestArena::GuestArenaReset_nid_postfix(block, bytes);
    GuestArena::GuestArenaRelease_nid_postfix(block, bytes);
#else
    munmap(block, bytes);
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
#endif
}

// VulkanDevice::FillBuffer's elision over write-watched, host-imported memory: a fill no storage
// image overlaps is dropped when an earlier fill of the same pattern into the same import covered
// its range and nothing stored to the range since; it is recorded when the flag is off, after a CPU
// or GPU store into the range, with another pattern, over a wider range, or after the range was
// registered again. A dropped fill leaves the range's stamp as it was; the bytes are the fill's
// either way. With APS5_NO_FILL_ELIDE=1 every fill is recorded.
int FillElisionTests(AgcDriver::VulkanDevice& device) {
    const bool elision = std::getenv("APS5_NO_FILL_ELIDE") == nullptr;
    void* block = AllocateWatched(BlockBytes, Unit);
    if (block == nullptr) {
        std::puts("skipped, no write watching");
        return VulkanTestSkipped;
    }
    struct Release {
        void* block;
        ~Release() { ReleaseWatched(block, BlockBytes); }
    } release{block};
    std::memset(block, 0x11, BlockBytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    GuestAllocations::Mutation().Add(block, BlockBytes, true, true);
    struct Unregister {
        void* block;
        ~Unregister() { GuestAllocations::Mutation().Remove(block); }
    } unregister{block};
    GuestMemory::CollectWritesUncached(address, BlockBytes);
    const auto* bytes = static_cast<const std::uint8_t*>(block);
    const auto cpuStore = [&](std::size_t offset, std::uint8_t value) { *static_cast<volatile std::uint8_t*>(static_cast<void*>(static_cast<std::uint8_t*>(block) + offset)) = value; };
    const auto holds = [&](std::size_t offset, std::size_t count, std::uint8_t value) {
        for (std::size_t at = offset; at < offset + count; ++at) {
            if (bytes[at] != value) return false;
        }
        return true;
    };
    // Whether the fill was recorded: a recorded fill stamps its range, a dropped one leaves it.
    const auto fill = [&](std::size_t offset, std::size_t count, const std::array<std::uint32_t, 4>& pattern, bool elidable) {
        const auto before = GuestMemory::TrackerGeneration();
        bool stored = false;
        {
            std::lock_guard lock(GuestMemory::GpuMutex());
            stored = device.FillBuffer(address + offset, count, pattern, elidable);
        }
        Require(stored, "fill elision: a fill into the imported block was refused");
        const bool recorded = !GuestMemory::UnchangedSince(address + offset, count, before);
        device.WaitIdle();
        return recorded;
    };
    bool imported = false;
    {
        std::lock_guard lock(GuestMemory::GpuMutex());
        imported = device.FillBuffer(address + Unit, 4 * Unit, Zero, true);
    }
    if (!imported) {
        std::puts("skipped, no host imports");
        return VulkanTestSkipped;
    }
    device.WaitIdle();
    if (!GuestMemory::Watched(address, BlockBytes)) {
        std::puts("skipped, host imports are compared, not watched");
        return VulkanTestSkipped;
    }
    Require(holds(Unit, 4 * Unit, 0x00) && holds(0, Unit, 0x11) && holds(5 * Unit, 3 * Unit, 0x11), "fill elision: the first fill stored the wrong bytes");
    // The same fill again, and one over part of its range: dropped, the bytes kept.
    Require(fill(Unit, 4 * Unit, Zero, true) != elision, "fill elision: a repeated fill was recorded, or dropped with elision off");
    Require(fill(2 * Unit, 2 * Unit, Zero, true) != elision, "fill elision: a fill inside an earlier one was recorded, or dropped with elision off");
    Require(holds(Unit, 4 * Unit, 0x00), "fill elision: a dropped fill changed the bytes");
    // Without the flag (a storage image over the range) the fill is recorded, and its stamp ends what
    // the earlier fill proved: the next one is recorded too, the one after that dropped.
    Require(fill(Unit, 4 * Unit, Zero, false), "fill elision: a fill without the flag was dropped");
    Require(fill(Unit, 4 * Unit, Zero, true), "fill elision: a fill after a recorded store was dropped");
    Require(fill(Unit, 4 * Unit, Zero, true) != elision, "fill elision: a fill after an equal recorded one was recorded, or dropped with elision off");
    // A CPU store into the range: recorded, the store overwritten.
    cpuStore(3 * Unit + 4096, 0x55);
    Require(fill(Unit, 4 * Unit, Zero, true), "fill elision: a fill after a CPU store was dropped");
    Require(holds(Unit, 4 * Unit, 0x00), "fill elision: the CPU's byte survived the fill");
    // A GPU store of another pattern over one unit: recorded, then the zero fill over it recorded.
    Require(fill(2 * Unit, Unit, Other, true), "fill elision: a fill of another pattern was dropped");
    Require(holds(2 * Unit, Unit, 0xab) && holds(Unit, Unit, 0x00), "fill elision: the other pattern's bytes are wrong");
    Require(fill(Unit, 4 * Unit, Zero, true), "fill elision: a fill after a GPU store was dropped");
    Require(holds(Unit, 4 * Unit, 0x00), "fill elision: the GPU store's bytes survived the fill");
    // A wider range than the earlier fill's: recorded.
    Require(fill(Unit, 5 * Unit, Zero, true), "fill elision: a fill wider than the earlier one was dropped");
    Require(holds(Unit, 5 * Unit, 0x00) && holds(6 * Unit, 2 * Unit, 0x11), "fill elision: the wider fill stored the wrong bytes");
    // The range registered again (a mapping change, which may change bytes without a stamp): recorded.
    GuestAllocations::Mutation().Remove(block);
    GuestAllocations::Mutation().Add(block, BlockBytes, true, true);
    Require(fill(Unit, 4 * Unit, Zero, true), "fill elision: a fill after the range was registered again was dropped");
    Require(fill(Unit, 4 * Unit, Zero, true) != elision, "fill elision: a repeated fill after the registration was recorded, or dropped with elision off");
    std::puts(elision ? "fill elision tests passed" : "fill elision off (APS5_NO_FILL_ELIDE): every fill recorded");
    return 0;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        return FillElisionTests(*device);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
