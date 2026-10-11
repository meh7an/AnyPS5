#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_WRITEWATCHCOVERAGE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_WRITEWATCHCOVERAGE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <limits>
#include <unordered_map>

namespace AgcDriver::GuestMemory {

class WriteWatchCoverage {
public:
    void Initialize(std::uint64_t address, std::size_t bytes) {
        base = address;
        size = bytes;
        excluded.assign(bytes / BlockBytes + (bytes % BlockBytes != 0), 0);
        excludedPerChunk.assign((excluded.size() + ChunkBlocks - 1) / ChunkBlocks, 0);
        excludedGenerations.clear();
    }

    // Whole chunks by their count of excluded blocks, the rest block by block: a render target's
    // range is checked on every draw that samples it.
    bool Covers(std::uint64_t address, std::size_t bytes) const {
        if (bytes == 0 || bytes > size || address < base || address - base > size - bytes) return false;
        const auto first = (address - base) / BlockBytes;
        const auto last = (address - base + bytes - 1) / BlockBytes;
        for (auto block = first; block <= last;) {
            if (block % ChunkBlocks == 0 && last - block >= ChunkBlocks - 1) {
                if (excludedPerChunk[block / ChunkBlocks] != 0) return false;
                block += ChunkBlocks;
                continue;
            }
            if (excluded[block] != 0) return false;
            ++block;
        }
        return true;
    }

    bool Exclude(std::uint64_t address, std::size_t bytes, std::uint64_t generation = 1) {
        if (bytes == 0 || size == 0) return false;
        if (address < base) {
            if (base - address >= bytes) return false;
            bytes -= base - address;
            address = base;
        }
        const auto offset = address - base;
        if (offset >= size) return false;
        bytes = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, size - offset));
        const auto first = offset / BlockBytes;
        const auto last = (offset + bytes - 1) / BlockBytes;
        bool changed = false;
        for (auto block = first; block <= last; ++block) {
            if (excluded[block] == 0) {
                changed = true;
                ++excludedPerChunk[block / ChunkBlocks];
            }
            auto& stamp = excludedGenerations[block];
            stamp = std::max(stamp, std::max<std::uint64_t>(generation, 1));
            excluded[block] = 1;
        }
        return changed;
    }

    bool Restore(std::uint64_t address, std::size_t bytes, std::uint64_t generation = std::numeric_limits<std::uint64_t>::max()) {
        if (bytes == 0 || size == 0) return false;
        if (address < base) {
            if (base - address >= bytes) return false;
            bytes -= base - address;
            address = base;
        }
        const auto offset = address - base;
        if (offset >= size) return false;
        bytes = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, size - offset));
        const auto first = offset / BlockBytes + (offset % BlockBytes != 0);
        const auto stop = offset + bytes;
        const auto end = stop / BlockBytes + (stop == size && stop % BlockBytes != 0);
        bool changed = false;
        for (auto block = first; block < end; ++block) {
            if (excluded[block] != 0 && excludedGenerations.at(block) < generation) {
                changed = true;
                excluded[block] = 0;
                --excludedPerChunk[block / ChunkBlocks];
                excludedGenerations.erase(block);
            }
        }
        return changed;
    }

private:
    static constexpr std::size_t BlockBytes = 65536;
    static constexpr std::size_t ChunkBlocks = 64;
    std::uint64_t base = 0;
    std::size_t size = 0;
    std::vector<std::uint8_t> excluded;
    // The excluded blocks of each chunk of ChunkBlocks blocks.
    std::vector<std::uint16_t> excludedPerChunk;
    std::unordered_map<std::uint64_t, std::uint64_t> excludedGenerations;
};

}

#endif
