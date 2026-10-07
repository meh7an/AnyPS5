#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWCACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWCACHE_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <vector>

namespace AgcDriver::DriverDetail {

struct DrawProgram {
    ShaderRecompiler::ShaderBinary binary;
    std::uint32_t userDataBase;
    std::uint32_t firstUserSgpr = 8;
    std::vector<std::uint32_t> userData;
    std::array<ShaderRecompiler::MemoryRegion, 2> memory;

    std::shared_ptr<const ShaderSnapshot> snapshot;
    std::size_t codeOffset = 0;
    // A merged stage's user pointer register, or 0: userData then starts with eight words, the
    // pointer's two (zero when it is unset) and six zeros, before the user words.
    std::uint32_t mergedPointer = 0;
    bool mergedPointerRequired = false;
};

struct DrawDecode {
    Graphics::State state;
    ShaderRecompiler::ShaderPixelStageInfo pixel;
    std::vector<DrawProgram> programs;
    std::vector<ShaderRecompiler::ProgramRole> roles;
};

struct DrawRecipeRecord {
    std::vector<std::weak_ptr<const DispatchVariant>> stages;
    std::shared_ptr<const DrawRecipe> recipe;
    bool Matches(const std::vector<std::shared_ptr<DispatchVariant>>& variants) const;
    bool Expired() const;
};

struct DrawEntry {

    std::shared_ptr<const DrawDecode> decode;

    std::vector<std::vector<std::shared_ptr<DispatchVariant>>> stages;

    std::atomic<std::shared_ptr<const std::vector<DrawRecipeRecord>>> recipes;
    std::uint64_t touched = 0;
    std::list<std::uint64_t>::iterator order;
};

enum class DrawMiss : std::size_t { FrontDiffering, FragmentDiffering, OtherDiffering, Layout, Gate, Stages, Count };

struct DrawEntryCounters {
    std::uint64_t lookups = 0, absent = 0, hits = 0, stageValidations = 0, stageEqual = 0, variantsCompared = 0, inserts = 0, variantsInserted = 0, variantsEvicted = 0, present = 0, unstable = 0, touches = 0, verifyHits = 0, verifyMismatches = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawMiss::Count)> misses{};
    std::array<std::uint64_t, MaxDispatchVariants> variantHitsByRank{};
    double validateUs = 0;

    std::uint64_t registerKeyLookups = 0, registerKeyHits = 0, decodeSkipped = 0, decodePartial = 0, facadeMismatches = 0, verifyDecodes = 0, verifyDecodeMismatches = 0;
    double keyUs = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

enum class DrawVerdict { Drawn, Nothing, Rejected };

struct StageCapture {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    std::uint64_t forgetSerial = 0;
    std::uint32_t pushOffset = 0;
    // The stage memo entry `regions` point into when the stage was served from it.
    std::shared_ptr<const void> memo;
};

// The stage memo (Step 4, DrawCapture.cpp): a stage's compiled result and the words its capture
// read, kept by its source, push offset and the user data bits the source's capture and results
// depend on other than as copies (ShaderRecompiler::UserDataKeyFor). A later stage with an equal key
// whose captured words are unchanged (no write stamp newer than the capture's over them, no storage
// image results pending and no unit shadow live there, no mapping changed) takes the result with
// its user words copied in (ShaderRecompiler::RecompileOverUserData) instead of capturing and
// recompiling.
struct StageMemoEntry {
    std::shared_ptr<const ShaderRecompiler::SourceHandle> handle;
    std::uint32_t pushOffset = 0;
    std::vector<std::uint32_t> key;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    std::shared_ptr<const std::vector<ShaderRecompiler::UserDataPatch>> patches;
    // The capture's reads (address and words) and the ranges validated, merged within 64 KiB (the
    // write tracker's block).
    std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> regions;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
    std::uint64_t generation = 0;
    // The guest registry's generation from before the capture.
    std::uint64_t mappings = 0;
};

}

#endif
