#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_PREPAREDDRAW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_PREPAREDDRAW_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace AgcDriver::DriverDetail {

// One draw on its way from Driver::prepareDraw to Driver::recordPrepared
// (docs/dev/FRONT_BACK_SPLIT.md). It holds the lists the preparation works in, kept with the record
// so a draw reuses the capacity of the one before, and everything the locked tail reads, owned or
// pinned: the stages point into `results`, the matched variants' results and `rectStages`, and the
// snapshots into the decode's registered shaders, the matched variants' words, `shaderMemory` and
// `decodeReads`.
struct PreparedDraw {
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    std::vector<ShaderRecompiler::LinkedProgram> linked;
    std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>> vertexInfos;
    // One recompile request per stage, filled in place by compileDrawStage (never cleared: a
    // request built anew cleared its 1.6 KB first, for every stage of every draw).
    std::vector<ShaderRecompiler::RecompileRequest> requests;
    std::vector<const ShaderRecompiler::RecompileResult*> programResults;
    std::vector<StageCapture> stageCaptures;
    std::vector<std::vector<ShaderRecompiler::MemoryRegion>> matchedRegions;
    std::vector<std::shared_ptr<DispatchVariant>> fresh;
    std::vector<bool> recompiled;
    std::vector<std::uint32_t> pushOffsets;
    std::vector<std::size_t> resultIndex;

    // What recordPrepared reads.
    std::shared_ptr<VulkanDevice> device;
    std::uint32_t queue = 0;
    std::shared_ptr<const DrawDecode> decode;
    Pm4::DrawParameters parameters{};
    std::vector<Graphics::CompiledShader> stages;
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    std::uint64_t drawKey = 0;
    std::vector<std::shared_ptr<DispatchVariant>> recipeStages;
    std::shared_ptr<const DrawRecipe> recipe;

    // What the stages and snapshots point into.
    std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> results;
    std::vector<std::shared_ptr<DispatchVariant>> matched;
    std::array<ShaderRecompiler::RecompileResult, 2> rectStages;
    bool rectList = false;
    std::optional<ShaderMemory> shaderMemory;
    std::vector<std::vector<Graphics::DecodeRead>> decodeReads;

    bool busy = false;

    // Empties the record, keeping every list's capacity (the inner ones of decodeReads and
    // matchedRegions too).
    void Clear() {
        memory.clear();
        linked.clear();
        vertexInfos.clear();
        programResults.clear();
        stageCaptures.clear();
        for (auto& regions : matchedRegions) regions.clear();
        fresh.clear();
        recompiled.clear();
        pushOffsets.clear();
        resultIndex.clear();
        device.reset();
        decode.reset();
        stages.clear();
        snapshots.clear();
        drawKey = 0;
        recipeStages.clear();
        recipe.reset();
        results.clear();
        matched.clear();
        if (rectList) {
            rectStages = {};
            rectList = false;
        }
        shaderMemory.reset();
        for (auto& reads : decodeReads) reads.clear();
    }
};

}

#endif
