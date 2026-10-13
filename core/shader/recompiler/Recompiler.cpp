#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include "CacheKey.hpp"
#include "CompiledVariant.hpp"
#include "VertexInputSpecialization.hpp"
#include "FragmentOutputSpecialization.hpp"
#include "SpirvBackend/SpirvSpecialization.hpp"
#include "RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "ShaderDiskCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include <list>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include "ControlFlow/include/ControlFlow/GraphBuilder.hpp"
#include "ControlFlow/include/ControlFlow/Structurizer.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaInstructionDecoder.hpp"
#include "IntermediateRepresentation/include/IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/include/Optimization/BindingAllocator.hpp"
#include "Optimization/include/Optimization/ConstantFolder.hpp"
#include "Optimization/include/Optimization/DeadCodeEliminator.hpp"
#include "Optimization/include/Optimization/DenormalFlushEliminator.hpp"
#include "Optimization/include/Optimization/DescriptorBindingBuilder.hpp"
#include "Optimization/include/Optimization/HostInterpolationChecker.hpp"
#include "Optimization/include/Optimization/MaskedSelectEliminator.hpp"
#include "Optimization/include/Optimization/ReadLaneEliminator.hpp"
#include "Optimization/include/Optimization/RequestMemoryView.hpp"
#include "Optimization/include/Optimization/ResourceMaterializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/include/Optimization/ResourceTracker.hpp"
#include "Optimization/include/Optimization/ShaderInfoCollector.hpp"
#include "Optimization/include/Optimization/SingleLaneAnalysis.hpp"
#include "Optimization/include/Optimization/SrtWalker.hpp"
#include "Optimization/SrtWalker/SrtInstructionPredicates.hpp"
#include "Optimization/include/Optimization/SsaBuilder.hpp"
#include "SpirvBackend/include/SpirvBackend/SpirvEmitter.hpp"
#if ANYPS5_ENABLE_SPIRV_TOOLS
#include "SpirvBackend/SpirvOptimizer.hpp"
#endif
#include "SpirvBackend/SpirvMemory/SpirvInputOutput.hpp"
#include "Translation/include/Translation/InstructionTranslator.hpp"
#include "Translation/include/Translation/ShaderInputInfoBuilder.hpp"
#include <exception>
#include <stdexcept>
#include <string>
#include <ControlFlow/RequestSerializer.hpp>

namespace ShaderRecompiler {

namespace {

ShaderStageKind toShaderStageKind(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Compute:
        return ShaderStageKind::Compute;
    case ShaderStage::Vertex:
        return ShaderStageKind::Vertex;
    case ShaderStage::TessellationControl:
        return ShaderStageKind::TessellationControl;
    case ShaderStage::TessellationEvaluation:
        return ShaderStageKind::TessellationEvaluation;
    case ShaderStage::Fragment:
        return ShaderStageKind::Pixel;
    case ShaderStage::Local:
        return ShaderStageKind::Local;
    case ShaderStage::Mesh:
        return ShaderStageKind::Mesh;
    case ShaderStage::Geometry:
        break;
    }
    throw std::runtime_error("ShaderRecompiler::Recompile: unsupported shader stage");
}

}

namespace {

// The host subgroup width wave64 programs are laid out for. Debug aid: APS5_SINGLE_LANE=<hex code
// addresses, comma separated, or "all"> keeps the listed programs at one guest lane per invocation.
std::uint32_t HostSubgroupSize(const RecompileRequest& request) {
    static const std::string list = [] { const char* text = std::getenv("APS5_SINGLE_LANE"); return text ? std::string(text) : std::string(); }();
    if (!list.empty()) {
        if (list == "all") return 64u;
        char address[32];
        std::snprintf(address, sizeof(address), "%llx", static_cast<unsigned long long>(request.shader.codeAddress));
        if (list.find(address) != std::string::npos) return 64u;
    }
    return request.target.subgroupSize;
}

// Whether SingleLaneAnalysis picks this request's layout: a wave64 compute program laid out for a
// 32-wide host runs at one lane per invocation where that gives every lane the same results, and
// at two otherwise. Debug aid: APS5_NO_SINGLE_LANE_RULE=1 keeps every such program at two.
bool SingleLaneCandidate(const RecompileRequest& request) {
    static const bool enabled = std::getenv("APS5_NO_SINGLE_LANE_RULE") == nullptr;
    return enabled && request.shader.stage == ShaderStage::Compute && request.context.waveSize == 64u && HostSubgroupSize(request) == 32u;
}

// The disk cache key's layout word: the host subgroup size, with bit 8 set where the single-lane
// rule picks the layout (the cached result holds its choice, which a run without the rule must not
// load).
std::uint32_t CacheLayout(const RecompileRequest& request) {
    return HostSubgroupSize(request) | (SingleLaneCandidate(request) ? 0x100u : 0u);
}

ShaderStageInputInfo RequestInputInfo(const RecompileRequest& request) {
    const auto* mesh = request.graphics && request.graphics->mesh ? &*request.graphics->mesh : nullptr;
    const auto* tessellation = request.graphics && request.graphics->tessellation ? &*request.graphics->tessellation : nullptr;
    return BuildShaderStageInputInfo(toShaderStageKind(request.shader.stage), request.context, HostSubgroupSize(request), mesh, tessellation);
}

std::uint32_t DeviceMemoryLdsBytes(const RecompileRequest& request) {
    if (!request.context.compute.has_value()) return 0u;
    const auto bytes = static_cast<std::uint64_t>(request.context.compute->ldsSizeDwords) * 4u;
    return bytes + 4u > request.target.maxWorkgroupSharedMemoryBytes ? static_cast<std::uint32_t>(bytes) : 0u;
}

// RequestInputInfo's errors without its input info: a vertex-family capture only validates.
void ValidateRequestInputs(const RecompileRequest& request) {
    const auto stage = toShaderStageKind(request.shader.stage);
    const bool vertexFamily = stage == ShaderStageKind::Vertex || stage == ShaderStageKind::Local || stage == ShaderStageKind::TessellationControl || stage == ShaderStageKind::TessellationEvaluation || stage == ShaderStageKind::Mesh;
    if (!vertexFamily) {
        static_cast<void>(RequestInputInfo(request));
        return;
    }
    const auto* mesh = request.graphics && request.graphics->mesh ? &*request.graphics->mesh : nullptr;
    const auto* tessellation = request.graphics && request.graphics->tessellation ? &*request.graphics->tessellation : nullptr;
    ValidateVertexStageInputs(stage, request.context, mesh, tessellation);
}

}

struct PreparedControlFlow {
    PreparedControlFlow(const RecompileRequest& request, const SwappcInfo& swappcInfo)
        : stage(request.shader.stage), swappc(swappcInfo), code(request.shader.code.begin(), request.shader.code.end()),
          decoded(RdnaInstructionDecoder{}.Decode(code)), cfg(GraphBuilder{}.Build(decoded, &swappc)) {
        Structurizer{}.Structurize(cfg);
    }

    PreparedControlFlow(const PreparedControlFlow&) = delete;
    PreparedControlFlow& operator=(const PreparedControlFlow&) = delete;

    bool Matches(const RecompileRequest& request, const SwappcInfo& swappcInfo) const {
        return stage == request.shader.stage && swappc.fetchCallAllowed == swappcInfo.fetchCallAllowed &&
               swappc.userDataBaseRegister == swappcInfo.userDataBaseRegister && swappc.userDataCount == swappcInfo.userDataCount &&
               std::ranges::equal(code, request.shader.code);
    }

    ShaderStage stage;
    SwappcInfo swappc;
    std::vector<std::uint32_t> code;
    RdnaProgram decoded;
    ControlFlowGraph cfg;
};

std::shared_ptr<const PreparedControlFlow> ShaderPreparationContext::AcquireFrontend(const RecompileRequest& request) {
    const auto inputInfo = RequestInputInfo(request);
    const SwappcInfo swappc{inputInfo.vertex != nullptr, request.context.userDataBaseRegister, static_cast<std::uint32_t>(request.context.userData.size())};
    static const bool reuse = std::getenv("APS5_NO_PERF_FRONTEND_PAIR") == nullptr;
    if (reuse && frontend != nullptr && frontend->Matches(request, swappc)) {
        return frontend;
    }
    auto prepared = std::make_shared<const PreparedControlFlow>(request, swappc);
    if (reuse) {
        frontend = prepared;
    }
    return prepared;
}

IrProgram PrepareResourceProgram(const RecompileRequest& request) {
    return PrepareResourceProgram(request, nullptr);
}

IrProgram PrepareResourceProgram(const RecompileRequest& request, ShaderPreparationContext* preparation) {
    ShaderPreparationContext local;
    const auto frontend = (preparation != nullptr ? *preparation : local).AcquireFrontend(request);
    const auto stageKind = toShaderStageKind(request.shader.stage);
    const auto inputInfo = RequestInputInfo(request);
    const auto& decoded = frontend->decoded;
    const auto& cfg = frontend->cfg;

    TranslateOptions translateOptions {};
    translateOptions.stage = stageKind;
    translateOptions.shaderHash = request.shader.codeAddress;
    translateOptions.waveSize = request.context.waveSize;
    translateOptions.userDataBaseRegister = request.context.userDataBaseRegister;
    translateOptions.userDataCount = static_cast<std::uint32_t>(request.context.userData.size());
    translateOptions.scratchDwords = request.context.compute.has_value() ? request.context.compute->scratchDwords : 0u;
    translateOptions.sharedMemoryBytes = DeviceMemoryLdsBytes(request);
    translateOptions.fragmentShaderBarycentricEnabled = request.target.fragmentShaderBarycentricEnabled;
    translateOptions.floatMode = request.context.floatMode;
    translateOptions.inputInfo = inputInfo;

    constexpr InstructionTranslator translator;

    EmbeddedFetchPlan embeddedFetch;
    if ((stageKind == ShaderStageKind::Vertex || stageKind == ShaderStageKind::Local) && inputInfo.vertex != nullptr && inputInfo.vertex->fetchEmbedded) {
        embeddedFetch = EmbeddedVertexFetchAnalyzer{}.Analyze(decoded, inputInfo.vertex->fetchAttribReg, inputInfo.vertex->fetchBufferReg, request.context.userDataBaseRegister, static_cast<std::uint32_t>(request.context.userData.size()), request.context.waveSize);
        translateOptions.embeddedFetch = &embeddedFetch;
    }
    auto program = translator.Translate(decoded, cfg, translateOptions);
    // Debug aid: APS5_DUMP_IR=<hex code address> (or "all") prints the program after each front-end pass.
    const auto dumpIr = [&](const char* pass) {
        static const std::string list = [] { const char* text = std::getenv("APS5_DUMP_IR"); return text ? std::string(text) : std::string(); }();
        if (list.empty()) return;
        char address[32];
        std::snprintf(address, sizeof(address), "%llx", static_cast<unsigned long long>(request.shader.codeAddress));
        if (list != "all" && list.find(address) == std::string::npos) return;
        std::fprintf(stderr, "==== IR 0x%s after %s\n%s\n", address, pass, ProgramToString(program).c_str());
    };

    constexpr SsaBuilder ssaBuilder;
    constexpr ConstantFolder constantFolder;
    constexpr DeadCodeEliminator deadCodeEliminator;
    constexpr ReadLaneEliminator readLaneEliminator;
    constexpr MaskedSelectEliminator maskedSelectEliminator;
    const auto simplify = [&] {
        dumpIr("translate");
        ssaBuilder.Rewrite(program);
        dumpIr("ssa");

        constantFolder.Fold(program);
        ResolveControlFlowIdentities(program);
        deadCodeEliminator.RemoveIdentities(program);
        deadCodeEliminator.Eliminate(program);
        dumpIr("fold");

        const auto readLaneStats = readLaneEliminator.Eliminate(program, translateOptions.waveSize);
        if (readLaneStats.rewrittenReads != 0u) {
            constantFolder.Fold(program);
            ResolveControlFlowIdentities(program);
            deadCodeEliminator.RemoveIdentities(program);
            deadCodeEliminator.Eliminate(program);
        }

        if (maskedSelectEliminator.Eliminate(program).removedSelects != 0u) {
            deadCodeEliminator.Eliminate(program);
        }
    };
    simplify();
    if (stageKind == ShaderStageKind::Pixel && !translateOptions.fragmentShaderBarycentricEnabled && !HostInterpolationChecker{}.Lower(program, *inputInfo.pixel)) {
        translateOptions.fragmentShaderBarycentricEnabled = true;
        program = translator.Translate(decoded, cfg, translateOptions);
        program.Metadata().barycentricEmulation = true;
        simplify();
    }

    constexpr DenormalFlushEliminator denormalFlushEliminator;
    if (denormalFlushEliminator.Eliminate(program).removedFlushes != 0u) {
        deadCodeEliminator.Eliminate(program);
    }

    constexpr SrtWalker srtWalker;
    srtWalker.BuildPlan(program);
    deadCodeEliminator.Eliminate(program);
    dumpIr("srt");

    constexpr ResourceTracker resourceTracker;
    resourceTracker.Track(program);
    deadCodeEliminator.Eliminate(program);
    dumpIr("resources");
    program.Resources().srgbDecodeFormats = request.target.srgbDecodeFormats;

    return program;
}

// A materialized result of one variant over one snapshot (Recompile(request, capture)): the
// shared immutable object every later capture that reproduces the snapshot receives, so Populate
// and the per-request copy run once per distinct snapshot.
struct ResultMemoEntry {
    std::uint64_t variantId = 0;
    std::uint64_t hash = 0;
    std::shared_ptr<const RecompileResult> result;
    // The source's memo clock at the entry's last use: a set replaces its least recently used.
    std::uint64_t used = 0;
};

// The result memo of a source: a set of ResultMemoWays entries per index (its folded low bits),
// so a lookup reads one or two cache lines instead of a hash node chain behind a division.
constexpr std::size_t ResultMemoSets = 64;
constexpr std::size_t ResultMemoWays = 4;
using ResultMemoTable = std::array<ResultMemoEntry, ResultMemoSets * ResultMemoWays>;

// The stage memo's view of a source's plan (UserDataKeyFor): the key, and per buffer and direct image
// the user word each descriptor dword copies (-1 for none).
struct UserDataAnalysis {
    std::shared_ptr<const UserDataKey> key;
    std::vector<std::array<std::int32_t, 4>> bufferCopies;
    std::vector<std::array<std::int32_t, 8>> imageCopies;
    // Per buffer, the SRT read slots of its V#'s words 0 and 1 when word 0 is a memory copy
    // (analyzeMemoryCopies), else {-1, -1}; and the raw read nodes of those slots, which a capture
    // traces (null without copies).
    std::vector<std::array<std::int32_t, 2>> bufferBaseReads;
    std::shared_ptr<const SrtReadTrace::CopyNodes> copyNodes;
};

struct EmissionFailure {
    std::uint64_t codeAddress;
    BindingLayout layout;
    std::exception_ptr failure;
};

struct SourceEntry {
    AgcDriver::Mutex mutex;
    // The plan's user data analysis (UserDataKeyFor) and the user data copies of each variant's
    // results by variant id (UserDataPatchesFor), made on first use (under mutex).
    std::shared_ptr<const UserDataAnalysis> userData;
    std::vector<std::pair<std::uint64_t, std::shared_ptr<const std::vector<UserDataPatch>>>> patches;
    // The memory copies of each variant's results by variant id (MemoryCopiesFor), likewise.
    std::vector<std::pair<std::uint64_t, std::shared_ptr<const std::vector<MemoryCopy>>>> memoryCopies;
    // Both for results with a binding plan (RecompileResult::bindingPlan), by plan: the plans are
    // kept for the process, so a plan's address names it.
    std::vector<std::pair<const void*, std::shared_ptr<const std::vector<UserDataPatch>>>> planPatches;
    std::vector<std::pair<const void*, std::shared_ptr<const std::vector<MemoryCopy>>>> planMemoryCopies;
    // The code the entry was built for: the key carries only a hash of it, so a candidate entry is
    // accepted only when its code matches word for word. Owned here because the request's span
    // points into a registration the driver may replace while the entry lives on.
    std::vector<std::uint32_t> code;
    std::shared_ptr<const IrResourcePlan> plan;
    // A plan build that threw (an unsupported resource chain or control flow) is remembered and
    // rethrown: the front end ran every pass before failing, ~13 ms per dispatch of a shader the
    // title issues every frame (0x1048947300 at the intro video). APS5_NO_FAILURE_MEMO=1 rebuilds.
    std::exception_ptr planFailure;
    std::unique_ptr<IrProgram> program;
    std::vector<std::shared_ptr<const CompiledVariant>> variants;
    // The result memo (under mutex), allocated with its first insert.
    std::unique_ptr<ResultMemoTable> memo;
    std::uint64_t memoClock = 0;
    // Memo misses since the last hit (under mutex), for the insert bypass.
    std::uint32_t memoMissRun = 0;
    std::vector<EmissionFailure> emissionFailures;
};

namespace {

struct ResourceProgram {
    explicit ResourceProgram(const RecompileRequest& request, ShaderPreparationContext* preparation = nullptr) : program(std::make_unique<IrProgram>(PrepareResourceProgram(request, preparation))), plan(std::make_shared<const IrResourcePlan>(ResourceMaterializer{}.ExtractPlan(*program))) {}

    std::unique_ptr<IrProgram> program;
    std::shared_ptr<const IrResourcePlan> plan;
};

std::shared_ptr<const IrResourcePlan> makeResourcePlan(const RecompileRequest& request) {
    return ResourceProgram(request).plan;
}

struct SourceKeyHash {
    std::size_t operator()(const std::vector<std::uint64_t>& key) const {
        std::size_t hash = 0;
        for (const auto value : key) {
            hash ^= static_cast<std::size_t>(value) + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) + (hash >> 2u);
            if constexpr (sizeof(std::size_t) < sizeof(value)) hash ^= static_cast<std::size_t>(value >> 32u);
        }
        return hash;
    }
};

bool FailureMemo() {
    static const bool memo = std::getenv("APS5_NO_FAILURE_MEMO") == nullptr;
    return memo;
}

std::shared_ptr<SourceEntry> getSource(const RecompileRequest& request, ShaderPreparationContext* preparation = nullptr) {
    static std::shared_mutex mutex;
    // Entries whose code hashes alike share a bucket; the code comparison picks the right one.
    static std::unordered_map<std::vector<std::uint64_t>, std::vector<std::shared_ptr<SourceEntry>>, SourceKeyHash> sources;
    struct SourceKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, SourceKeyStorage>();
    RecompileCacheKey::Build(request, key);
    key.push_back(HostSubgroupSize(request));
    const auto find = [&]() -> std::shared_ptr<SourceEntry> {
        const auto found = sources.find(key);
        if (found == sources.end()) return nullptr;
        for (const auto& entry : found->second) {
            if (std::equal(entry->code.begin(), entry->code.end(), request.shader.code.begin(), request.shader.code.end())) return entry;
        }
        return nullptr;
    };
    std::shared_ptr<SourceEntry> source;
    {
        std::shared_lock lock(mutex);
        source = find();
    }
    if (source == nullptr) {
        std::unique_lock lock(mutex);
        source = find();
        if (source == nullptr) {
            source = std::make_shared<SourceEntry>();
            source->code.assign(request.shader.code.begin(), request.shader.code.end());
            auto& bucket = sources[key];
            if (!bucket.empty()) {
                // A second entry under one key is a code hash collision (or the unhashed key with
                // identical code, which cannot happen); each is reported under the profile switch.
                static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
                static std::uint64_t collisions = 0;
                ++collisions;
                if (profile) std::fprintf(stderr, "[recompile] source key collision %llu: %zu entries share a key (%zu code words)\n", static_cast<unsigned long long>(collisions), bucket.size() + 1, request.shader.code.size());
            }
            bucket.push_back(source);
        }
    }
    {
        std::lock_guard lock(source->mutex);
        if (source->plan == nullptr) {
            if (FailureMemo() && source->planFailure) std::rethrow_exception(source->planFailure);
            try {
                ResourceProgram resource(request, preparation);
                source->plan = std::move(resource.plan);
                source->program = std::move(resource.program);
            } catch (...) {
                if (FailureMemo()) source->planFailure = std::current_exception();
                throw;
            }
        }
    }
    return source;
}

std::array<std::uint32_t, 3> partialThreads(const RecompileRequest& request) {
    return request.context.compute ? request.context.compute->partialThreads : std::array<std::uint32_t, 3>{};
}

// The user word `value` copies (through bit casts and the lane read of a uniform value, as the walk
// evaluates them), or -1.
std::int32_t copiedUserWord(const IrResourcePlan& plan, IrValue* value) {
    auto* resolved = value != nullptr ? value->Resolve() : nullptr;
    while (resolved != nullptr && resolved->ArgumentCount() >= 1 && (resolved->Opcode() == IrOpcode::BitCastU32F32 || resolved->Opcode() == IrOpcode::BitCastF32U32 || resolved->Opcode() == IrOpcode::ReadFirstLane)) resolved = resolved->Argument(0)->Resolve();
    if (resolved == nullptr || resolved->Opcode() != IrOpcode::GetUserData || resolved->ArgumentCount() < 1) return -1;
    const auto reg = RegIndex(static_cast<ScalarReg>(resolved->Argument(0)->Register().index));
    if (reg < plan.userDataBase || reg - plan.userDataBase >= plan.userDataCount) return -1;
    return static_cast<std::int32_t>(reg - plan.userDataBase);
}

// Marks every user word the values reachable from `root` read (the walk follows a ReadConst into its
// slot's read); `unknown` when one is read outside the plan's user data. The values in `opaque` (a
// relocatable base's handles, findRelocatableBase) are not followed.
void markUserWords(const IrResourcePlan& plan, IrValue* root, std::vector<std::uint8_t>& read, std::unordered_set<const IrValue*>& visited, bool& unknown, const std::unordered_set<const IrValue*>* opaque = nullptr) {
    std::vector<IrValue*> stack;
    if (root != nullptr) stack.push_back(root);
    while (!stack.empty()) {
        auto* value = stack.back();
        stack.pop_back();
        if (value == nullptr) continue;
        value = value->Resolve();
        if (value == nullptr || !visited.insert(value).second) continue;
        if (opaque != nullptr && opaque->contains(value)) continue;
        if (value->Opcode() == IrOpcode::GetUserData) {
            if (value->ArgumentCount() < 1) {
                unknown = true;
                continue;
            }
            const auto reg = RegIndex(static_cast<ScalarReg>(value->Argument(0)->Register().index));
            if (reg < plan.userDataBase || reg - plan.userDataBase >= read.size()) unknown = true;
            else read[reg - plan.userDataBase] = 1;
            continue;
        }
        if (value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() > 1) {
            const auto* slot = value->Argument(1)->Resolve();
            if (slot != nullptr && slot->HasImmediate() && slot->ImmediateU32() < plan.srtReads.size()) stack.push_back(plan.srtReads[slot->ImmediateU32()].value);
        }
        for (std::size_t i = 0; i < value->ArgumentCount(); ++i) stack.push_back(value->Argument(i));
    }
}

bool MemoryCopiesEnabled() {
    static const bool enabled = std::getenv("APS5_NO_MEMORY_COPIES") == nullptr;
    return enabled;
}

bool RelocatableBasesEnabled() {
    static const bool enabled = std::getenv("APS5_NO_RELOCATED_STAGE_MEMO") == nullptr;
    return enabled;
}

// A plan's relocatable base candidate (UserDataKey::relocatableBase): the low user word of the pair
// every raw read of the plan addresses, each a LoadAddressU32 of GetAddressResource(word k, word
// k + 1) at an immediate offset; -1 when a raw read reads through anything else (a pointer read from
// memory, a buffer, another pair) or at a computed offset. The reads' handles go to `handles`, for
// the marking to leave opaque: whether something else reads the pair decides (analyzeUserData).
// APS5_NO_RELOCATED_STAGE_MEMO=1 finds none.
std::int32_t findRelocatableBase(const IrResourcePlan& plan, std::unordered_set<const IrValue*>& handles) {
    if (!RelocatableBasesEnabled() || !plan.srtPlanComplete || plan.requiresSpecializationMemory || plan.srtReads.empty()) return -1;
    std::int32_t base = -1;
    for (const auto& stored : plan.valueStorage) {
        const IrValue* value = stored.get();
        if (value == nullptr || value->IsIdentity()) continue;
        const auto opcode = value->Opcode();
        if ((opcode != IrOpcode::LoadAddressU32 && opcode != IrOpcode::ReadConstBuffer) || !Detail::IsRawRead(plan, *value)) continue;
        if (opcode != IrOpcode::LoadAddressU32 || value->ArgumentCount() < 2) return -1;
        auto* handle = value->Argument(0)->Resolve();
        const auto* offset = value->Argument(1)->Resolve();
        if (handle->Opcode() != IrOpcode::GetAddressResource || handle->ArgumentCount() < 2 || !offset->HasImmediate()) return -1;
        const auto low = copiedUserWord(plan, handle->Argument(0));
        const auto high = copiedUserWord(plan, handle->Argument(1));
        if (low < 0 || high != low + 1 || (base >= 0 && low != base)) return -1;
        base = low;
        handles.insert(handle);
    }
    return base;
}

// The SRT read slot `value` is (through bit casts and the lane read of a uniform value): a ReadConst
// of the slot or the slot's raw read itself, by `rawReads`; -1 for anything else.
std::int32_t srtReadSlot(const IrResourcePlan& plan, const std::unordered_map<const IrValue*, std::uint32_t>& rawReads, IrValue* value) {
    auto* resolved = value != nullptr ? value->Resolve() : nullptr;
    while (resolved != nullptr && resolved->ArgumentCount() >= 1 && (resolved->Opcode() == IrOpcode::BitCastU32F32 || resolved->Opcode() == IrOpcode::BitCastF32U32 || resolved->Opcode() == IrOpcode::ReadFirstLane)) resolved = resolved->Argument(0)->Resolve();
    if (resolved == nullptr) return -1;
    if (resolved->Opcode() == IrOpcode::ReadConst) {
        if (resolved->ArgumentCount() < 2 || resolved->Argument(1) == nullptr) return -1;
        const auto* slot = resolved->Argument(1)->Resolve();
        return slot != nullptr && slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < plan.srtReads.size() ? static_cast<std::int32_t>(slot->ImmediateU32()) : -1;
    }
    const auto found = rawReads.find(resolved);
    return found != rawReads.end() ? static_cast<std::int32_t>(found->second) : -1;
}

// A plan's memory copies (UserDataAnalysis::bufferBaseReads): a buffer whose V# word 0 is an SRT read
// that nothing else in the plan uses (no other descriptor dword, control flow condition, uniform fill
// value or read address reaches it, as Detail::ComputePureFlatSlots asks of a pure slot) and whose
// word 1 is a read as well. That word 0 is the buffer's base, which the walk never follows and the
// specialization only tests for zero together with word 1's low half: a capture whose words differ
// only there has the same result but for the copied word, in the buffer's binding and in the
// flattened SRT. None where the walk reads memory the trace does not see (ComputePureFlatSlots'
// conditions). APS5_NO_MEMORY_COPIES=1 finds none.
void analyzeMemoryCopies(const IrResourcePlan& plan, UserDataAnalysis& analysis) {
    analysis.bufferBaseReads.assign(plan.info.buffers.size(), {-1, -1});
    const auto& sources = plan.descriptorSources;
    if (!MemoryCopiesEnabled() || plan.srtReads.empty() || !plan.srtPlanComplete || plan.requiresSpecializationMemory) return;
    if (std::any_of(sources.begin(), sources.end(), [](const DescriptorSource& source) { return source.indirectImage.has_value(); })) return;
    if (plan.uniformFill.fill.kind != UniformFillKind::None && plan.uniformFill.fill.words > plan.uniformFill.values.size()) return;
    std::unordered_map<const IrValue*, std::uint32_t> rawReads;
    for (std::uint32_t slot = 0; slot < plan.srtReads.size(); ++slot) {
        if (plan.srtReads[slot].value != nullptr) rawReads.emplace(plan.srtReads[slot].value->Resolve(), slot);
    }
    // The candidates: per buffer, the reads its words 0 and 1 are; their sources' word 0 is left out
    // of the marking below (a copy is its only use).
    std::vector<std::array<std::int32_t, 2>> candidates(plan.info.buffers.size(), {-1, -1});
    std::vector<std::uint8_t> candidateSource(sources.size(), 0);
    for (std::size_t i = 0; i < plan.info.buffers.size(); ++i) {
        const auto sourceIndex = plan.info.buffers[i].source;
        if (sourceIndex >= sources.size() || sources[sourceIndex].dwordCount < 2) continue;
        const auto& source = sources[sourceIndex];
        const auto base = srtReadSlot(plan, rawReads, source.dwords[0]);
        const auto pair = srtReadSlot(plan, rawReads, source.dwords[1]);
        if (base < 0 || pair < 0) continue;
        candidates[i] = {base, pair};
        candidateSource[sourceIndex] = 1;
    }
    if (std::none_of(candidateSource.begin(), candidateSource.end(), [](std::uint8_t candidate) { return candidate != 0; })) return;
    // Every read reached from anything but a candidate's word 0.
    std::vector<std::uint8_t> used(plan.srtReads.size(), 0);
    std::unordered_set<const IrValue*> visited;
    std::vector<IrValue*> pending;
    const auto mark = [&](IrValue* root) {
        if (root == nullptr) return;
        pending.push_back(root);
        while (!pending.empty()) {
            IrValue* value = pending.back()->Resolve();
            pending.pop_back();
            if (value == nullptr || !visited.insert(value).second) continue;
            if (value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() >= 2 && value->Argument(1) != nullptr) {
                const IrValue* slot = value->Argument(1)->Resolve();
                if (slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < used.size()) {
                    used[slot->ImmediateU32()] = 1;
                    pending.push_back(plan.srtReads[slot->ImmediateU32()].value);
                }
            }
            if (const auto found = rawReads.find(value); found != rawReads.end()) used[found->second] = 1;
            for (std::size_t i = 0; i < value->ArgumentCount(); ++i) {
                if (value->Argument(i) != nullptr) pending.push_back(value->Argument(i));
            }
        }
    };
    for (std::size_t s = 0; s < sources.size(); ++s) {
        for (std::uint32_t d = 0; d < sources[s].dwordCount && d < sources[s].dwords.size(); ++d) {
            if (d != 0 || candidateSource[s] == 0) mark(sources[s].dwords[d]);
        }
    }
    for (const auto& block : plan.controlFlow) mark(block.condition);
    if (plan.uniformFill.fill.kind != UniformFillKind::None) {
        for (std::uint32_t i = 0; i < plan.uniformFill.fill.words; ++i) mark(plan.uniformFill.values[i]);
    }
    // The address cones: the arguments of every read, never the read itself.
    for (const auto& read : plan.srtReads) {
        if (read.value == nullptr) continue;
        IrValue* raw = read.value->Resolve();
        for (std::size_t i = 0; i < raw->ArgumentCount(); ++i) mark(raw->Argument(i));
    }
    SrtReadTrace::CopyNodes nodes;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto [base, pair] = candidates[i];
        if (base < 0 || used[static_cast<std::size_t>(base)] != 0) continue;
        analysis.bufferBaseReads[i] = {base, pair};
        for (const auto slot : {base, pair}) nodes.emplace_back(plan.srtReads[static_cast<std::size_t>(slot)].value->Resolve(), static_cast<std::uint32_t>(slot));
    }
    if (nodes.empty()) return;
    std::sort(nodes.begin(), nodes.end(), [](const auto& left, const auto& right) { return std::less<const IrValue*>{}(left.first, right.first) || (left.first == right.first && left.second < right.second); });
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    analysis.copyNodes = std::make_shared<const SrtReadTrace::CopyNodes>(std::move(nodes));
}

// The user data analysis of a plan. A buffer's or direct image's descriptor dword that is a copy of
// a user word is a copy; every other user word read anywhere in the plan (the other descriptor
// dwords, samplers, bindless tables and the buffers they index, the SRT reads and their addresses,
// control flow, the uniform fill) is read. The key keeps a read word whole, and of a copied one the
// bits the specialization decodes: a V#'s stride, swizzle, size and format (words 1 high, 2, 3, the
// base's zero-ness through bufferBases), a T#'s everything but its base (words 0 and 1 low byte).
std::shared_ptr<const UserDataAnalysis> analyzeUserData(const IrResourcePlan& plan) {
    auto analysis = std::make_shared<UserDataAnalysis>();
    const auto count = static_cast<std::size_t>(plan.userDataCount);
    std::vector<std::uint8_t> read(count, 0);
    bool unknown = false;
    std::unordered_set<const IrValue*> visited;
    std::unordered_set<const IrValue*> baseHandles;
    const auto base = findRelocatableBase(plan, baseHandles);
    const auto* opaque = base >= 0 ? &baseHandles : nullptr;
    const auto mark = [&](IrValue* root) { markUserWords(plan, root, read, visited, unknown, opaque); };
    const auto& sources = plan.descriptorSources;
    // Sources read as a whole: the samplers', the bindless tables' and the buffers they index.
    std::vector<std::uint8_t> whole(sources.size(), 0);
    for (const auto& sampler : plan.info.samplers) {
        if (sampler.source < whole.size()) whole[sampler.source] = 1;
        else unknown = true;
    }
    for (const auto& image : plan.info.images) {
        if (image.source >= sources.size()) {
            unknown = true;
            continue;
        }
        const auto& indirect = sources[image.source].indirectImage;
        if (!indirect.has_value()) continue;
        whole[image.source] = 1;
        for (const auto index : {indirect->materialSource, indirect->heapSource}) {
            if (index < whole.size()) whole[index] = 1;
            else unknown = true;
        }
    }
    for (std::size_t s = 0; s < sources.size(); ++s) {
        if (whole[s] == 0) continue;
        for (std::uint32_t d = 0; d < sources[s].dwordCount && d < sources[s].dwords.size(); ++d) mark(sources[s].dwords[d]);
    }
    const auto copies = [&](std::uint32_t sourceIndex, std::span<std::int32_t> out) {
        std::fill(out.begin(), out.end(), -1);
        if (sourceIndex >= sources.size()) {
            unknown = true;
            return;
        }
        const auto& source = sources[sourceIndex];
        for (std::uint32_t d = 0; d < source.dwordCount && d < source.dwords.size(); ++d) {
            const auto word = whole[sourceIndex] == 0 ? copiedUserWord(plan, source.dwords[d]) : -1;
            if (word >= 0 && d < out.size()) out[d] = word;
            else mark(source.dwords[d]);
        }
    };
    analysis->bufferCopies.resize(plan.info.buffers.size());
    for (std::size_t i = 0; i < plan.info.buffers.size(); ++i) copies(plan.info.buffers[i].source, analysis->bufferCopies[i]);
    analysis->imageCopies.resize(plan.info.images.size());
    for (std::size_t i = 0; i < plan.info.images.size(); ++i) {
        const auto source = plan.info.images[i].source;
        if (source < sources.size() && sources[source].indirectImage.has_value()) analysis->imageCopies[i].fill(-1);
        else copies(source, analysis->imageCopies[i]);
    }
    for (const auto& srtRead : plan.srtReads) mark(srtRead.value);
    for (const auto& block : plan.controlFlow) mark(block.condition);
    if (plan.uniformFill.fill.kind != UniformFillKind::None) {
        for (auto* value : plan.uniformFill.values) mark(value);
    }
    auto key = std::make_shared<UserDataKey>();
    key->keepBits.assign(count, 0u);
    for (std::size_t k = 0; k < count; ++k) {
        if (unknown || read[k] != 0) key->keepBits[k] = ~0u;
    }
    for (const auto& words : analysis->bufferCopies) {
        for (std::size_t d = 0; d < words.size(); ++d) {
            if (words[d] >= 0) key->keepBits[static_cast<std::size_t>(words[d])] |= d == 0 ? 0u : d == 1 ? 0xffff0000u : ~0u;
        }
        // The base's other half read from memory: whether it is zero is the pair's, so the word is kept.
        if (words[0] >= 0 && words[1] >= 0) key->bufferBases.emplace_back(static_cast<std::uint32_t>(words[0]), static_cast<std::uint32_t>(words[1]));
        else if (words[0] >= 0) key->keepBits[static_cast<std::size_t>(words[0])] = ~0u;
        else if (words[1] >= 0) key->keepBits[static_cast<std::size_t>(words[1])] = ~0u;
    }
    for (const auto& words : analysis->imageCopies) {
        for (std::size_t d = 0; d < words.size(); ++d) {
            if (words[d] >= 0) key->keepBits[static_cast<std::size_t>(words[d])] |= d == 0 ? 0u : d == 1 ? 0xffffff00u : ~0u;
        }
    }
    std::sort(key->bufferBases.begin(), key->bufferBases.end());
    key->bufferBases.erase(std::unique(key->bufferBases.begin(), key->bufferBases.end()), key->bufferBases.end());
    if (base >= 0) {
        // The pair relocates when the reads' handles were its only readers (the marking left them
        // opaque) and no descriptor copies it; else it is kept whole, as any read word.
        const auto pair = static_cast<std::size_t>(base);
        const auto copiesPair = [&](const auto& words) { return std::any_of(words.begin(), words.end(), [&](std::int32_t word) { return word == base || word == base + 1; }); };
        const bool copied = std::any_of(analysis->bufferCopies.begin(), analysis->bufferCopies.end(), copiesPair) || std::any_of(analysis->imageCopies.begin(), analysis->imageCopies.end(), copiesPair);
        if (!unknown && read[pair] == 0 && read[pair + 1] == 0 && !copied) key->relocatableBase = base;
        else key->keepBits[pair] = key->keepBits[pair + 1] = ~0u;
    }
    analysis->key = std::move(key);
    analyzeMemoryCopies(plan, *analysis);
    return analysis;
}

// The user data copies of a variant's results: its bindings in layout order (Populate's), the inline
// descriptor words the analysis found, the shader data or push words (the layout's user registers).
std::shared_ptr<const std::vector<UserDataPatch>> makeUserDataPatches(const UserDataAnalysis& analysis, const CompiledVariant& variant, std::size_t userWords) {
    auto patches = std::make_shared<std::vector<UserDataPatch>>();
    const auto& layout = variant.bindings.layout;
    const auto base = variant.info.userDataBase;
    for (std::size_t b = 0; b < layout.descriptors.size(); ++b) {
        const auto& logical = layout.descriptors[b];
        const auto binding = static_cast<std::uint32_t>(b);
        if (logical.kind == DescriptorBindingKind::Buffers) {
            for (std::size_t e = 0; e < logical.resources.size(); ++e) {
                const auto resource = logical.resources[e];
                if (resource >= analysis.bufferCopies.size()) continue;
                for (std::uint32_t d = 0; d < 4u; ++d) {
                    if (const auto word = analysis.bufferCopies[resource][d]; word >= 0) patches->push_back({binding, static_cast<std::uint32_t>(e * 4u + d), static_cast<std::uint32_t>(word)});
                }
            }
        } else if (logical.kind == DescriptorBindingKind::ShaderData) {
            for (std::size_t i = 0; i < layout.userDataRegisters.size(); ++i) {
                const auto reg = layout.userDataRegisters[i];
                if (reg >= base && reg - base < userWords) patches->push_back({binding, static_cast<std::uint32_t>(i), reg - base});
            }
        } else if (ImageBindingResourceClass(logical.kind) != ImageResourceClass::None) {
            // Direct images are 8 dwords each; table slots past the plan's images come from memory.
            for (std::size_t e = 0; e < logical.resources.size(); ++e) {
                const auto resource = logical.resources[e];
                if (resource >= analysis.imageCopies.size()) continue;
                for (std::uint32_t d = 0; d < 8u; ++d) {
                    if (const auto word = analysis.imageCopies[resource][d]; word >= 0) patches->push_back({binding, static_cast<std::uint32_t>(e * 8u + d), static_cast<std::uint32_t>(word)});
                }
            }
        }
    }
    if (layout.UsesPushData()) {
        for (std::size_t i = 0; i < layout.userDataRegisters.size(); ++i) {
            const auto reg = layout.userDataRegisters[i];
            if (reg >= base && reg - base < userWords) patches->push_back({UserDataPatch::PushConstants, static_cast<std::uint32_t>(i), reg - base});
        }
    }
    return patches;
}

// The memory copies of a variant's results (MemoryCopiesFor): each buffer binding element whose V#
// the analysis found a base read for, at its word 0, and each such read in the flattened SRT, at
// its flat offset. Empty without any.
std::vector<MemoryCopy> makeMemoryCopies(const UserDataAnalysis& analysis, const CompiledVariant& variant, const IrResourcePlan& plan) {
    std::vector<MemoryCopy> copies;
    const auto& layout = variant.bindings.layout;
    for (std::size_t b = 0; b < layout.descriptors.size(); ++b) {
        const auto& logical = layout.descriptors[b];
        if (logical.kind != DescriptorBindingKind::Buffers) continue;
        for (std::size_t e = 0; e < logical.resources.size(); ++e) {
            const auto resource = logical.resources[e];
            if (resource >= analysis.bufferBaseReads.size()) continue;
            const auto [base, pair] = analysis.bufferBaseReads[resource];
            if (base >= 0) copies.push_back({static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(e * 4u), static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(pair)});
        }
    }
    const auto buffers = copies.size();
    for (std::size_t b = 0; b < layout.descriptors.size(); ++b) {
        if (layout.descriptors[b].kind != DescriptorBindingKind::FlattenedSrt) continue;
        for (std::size_t c = 0; c < buffers; ++c) {
            const auto copy = copies[c];
            const auto flat = plan.srtReads[copy.slot].flatOffset;
            const bool listed = std::any_of(copies.begin() + static_cast<std::ptrdiff_t>(buffers), copies.end(), [&](const MemoryCopy& other) { return other.binding == b && other.word == flat; });
            if (!listed) copies.push_back({static_cast<std::uint32_t>(b), flat, copy.slot, copy.pair});
        }
    }
    return copies;
}

// As makeUserDataPatches, over a result's binding plan (RecompileResult::bindingPlan) rather than the
// variant's layout: the plan's bindings are the result's, each element with its resource.
std::shared_ptr<const std::vector<UserDataPatch>> makePlanUserDataPatches(const UserDataAnalysis& analysis, const CompiledVariant& variant, const DescriptorBindingPlan& plan, std::size_t userWords) {
    auto patches = std::make_shared<std::vector<UserDataPatch>>();
    const auto& layout = variant.bindings.layout;
    const auto base = variant.info.userDataBase;
    for (std::size_t b = 0; b < plan.bindings.size(); ++b) {
        const auto& entry = plan.bindings[b];
        const auto binding = static_cast<std::uint32_t>(b);
        const auto role = entry.descriptor.role;
        if (role == DescriptorRole::GuestBuffers) {
            for (std::size_t e = 0; e < entry.resources.size(); ++e) {
                const auto resource = entry.resources[e];
                if (resource >= analysis.bufferCopies.size()) continue;
                for (std::uint32_t d = 0; d < 4u; ++d) {
                    if (const auto word = analysis.bufferCopies[resource][d]; word >= 0) patches->push_back({binding, static_cast<std::uint32_t>(e * 4u + d), static_cast<std::uint32_t>(word)});
                }
            }
        } else if (role == DescriptorRole::ShaderData) {
            for (std::size_t i = 0; i < layout.userDataRegisters.size(); ++i) {
                const auto reg = layout.userDataRegisters[i];
                if (reg >= base && reg - base < userWords) patches->push_back({binding, static_cast<std::uint32_t>(i), reg - base});
            }
        } else if (role == DescriptorRole::GuestImages) {
            for (std::size_t e = 0; e < entry.resources.size(); ++e) {
                const auto resource = entry.resources[e];
                if (resource >= analysis.imageCopies.size()) continue;
                for (std::uint32_t d = 0; d < 8u; ++d) {
                    if (const auto word = analysis.imageCopies[resource][d]; word >= 0) patches->push_back({binding, static_cast<std::uint32_t>(e * 8u + d), static_cast<std::uint32_t>(word)});
                }
            }
        }
    }
    if (layout.UsesPushData()) {
        for (std::size_t i = 0; i < layout.userDataRegisters.size(); ++i) {
            const auto reg = layout.userDataRegisters[i];
            if (reg >= base && reg - base < userWords) patches->push_back({UserDataPatch::PushConstants, static_cast<std::uint32_t>(i), reg - base});
        }
    }
    return patches;
}

// As makeMemoryCopies, over a result's binding plan (see makePlanUserDataPatches).
std::vector<MemoryCopy> makePlanMemoryCopies(const UserDataAnalysis& analysis, const DescriptorBindingPlan& bindings, const IrResourcePlan& plan) {
    std::vector<MemoryCopy> copies;
    for (std::size_t b = 0; b < bindings.bindings.size(); ++b) {
        const auto& entry = bindings.bindings[b];
        if (entry.descriptor.role != DescriptorRole::GuestBuffers) continue;
        for (std::size_t e = 0; e < entry.resources.size(); ++e) {
            const auto resource = entry.resources[e];
            if (resource >= analysis.bufferBaseReads.size()) continue;
            const auto [base, pair] = analysis.bufferBaseReads[resource];
            if (base >= 0) copies.push_back({static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(e * 4u), static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(pair)});
        }
    }
    const auto buffers = copies.size();
    for (std::size_t b = 0; b < bindings.bindings.size(); ++b) {
        if (bindings.bindings[b].descriptor.role != DescriptorRole::FlattenedSrt) continue;
        for (std::size_t c = 0; c < buffers; ++c) {
            const auto copy = copies[c];
            const auto flat = plan.srtReads[copy.slot].flatOffset;
            const bool listed = std::any_of(copies.begin() + static_cast<std::ptrdiff_t>(buffers), copies.end(), [&](const MemoryCopy& other) { return other.binding == b && other.word == flat; });
            if (!listed) copies.push_back({static_cast<std::uint32_t>(b), flat, copy.slot, copy.pair});
        }
    }
    return copies;
}

std::uint64_t nextVariantId() {
    static std::atomic<std::uint64_t> variants{0};
    return variants.fetch_add(1, std::memory_order_relaxed) + 1;
}

CompiledVariant compileVariant(const RecompileRequest& request, IrProgram program, std::exception_ptr* emissionFailure = nullptr) try {
    const auto inputInfo = RequestInputInfo(request);
    constexpr DeadCodeEliminator deadCodeEliminator;
    constexpr ResourceMaterializer resourceMaterializer;
    const bool nativeSampleOffsets = request.target.nonConstantImageOffsets && std::find(request.target.supportedCapabilities.begin(), request.target.supportedCapabilities.end(), spv::CapabilityImageGatherExtended) != request.target.supportedCapabilities.end();
    resourceMaterializer.ApplyStaticInterface(program, nativeSampleOffsets);

    ResolveControlFlowIdentities(program);
    deadCodeEliminator.RemoveIdentities(program);
    deadCodeEliminator.Eliminate(program);

    constexpr ShaderInfoCollector shaderInfoCollector;
    shaderInfoCollector.Collect(program, inputInfo);

    constexpr BindingAllocator bindingAllocator;
    auto bindings = bindingAllocator.Allocate(program, request.layout);

    // The layout the emitter lays the program out for (see SingleLaneCandidate). Debug aid:
    // APS5_TRACE_SINGLE_LANE=1 prints each candidate's verdict.
    auto emitInfo = inputInfo;
    ShaderComputeInputInfo singleLane;
    auto hostSubgroupSize = HostSubgroupSize(request);
    if (SingleLaneCandidate(request) && inputInfo.compute != nullptr) {
        const auto verdict = AnalyzeSingleLane(program, *inputInfo.compute);
        static const bool trace = std::getenv("APS5_TRACE_SINGLE_LANE") != nullptr;
        if (trace) {
            const auto& threads = inputInfo.compute->threadsNum;
            const std::string at = verdict.at == nullptr ? std::string() : " at " + DescribeValue(verdict.at, 3);
            std::fprintf(stderr, "[single-lane] program 0x%llx (%ux%ux%u threads): %s%s%s\n", static_cast<unsigned long long>(request.shader.codeAddress), threads[0], threads[1], threads[2], verdict.eligible ? "one lane per invocation" : "two lanes per invocation: ", verdict.eligible ? "" : verdict.reason, at.c_str());
        }
        if (verdict.eligible) {
            singleLane = *inputInfo.compute;
            singleLane.hostSubgroupSize = 64u;
            emitInfo.compute = &singleLane;
            hostSubgroupSize = 64u;
        }
    }

    SpirvTargetOptions targetOptions {};
    targetOptions.vulkanVersion = request.target.vulkanVersion;
    targetOptions.spirvVersion = request.target.spirvVersion;
    targetOptions.subgroupSize = request.target.subgroupSize;
    targetOptions.bdaAbiVersion = request.target.bdaAbiVersion;
    targetOptions.supportedCapabilities = request.target.supportedCapabilities;
    targetOptions.supportedExtensions = request.target.supportedExtensions;
    targetOptions.nonConstantImageOffsets = request.target.nonConstantImageOffsets;
    targetOptions.narrowSubgroupClock = request.target.narrowSubgroupClock;
    targetOptions.float64DenormPreserve = request.target.float64DenormPreserve;
    targetOptions.robustBufferAccess2 = request.target.robustBufferAccess2;

    constexpr SpirvEmitter spirvEmitter;
    CompiledShaderArtifact result;
    result.variantId = nextVariantId();
    result.spirv = spirvEmitter.Emit(program, emitInfo, bindings, targetOptions);

    result.bdaAbiVersion = program.Info().usesDma || program.Info().usesFaultBuffer ? request.target.bdaAbiVersion : 0u;
    result.memoryOffsetDword = bindings.layout.memoryOffsetDword;
    result.shaderDataDwords = bindings.layout.ShaderDataDwords();
    result.imageMetadataDword = bindings.layout.ImageMetadataDword();
    result.runtimeImageCount = bindings.layout.runtimeImageCount;
    for (std::uint32_t index = 0; index < program.Info().images.size(); ++index) {
        if (program.Info().images[index].indirectRoot != ImageResource::NoIndirectImage) result.runtimeImageResources.push_back(index);
    }
    result.hostSubgroupSize = hostSubgroupSize;
    result.vertexOffsetSgpr = program.Info().vertexOffsetSgpr;
    result.instanceOffsetSgpr = program.Info().instanceOffsetSgpr;
    result.vertexOffsetShared = program.Info().vertexOffsetShared;
    result.instanceOffsetShared = program.Info().instanceOffsetShared;
    result.vertexOffsetConflict = program.Info().vertexOffsetConflict;
    result.instanceOffsetConflict = program.Info().instanceOffsetConflict;
    for (const auto& output : program.Info().outputs) {
        if (output.kind == StageOutputKind::Parameter) result.parameterExports.push_back(output.location);
    }
    if (request.shader.stage == ShaderStage::Fragment) {
        result.fragmentParameters = DescribeFragmentParameters(program, inputInfo);
        const auto& inputs = program.Info().inputs;
        const auto reads = [&](StageInputKind kind) { return std::any_of(inputs.begin(), inputs.end(), [&](const auto& input) { return input.kind == kind; }); };
        if (program.Metadata().barycentricEmulation) result.barycentricEmulation = {true, reads(StageInputKind::BaryCoordSmooth), reads(StageInputKind::BaryCoordNoPerspective)};
    }
    if (request.shader.stage == ShaderStage::Vertex || request.shader.stage == ShaderStage::Local) {
        if (inputInfo.vertex == nullptr) throw std::runtime_error("vertex input metadata is missing");
        for (const auto& input : program.Info().inputs) {
            if (input.kind != StageInputKind::Parameter) continue;
            if (input.location >= static_cast<std::uint32_t>(inputInfo.vertex->resourcesNum)) throw std::runtime_error("vertex attribute location exceeds resource count");
            result.vertexInputs.push_back({input.location, input.componentCount, inputInfo.vertex->resourcesDst[input.location].fetchIndex});
            if (inputInfo.vertex->fetchEmbedded) {
                for (const auto& block : program.Blocks()) {
                    for (const auto* instruction : block->Instructions()) {
                        if (instruction->Opcode() == IrOpcode::GetAttribute && instruction->Argument(0)->Resolve()->ImmediateU32() == input.location) result.vertexInputs.back().outputMask |= 1u << instruction->Argument(1)->Resolve()->ImmediateU32();
                    }
                }
            }
        }
    }

    const bool embedded = !result.vertexInputs.empty() && inputInfo.vertex != nullptr && inputInfo.vertex->fetchEmbedded;
    if (embedded) PrepareVertexInputSpecialization(result);
#if ANYPS5_ENABLE_SPIRV_TOOLS
    result.spirv = ValidateAndOptimizeSpirv(result.spirv, request.target.vulkanVersion, request.target.spirvVersion, request.target.nonConstantImageOffsets, !embedded);
    if (embedded) {
        std::array<std::uint32_t, ShaderVertexStageInfo::MaxResources> classes{};
        for (std::uint32_t kind = 1u; kind < 3u; ++kind) {
            classes.fill(kind);
            const auto specialized = SpecializeVertexInputTypes(result, classes);
            static_cast<void>(ValidateAndOptimizeSpirv(specialized, request.target.vulkanVersion, request.target.spirvVersion, request.target.nonConstantImageOffsets, false));
        }
    }
#endif

    return {request.layout, std::move(program).TakeCompiledInfo(), std::move(static_cast<CompiledBindingLayout&>(bindings)), std::move(result)};
} catch (const std::bad_alloc&) {
    throw;
} catch (...) {
    if (emissionFailure != nullptr) *emissionFailure = std::current_exception();
    throw;
}

struct SpecializedModule {
    std::uint64_t specializationId = 0;
    SharedSpirv spirv;
    std::vector<std::uint32_t> bindings;
    bool pushData = false;
};

std::shared_ptr<const SpecializedModule> buildSpecializedModule(const CompiledShaderArtifact& artifact, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants, const SpirvTarget& target) {
    auto source = SpecializeVertexInputTypes(artifact, classes);
    std::map<std::uint32_t, std::uint32_t> supplied;
    for (const auto& constant : constants) {
        if (!supplied.emplace(constant.id, constant.value).second) throw std::runtime_error("duplicate prepared specialization ID");
    }
    std::map<std::uint32_t, std::uint32_t> values;
    std::set<std::uint32_t> specializedIds;
    const auto& words = source.Words();
    if (words.size() < 5u || words[0] != spv::MagicNumber) throw std::runtime_error("invalid prepared specialization module");
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(words[cursor] & 0xffffu);
        if (count == 0u || count > words.size() - cursor) throw std::runtime_error("truncated prepared specialization instruction");
        if (op == spv::OpDecorate && count == 4u && words[cursor + 2u] == spv::DecorationSpecId) {
            const auto found = supplied.find(words[cursor + 3u]);
            if (found == supplied.end() || !values.emplace(words[cursor + 1u], found->second).second) throw std::runtime_error("missing or duplicate prepared specialization value");
            specializedIds.insert(found->first);
        }
        cursor += count;
    }
    std::vector<std::uint32_t> materialized(words.begin(), words.begin() + 5);
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(words[cursor] & 0xffffu);
        if (op == spv::OpSpecConstant) {
            if (count != 4u || !values.contains(words[cursor + 2u])) throw std::runtime_error("invalid prepared specialization constant");
            materialized.insert(materialized.end(), {(4u << 16u) | spv::OpConstant, words[cursor + 1u], words[cursor + 2u], values.at(words[cursor + 2u])});
        } else if (!(op == spv::OpDecorate && count == 4u && words[cursor + 2u] == spv::DecorationSpecId)) {
            materialized.insert(materialized.end(), words.begin() + cursor, words.begin() + cursor + count);
        }
        cursor += count;
    }
    materialized = SpecializeFragmentOutputs(SpecializeSpirv(materialized), constants, specializedIds);
#if ANYPS5_ENABLE_SPIRV_TOOLS
    materialized = ValidateAndOptimizeSpirv(materialized, target.vulkanVersion, target.spirvVersion, target.nonConstantImageOffsets, true, true);
#endif
    std::map<std::uint32_t, std::uint32_t> descriptorVariables;
    std::set<std::uint32_t> usedDescriptors;
    for (std::size_t cursor = 5; cursor < materialized.size();) {
        const auto count = materialized[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(materialized[cursor] & 0xffffu);
        if (op == spv::OpDecorate && count == 4u && materialized[cursor + 2u] == spv::DecorationBinding) descriptorVariables.emplace(materialized[cursor + 1u], materialized[cursor + 3u]);
        if ((op == spv::OpAccessChain || op == spv::OpInBoundsAccessChain || op == spv::OpPtrAccessChain || op == spv::OpInBoundsPtrAccessChain || op == spv::OpLoad || op == spv::OpCopyObject) && count >= 4u) usedDescriptors.insert(materialized[cursor + 3u]);
        cursor += count;
    }
    const auto unused = [&](std::uint32_t id) { return descriptorVariables.contains(id) && !usedDescriptors.contains(id); };
    std::vector<std::uint32_t> compact(materialized.begin(), materialized.begin() + 5);
    for (std::size_t cursor = 5; cursor < materialized.size();) {
        const auto count = materialized[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(materialized[cursor] & 0xffffu);
        if (op == spv::OpEntryPoint) {
            const auto start = compact.size();
            std::size_t interfaceIndex = 3u;
            for (; interfaceIndex < count; ++interfaceIndex) {
                const auto word = materialized[cursor + interfaceIndex];
                if ((word & 0xffu) == 0u || (word & 0xff00u) == 0u || (word & 0xff0000u) == 0u || (word & 0xff000000u) == 0u) { ++interfaceIndex; break; }
            }
            compact.insert(compact.end(), materialized.begin() + cursor, materialized.begin() + cursor + interfaceIndex);
            for (auto index = interfaceIndex; index < count; ++index) if (!unused(materialized[cursor + index])) compact.push_back(materialized[cursor + index]);
            compact[start] = (static_cast<std::uint32_t>(compact.size() - start) << 16u) | spv::OpEntryPoint;
        } else if (!((op == spv::OpVariable && unused(materialized[cursor + 2u])) || ((op == spv::OpDecorate || op == spv::OpName) && unused(materialized[cursor + 1u])))) {
            compact.insert(compact.end(), materialized.begin() + cursor, materialized.begin() + cursor + count);
        }
        cursor += count;
    }
    materialized = std::move(compact);
#if ANYPS5_ENABLE_SPIRV_TOOLS
    materialized = ValidateAndOptimizeSpirv(materialized, target.vulkanVersion, target.spirvVersion, target.nonConstantImageOffsets, false);
#endif
    auto module = std::make_shared<SpecializedModule>();
    std::map<std::uint32_t, std::uint32_t> bindingNumbers;
    for (std::size_t cursor = 5; cursor < materialized.size();) {
        const auto count = materialized[cursor] >> 16u;
        const auto op = static_cast<spv::Op>(materialized[cursor] & 0xffffu);
        if (op == spv::OpDecorate && count == 4u && materialized[cursor + 2u] == spv::DecorationBinding) bindingNumbers.emplace(materialized[cursor + 1u], materialized[cursor + 3u]);
        if (op == spv::OpVariable && count >= 4u) {
            if (const auto found = bindingNumbers.find(materialized[cursor + 2u]); found != bindingNumbers.end()) module->bindings.push_back(found->second);
            module->pushData |= materialized[cursor + 3u] == spv::StorageClassPushConstant;
        }
        cursor += count;
    }
    module->spirv = std::move(materialized);
    module->specializationId = constants.empty() ? 0u : nextVariantId();
    return module;
}

struct SpecializedModuleEntry {
    std::once_flag ready;
    std::exception_ptr failure;
    std::shared_ptr<const SpecializedModule> module;
};

struct PreparedModuleEntry {
    std::once_flag ready;
    std::exception_ptr failure;
    std::shared_ptr<const SpecializedModule> module;
    DescriptorBindingPlan bindings;
};

struct PreparedBindingPlan {
    std::once_flag ready;
    std::exception_ptr failure;
    DescriptorBindingPlan bindings;
    std::shared_mutex mutex;
    std::map<std::vector<std::uint32_t>, std::shared_ptr<PreparedModuleEntry>> modules;
};

std::shared_ptr<PreparedBindingPlan> preparedBindingPlan(const CompiledVariant& variant, const ResourceSnapshot& snapshot, std::span<const std::uint8_t> exports) {
    struct BindingPlanKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, BindingPlanKeyStorage>();
    key.clear();
    key.push_back(variant.artifact.variantId);
    for (std::size_t index = 0; index < variant.info.info.buffers.size(); ++index) {
        const auto& descriptor = snapshot.buffers.at(index);
        key.push_back(descriptor.dwordCount);
        key.push_back(descriptor.dwords[1] & 0xffff0000u);
        key.push_back(descriptor.dwords[3]);
        const bool present = descriptor.dwords[2] != 0u && (descriptor.dwords[0] != 0u || (descriptor.dwords[1] & 0xffffu) != 0u);
        key.push_back(present ? 4u | (descriptor.dwords[0] & 3u) : 0u);
    }
    for (std::size_t index = 0; index < variant.info.info.images.size(); ++index) {
        const auto& descriptor = snapshot.images.at(index);
        key.push_back(descriptor.dwordCount);
        key.push_back(descriptor.dwords[0] != 0u || (descriptor.dwords[1] & 0xffu) != 0u);
        key.push_back(descriptor.dwords[1] & ~0xffu);
        for (std::size_t word = 2; word < 8; ++word) key.push_back(descriptor.dwords[word]);
    }
    for (std::size_t index = 0; index < variant.info.info.samplers.size(); ++index) {
        const auto& descriptor = snapshot.samplers.at(index);
        key.push_back(descriptor.dwordCount);
        for (std::size_t word = 0; word < 4; ++word) key.push_back(descriptor.dwords[word]);
    }
    key.push_back(exports.size());
    for (const auto mapping : exports) key.push_back(mapping);
    static std::shared_mutex mutex;
    static std::map<std::vector<std::uint64_t>, std::shared_ptr<PreparedBindingPlan>> plans;
    std::shared_ptr<PreparedBindingPlan> plan;
    {
        std::shared_lock lock(mutex);
        if (const auto found = plans.find(key); found != plans.end()) plan = found->second;
    }
    if (plan == nullptr) {
        std::unique_lock lock(mutex);
        const auto found = plans.find(key);
        plan = found != plans.end() ? found->second : plans.emplace(key, std::make_shared<PreparedBindingPlan>()).first->second;
    }
    std::call_once(plan->ready, [&] {
        try {
            plan->bindings = DescriptorBindingBuilder{}.Prepare(variant.bindings.layout, variant.info.info, variant.info.stage, snapshot, exports);
        } catch (...) {
            plan->failure = std::current_exception();
        }
    });
    if (plan->failure) std::rethrow_exception(plan->failure);
    return plan;
}

std::shared_ptr<const SpecializedModule> specializeModule(const CompiledShaderArtifact& artifact, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants, const SpirvTarget& target) {
    struct ModuleKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, ModuleKeyStorage>();
    key.clear();
    key.push_back(artifact.variantId);
    for (const auto& constant : constants) key.push_back((static_cast<std::uint64_t>(constant.id) << 32u) | constant.value);
    static std::shared_mutex mutex;
    static std::map<std::vector<std::uint64_t>, std::shared_ptr<SpecializedModuleEntry>> modules;
    std::shared_ptr<SpecializedModuleEntry> entry;
    {
        std::shared_lock lock(mutex);
        if (const auto found = modules.find(key); found != modules.end()) entry = found->second;
    }
    if (entry == nullptr) {
        std::unique_lock lock(mutex);
        const auto found = modules.find(key);
        entry = found != modules.end() ? found->second : modules.emplace(key, std::make_shared<SpecializedModuleEntry>()).first->second;
    }
    std::call_once(entry->ready, [&] {
        try {
            entry->module = buildSpecializedModule(artifact, classes, constants, target);
        } catch (...) {
            entry->failure = std::current_exception();
        }
    });
    if (entry->failure) std::rethrow_exception(entry->failure);
    return entry->module;
}

RecompileResult materializeResult(const CompiledVariant& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot) {
    RecompileResult result;
    const auto& artifact = variant.artifact;
    static_cast<CompiledShaderArtifact&>(result) = {
        .memoryOffsetDword = artifact.memoryOffsetDword,
        .shaderDataDwords = artifact.shaderDataDwords,
        .imageMetadataDword = artifact.imageMetadataDword,
        .runtimeImageCount = artifact.runtimeImageCount,
        .runtimeImageResources = artifact.runtimeImageResources,
        .bdaAbiVersion = artifact.bdaAbiVersion,
        .runtimeAbiVersion = artifact.runtimeAbiVersion,
        .vertexInputs = artifact.vertexInputs,
        .vertexOffsetSgpr = artifact.vertexOffsetSgpr,
        .instanceOffsetSgpr = artifact.instanceOffsetSgpr,
        .vertexOffsetShared = artifact.vertexOffsetShared,
        .instanceOffsetShared = artifact.instanceOffsetShared,
        .vertexOffsetConflict = artifact.vertexOffsetConflict,
        .instanceOffsetConflict = artifact.instanceOffsetConflict,
        .hostSubgroupSize = artifact.hostSubgroupSize,
        .parameterExports = artifact.parameterExports,
        .fragmentParameters = artifact.fragmentParameters,
        .barycentricEmulation = artifact.barycentricEmulation,
        .variantId = artifact.variantId
    };
    const auto plan = preparedBindingPlan(variant, snapshot, request.context.pixel ? std::span<const std::uint8_t>(request.context.pixel->targetExportMapping) : std::span<const std::uint8_t>{});
    const DescriptorBindingPlan* bindingPlan = &plan->bindings;
    std::shared_ptr<PreparedModuleEntry> entry;
    struct LocalModuleKeyStorage {};
    auto& moduleKey = HostThreadLocal<std::vector<std::uint32_t>, LocalModuleKeyStorage>();
    moduleKey.clear();
    if (variant.bindings.layout.UsesPushData()) moduleKey.push_back(request.layout.pushConstantOffsetBytes / 4u);
    if (request.context.pixel) {
        for (const auto packing : request.context.pixel->targetExportPacking) moduleKey.push_back(static_cast<std::uint32_t>(packing));
        moduleKey.push_back(request.context.pixel->dualSourceBlend ? 1u : 0u);
    }
    result.vertexAttributes.reserve(result.vertexInputs.size());
    std::array<std::uint32_t, ShaderVertexStageInfo::MaxResources> vertexClasses{};
    for (const auto& input : result.vertexInputs) {
        if (!request.context.vertex || input.location >= request.context.vertex->resourcesNum || input.location >= request.context.vertex->resources.size()) throw std::runtime_error("Shader cache: invalid vertex attribute metadata");
        const auto& vertex = *request.context.vertex;
        const auto& resource = vertex.resources[input.location];
        const auto fetchIndex = vertex.fetchEmbedded ? vertex.resourcesDst[input.location].fetchIndex : input.fetchIndex;
        result.vertexAttributes.push_back({input.location, input.components, resource, fetchIndex});
        if (!artifact.vertexInputPatches.empty()) {
            const auto numeric = VertexInputNumericClass(static_cast<IrBufferFormat>((resource.fields[3] >> 12u) & 0x7fu));
            if (numeric == IrTextureNumericClass::Unsupported) throw std::runtime_error("unsupported prepared vertex format");
            const auto kind = numeric == IrTextureNumericClass::Float ? 0u : numeric == IrTextureNumericClass::Sint ? 1u : 2u;
            vertexClasses.at(input.location) = kind;
            std::uint32_t selectors = kind << 12u;
            auto& formatComponents = result.vertexAttributes.back().formatComponents;
            formatComponents = 1u;
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const auto selector = (input.outputMask & (1u << component)) != 0u ? (resource.fields[3] >> (component * 3u)) & 7u : 0u;
                if (selector == 2u || selector == 3u) throw std::runtime_error("reserved prepared vertex component selector");
                if (selector >= 4u) formatComponents = std::max(formatComponents, selector - 3u);
                selectors |= selector << (component * 3u);
            }
            moduleKey.push_back(selectors);
        }
    }
    if (!plan->bindings.specialization.empty() || variant.bindings.layout.UsesPushData() || !artifact.vertexInputPatches.empty() || request.context.pixel) {
        {
            std::shared_lock lock(plan->mutex);
            if (const auto found = plan->modules.find(moduleKey); found != plan->modules.end()) entry = found->second;
        }
        if (entry == nullptr) {
            std::unique_lock lock(plan->mutex);
            const auto found = plan->modules.find(moduleKey);
            entry = found != plan->modules.end() ? found->second : plan->modules.emplace(moduleKey, std::make_shared<PreparedModuleEntry>()).first->second;
        }
        const auto prepare = [&] {
            auto constants = plan->bindings.specialization;
            std::size_t index = 0;
            if (variant.bindings.layout.UsesPushData()) constants.push_back({PipelineSpecialization::PushDataOffset, moduleKey[index++]});
            if (request.context.pixel) {
                for (std::uint32_t target = 0; target < request.context.pixel->targetExportPacking.size(); ++target) constants.push_back({PipelineSpecialization::ExportPackingBase + target, moduleKey[index++]});
                constants.push_back({PipelineSpecialization::DualSourceBlend, moduleKey[index++]});
            }
            if (!artifact.vertexInputPatches.empty()) {
                for (const auto& input : result.vertexInputs) {
                    const auto selectors = moduleKey[index++];
                    const auto first = PipelineSpecialization::VertexBase + input.location * PipelineSpecialization::VertexWords;
                    for (std::uint32_t component = 0; component < 4u; ++component) constants.push_back({first + component, (selectors >> (component * 3u)) & 7u});
                    const auto kind = selectors >> 12u;
                    constants.push_back({first + 4u, kind == 0u ? 0x3f800000u : 1u});
                    constants.push_back({first + 5u, kind});
                }
            }
            const auto module = specializeModule(variant.artifact, vertexClasses, constants, request.target);
            if (variant.bindings.layout.UsesPushData() && !module->pushData) throw std::runtime_error("specialization removed the prepared push constant interface");
            auto selected = DescriptorBindingBuilder{}.Select(plan->bindings, module->bindings);
            entry->bindings = std::move(selected);
            entry->module = module;
        };
        std::call_once(entry->ready, [&] {
            try {
                prepare();
            } catch (...) {
                entry->failure = std::current_exception();
            }
        });
        if (entry->failure) std::rethrow_exception(entry->failure);
        const auto& module = entry->module;
        result.specializationId = module->specializationId;
        result.spirv = module->spirv;
        bindingPlan = &entry->bindings;
        result.bindingPlan = std::shared_ptr<const void>(entry, &entry->bindings);
    } else {
        result.spirv = artifact.spirv;
        result.bindingPlan = std::shared_ptr<const void>(plan, &plan->bindings);
    }
    BindingAllocationResult bindings;
    DescriptorBindingBuilder{}.Populate(bindings, variant.bindings, *bindingPlan, variant.info.userDataBase, snapshot, partialThreads(request));
    result.workgroupMemoryDwords = WorkgroupMemoryStrideDwords(variant.info.info);
    result.bindings = std::move(bindings.bindings);
    result.pushConstants = std::move(bindings.pushConstants);
    result.poisonedSrtReads = static_cast<std::uint32_t>(snapshot.srtPoison.size()) + snapshot.nullRootReads;
    return result;
}

bool sameLayout(const BindingLayout& left, const BindingLayout& right) {
    return left.descriptorSet == right.descriptorSet && left.firstBinding == right.firstBinding && left.pushConstantOffsetBytes == right.pushConstantOffsetBytes && left.pushConstantSizeBytes == right.pushConstantSizeBytes;
}

std::shared_ptr<const CompiledVariant> findOrCompileVariant(SourceEntry& source, const RecompileRequest& request, bool& cacheHit, ShaderPreparationContext* preparation = nullptr) {
    for (const auto& candidate : source.variants) {
        if (sameLayout(candidate->layout, request.layout)) {
            cacheHit = true;
            return candidate;
        }
    }
    for (const auto& failure : source.emissionFailures) {
        if (failure.codeAddress == request.shader.codeAddress && sameLayout(failure.layout, request.layout)) std::rethrow_exception(failure.failure);
    }
    cacheHit = false;
    const bool disk = ShaderDiskCache::Enabled() && !DebugProbeActive();
    std::vector<std::byte> diskKey;
    std::shared_ptr<const CompiledVariant> variant;
    if (disk) {
        ShaderDiskCache::BuildKey(request, CacheLayout(request), diskKey);
        CompiledVariant loaded;
        if (ShaderDiskCache::Load(diskKey, loaded)) {
            loaded.layout = request.layout;
            loaded.artifact.variantId = nextVariantId();
            variant = std::make_shared<const CompiledVariant>(std::move(loaded));
        }
    }
    if (variant == nullptr) {
        auto program = source.program != nullptr ? std::move(*source.program) : PrepareResourceProgram(request, preparation);
        source.program.reset();
        std::exception_ptr emissionFailure;
        try {
            variant = std::make_shared<const CompiledVariant>(compileVariant(request, std::move(program), &emissionFailure));
        } catch (...) {
            if (emissionFailure != nullptr && FailureMemo()) source.emissionFailures.push_back({request.shader.codeAddress, request.layout, emissionFailure});
            throw;
        }
        if (disk) ShaderDiskCache::Store(std::move(diskKey), variant);
    }
    source.program.reset();
    source.variants.push_back(variant);
    return variant;
}

RecompileResult materializeVariant(SourceEntry& source, const RecompileRequest& request, const ResourceSnapshot& snapshot) {
    std::shared_ptr<const CompiledVariant> variant;
    bool cacheHit = false;
    {
        std::lock_guard lock(source.mutex);
        variant = findOrCompileVariant(source, request, cacheHit);
    }
    auto result = materializeResult(*variant, request, snapshot);
    result.cacheHit = cacheHit;
    return result;
}

RecompileResult RecompileImpl(const RecompileRequest& request) {
    static_cast<void>(RequestInputInfo(request));
    RequestMemoryView memory(request.context.memory);
    const auto runtime = memory.MakeRuntime(request.context.userData, request.shader.codeAddress);
    ResourceSnapshot snapshot;
    constexpr ResourceMaterializer materializer;
    if (!request.useCache) {
        auto program = PrepareResourceProgram(request);
        const auto plan = materializer.ExtractPlan(program);
        materializer.Materialize(plan, runtime, snapshot);
        const auto variant = compileVariant(request, std::move(program));
        return materializeResult(variant, request, snapshot);
    }
    const auto source = getSource(request);
    materializer.Materialize(*source->plan, runtime, snapshot);
    return materializeVariant(*source, request, snapshot);
}

// APS5_NO_RESULT_MEMO=1: every Recompile(request, capture) materializes its own result as before.
bool ResultMemo() {
    static const bool resultMemo = std::getenv("APS5_NO_RESULT_MEMO") == nullptr;
    return resultMemo;
}

// The memo set an index falls in, or null while the source's memo is empty.
ResultMemoEntry* memoSet(SourceEntry& source, std::uint64_t index) {
    if (source.memo == nullptr) return nullptr;
    return source.memo->data() + static_cast<std::size_t>((index ^ (index >> 32u)) % ResultMemoSets) * ResultMemoWays;
}

// A source whose memo missed MemoBypassMisses times in a row stops inserting its results (an insert
// allocated two nodes and evicted a cold result, destroyed on the worker), but for every
// MemoProbeMisses-th miss, so a snapshot that starts repeating is memoized again. Lookups go on.
// APS5_NO_MEMO_BYPASS=1 inserts every miss.
constexpr std::uint32_t MemoBypassMisses = 64;
constexpr std::uint32_t MemoProbeMisses = 16;

bool MemoBypass() {
    static const bool bypass = std::getenv("APS5_NO_MEMO_BYPASS") == nullptr;
    return bypass;
}

struct ResultMemoCounters {
    std::atomic<std::uint64_t> hits{0}, misses{0}, evictions{0}, bypassed{0}, populateNanoseconds{0};
    std::atomic<std::int64_t> lastReport{0};
};

ResultMemoCounters& resultMemoCounters() {
    static ResultMemoCounters counters;
    return counters;
}

void reportResultMemo() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& counters = resultMemoCounters();
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000ll || !counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const auto hits = counters.hits.exchange(0, std::memory_order_relaxed);
    const auto misses = counters.misses.exchange(0, std::memory_order_relaxed);
    const auto evictions = counters.evictions.exchange(0, std::memory_order_relaxed);
    const auto bypassed = counters.bypassed.exchange(0, std::memory_order_relaxed);
    const auto populate = counters.populateNanoseconds.exchange(0, std::memory_order_relaxed);
    std::fprintf(stderr, "[recompile] result memo (10 s): %llu hits, %llu misses (%.1f%% hits, %llu not inserted), Populate %.1f us per miss / %.1f ms in total, %llu evictions\n", static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses), hits + misses != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(hits + misses) : 0.0, static_cast<unsigned long long>(bypassed), misses != 0 ? static_cast<double>(populate) / 1000.0 / static_cast<double>(misses) : 0.0, static_cast<double>(populate) / 1e6, static_cast<unsigned long long>(evictions));
}

// Everything materializeResult reads besides the variant: the snapshot (the descriptor words, the
// flattened SRT, the user data, the uniform fill) and, for the vertex family, the V# table the
// attributes are resolved from.
std::uint64_t snapshotHash(const RecompileRequest& request, const ResourceSnapshot& snapshot) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= 0x100000001b3ull;
    };
    const auto mixWords = [&](std::span<const std::uint32_t> words) {
        mix(words.size());
        for (const auto word : words) mix(word);
    };
    const auto mixDescriptors = [&](const std::vector<DescriptorValue>& values) {
        mix(values.size());
        for (const auto& value : values) {
            mix(value.dwordCount);
            for (std::uint32_t i = 0; i < value.dwordCount && i < value.dwords.size(); ++i) mix(value.dwords[i]);
        }
    };
    mixDescriptors(snapshot.buffers);
    mixDescriptors(snapshot.images);
    mixDescriptors(snapshot.samplers);
    mixWords(snapshot.flattenedSrt);
    mixWords(snapshot.userData);
    mix(static_cast<std::uint64_t>(snapshot.uniformFill.kind));
    mix(snapshot.uniformFill.resource);
    for (const auto stride : snapshot.uniformFill.groupStride) mix(stride);
    mix(snapshot.uniformFill.words);
    mix(snapshot.uniformFill.value);
    for (const auto threads : partialThreads(request)) mix(threads);
    mix(request.layout.pushConstantOffsetBytes);
    if (request.context.pixel) {
        for (const auto mapping : request.context.pixel->targetExportMapping) mix(mapping);
        for (const auto packing : request.context.pixel->targetExportPacking) mix(static_cast<std::uint64_t>(packing));
        mix(request.context.pixel->dualSourceBlend);
    }
    if (request.context.vertex) {
        const auto& vertex = *request.context.vertex;
        const auto count = std::min<std::uint32_t>(vertex.resourcesNum, ShaderVertexStageInfo::MaxResources);
        mix(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            for (const auto field : vertex.resources[i].fields) mix(field);
            mix(vertex.resourcesDst[i].fetchIndex);
        }
    } else {
        mix(1ull << 32u);
    }
    return hash;
}

// The memo'd result of `source`'s variant for the snapshot (design13 R5): a hit returns the shared
// object, a miss materializes outside the source mutex and inserts (a concurrent miss's object is
// as good). `memoHit` reports the hit.
std::shared_ptr<const RecompileResult> materializePreparedMemoized(SourceEntry& source, const std::shared_ptr<const CompiledVariant>& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot, bool cacheHit, bool* memoHit) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto hash = snapshotHash(request, snapshot);
    std::uint64_t index = 0;
    bool insert = true;
    auto& counters = resultMemoCounters();
    {
        std::lock_guard lock(source.mutex);
        index = (variant->artifact.variantId * 0x9e3779b97f4a7c15ull) ^ hash;
        if (auto* set = memoSet(source, index)) {
            for (std::size_t way = 0; way < ResultMemoWays; ++way) {
                auto& entry = set[way];
                if (entry.result == nullptr || entry.variantId != variant->artifact.variantId || entry.hash != hash) continue;
                entry.used = ++source.memoClock;
                source.memoMissRun = 0;
                counters.hits.fetch_add(1, std::memory_order_relaxed);
                if (memoHit != nullptr) *memoHit = true;
                reportResultMemo();
                return entry.result;
            }
        }
        insert = !MemoBypass() || source.memoMissRun < MemoBypassMisses || source.memoMissRun % MemoProbeMisses == 0;
        ++source.memoMissRun;
    }
    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto result = std::make_shared<RecompileResult>(materializeResult(*variant, request, snapshot));
    result->cacheHit = cacheHit;
    if (profile) counters.populateNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()), std::memory_order_relaxed);
    counters.misses.fetch_add(1, std::memory_order_relaxed);
    std::shared_ptr<const RecompileResult> shared = std::move(result);
    if (!insert) {
        counters.bypassed.fetch_add(1, std::memory_order_relaxed);
        reportResultMemo();
        return shared;
    }
    {
        std::lock_guard lock(source.mutex);
        if (source.memo == nullptr) source.memo = std::make_unique<ResultMemoTable>();
        auto* set = memoSet(source, index);
        auto* victim = set;
        bool present = false;
        for (std::size_t way = 0; way < ResultMemoWays && !present; ++way) {
            auto& entry = set[way];
            if (entry.result != nullptr && entry.variantId == variant->artifact.variantId && entry.hash == hash) {
                entry.used = ++source.memoClock;
                shared = entry.result;
                present = true;
            } else if (entry.used < victim->used) {
                victim = &entry;
            }
        }
        if (!present) {
            if (victim->result != nullptr) counters.evictions.fetch_add(1, std::memory_order_relaxed);
            *victim = {variant->artifact.variantId, hash, shared, ++source.memoClock};
        }
    }
    reportResultMemo();
    return shared;
}

std::shared_ptr<const RecompileResult> materializeMemoized(SourceEntry& source, const RecompileRequest& request, const ResourceSnapshot& snapshot, bool* memoHit) {
    std::shared_ptr<const CompiledVariant> variant;
    bool cacheHit = false;
    {
        std::lock_guard lock(source.mutex);
        variant = findOrCompileVariant(source, request, cacheHit);
    }
    return materializePreparedMemoized(source, variant, request, snapshot, cacheHit, memoHit);
}

// The capture already resolved the source entry (stage input validation included) and materialized
// the request over exactly the words the driver captured, so neither is repeated here.
std::shared_ptr<const RecompileResult> RecompileImpl(const RecompileRequest& request, const ResourceCapture& capture, bool* memoHit) {
    if (!request.useCache || capture.source == nullptr) {
        auto program = PrepareResourceProgram(request);
        const auto variant = compileVariant(request, std::move(program));
        return std::make_shared<const RecompileResult>(materializeResult(variant, request, capture.snapshot));
    }
    if (!ResultMemo()) return std::make_shared<const RecompileResult>(materializeVariant(*capture.source, request, capture.snapshot));
    return materializeMemoized(*capture.source, request, capture.snapshot, memoHit);
}

template <typename Impl>
auto recompileReporting(const RecompileRequest& request, Impl&& impl) -> decltype(impl()) {
    try {
        return impl();
    } catch (const std::exception& e) {
        constexpr auto requestSerializer = RequestSerializer{};
        const auto inputInfo = "\nRecompileRequest:\n" + requestSerializer.Serialize(request);
        throw std::runtime_error(std::string("ShaderRecompiler::Recompile: ") + e.what() + inputInfo);
    } catch (...) {
        throw std::runtime_error("ShaderRecompiler::Recompile: unknown exception");
    }
}

}

std::shared_ptr<const IrResourcePlan> GetResourcePlan(const RecompileRequest& request) {
    static_cast<void>(RequestInputInfo(request));
    if (request.useCache) return getSource(request)->plan;
    return makeResourcePlan(request);
}

namespace {

// The raw reads of a source's memory copies (UserDataAnalysis::copyNodes), analyzing the plan first.
std::shared_ptr<const SrtReadTrace::CopyNodes> memoryCopyNodes(SourceEntry& source) {
    std::lock_guard lock(source.mutex);
    if (source.plan == nullptr) return nullptr;
    if (source.userData == nullptr) source.userData = analyzeUserData(*source.plan);
    return source.userData->copyNodes;
}

// The capture's materialization; with pure flat slots in the plan, or memory copies (MemoryCopiesFor),
// the walk's read addresses are traced into the capture (ResourceCapture::readTrace).
void materializeCapture(ResourceCapture& capture, const SrtRuntime& runtime) {
    const auto& plan = *capture.plan;
    auto copyNodes = capture.source != nullptr ? memoryCopyNodes(*capture.source) : nullptr;
    if (copyNodes == nullptr && std::none_of(plan.pureFlatSlots.begin(), plan.pureFlatSlots.end(), [](std::uint8_t pure) { return pure != 0u; })) {
        ResourceMaterializer{}.Materialize(plan, runtime, capture.snapshot);
        return;
    }
    SrtRuntime traced = runtime;
    capture.readTrace.copyNodes = std::move(copyNodes);
    traced.readTrace = &capture.readTrace;
    ResourceMaterializer{}.Materialize(plan, traced, capture.snapshot);
    auto& other = capture.readTrace.otherReads;
    std::sort(other.begin(), other.end());
    other.erase(std::unique(other.begin(), other.end()), other.end());
}

}

std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // Validates the stage inputs once per request, as GetResourcePlan and Recompile(request) do.
    static_cast<void>(RequestInputInfo(request));
    auto capture = std::make_shared<ResourceCapture>();
    if (request.useCache) {
        capture->source = getSource(request);
        capture->plan = capture->source->plan;
    } else {
        capture->plan = makeResourcePlan(request);
    }
    if (profile) capture->sourceNanoseconds = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
    materializeCapture(*capture, runtime);
    return capture;
}

std::shared_ptr<const SourceHandle> ResolveSource(const RecompileRequest& request) {
    if (!request.useCache) return nullptr;
    static_cast<void>(RequestInputInfo(request));
    return std::make_shared<const SourceHandle>(SourceHandle{getSource(request)});
}

std::shared_ptr<const SourceHandle> PrepareShader(const RecompileRequest& request) {
    return PrepareShader(request, nullptr);
}

std::shared_ptr<const SourceHandle> PrepareShader(const RecompileRequest& request, ShaderPreparationContext* preparation) {
    return recompileReporting(request, [&]() -> std::shared_ptr<const SourceHandle> {
        static_cast<void>(RequestInputInfo(request));
        auto handle = std::make_shared<SourceHandle>();
        handle->source = getSource(request, preparation);
        RecompileCacheKey::BuildInterface(request, handle->staticKey);
        handle->staticKey.push_back(HostSubgroupSize(request));
        bool cacheHit = false;
        std::lock_guard lock(handle->source->mutex);
        handle->artifact = findOrCompileVariant(*handle->source, request, cacheHit, preparation);
        return handle;
    });
}

bool MatchesPreparedShader(const RecompileRequest& request, const SourceHandle& handle) {
    struct PreparedKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, PreparedKeyStorage>();
    BuildPreparedShaderKey(request, key);
    return MatchesPreparedShader(request, handle, key);
}

void BuildPreparedShaderKey(const RecompileRequest& request, std::vector<std::uint64_t>& key) {
    RecompileCacheKey::BuildInterface(request, key);
    key.push_back(HostSubgroupSize(request));
}

bool MatchesPreparedShader(const RecompileRequest& request, const SourceHandle& handle, std::span<const std::uint64_t> key) {
    if (handle.source == nullptr || handle.artifact == nullptr) return false;
    const auto& layout = request.layout;
    const auto& prepared = handle.artifact->bindings.layout;
    if (layout.descriptorSet != handle.artifact->layout.descriptorSet || layout.firstBinding != handle.artifact->layout.firstBinding) return false;
    if (layout.pushConstantOffsetBytes % 4u != 0u || layout.pushConstantSizeBytes % 4u != 0u || layout.pushConstantOffsetBytes > NativePushConstantSize || layout.pushConstantSizeBytes > NativePushConstantSize - layout.pushConstantOffsetBytes) return false;
    const auto words = prepared.ShaderDataDwords();
    const bool usesPush = prepared.runtimeImageCount == 0u && words != 0u && words <= layout.pushConstantSizeBytes / 4u;
    if (prepared.UsesPushData() != usesPush) return false;
    if (!std::ranges::equal(key, handle.staticKey) || handle.source->code.size() != request.shader.code.size()) return false;
    return handle.source->code.data() == request.shader.code.data() || std::ranges::equal(handle.source->code, request.shader.code);
}

void ValidatePreparedInputs(const RecompileRequest& request) {
    if (request.shader.stage != ShaderStage::Compute && request.shader.stage != ShaderStage::Fragment) ValidateRequestInputs(request);
}

std::span<const std::uint32_t> GetPreparedCode(const SourceHandle& handle) {
    if (handle.source == nullptr || handle.artifact == nullptr) throw std::runtime_error("ShaderRecompiler: prepared artifact is missing");
    return handle.source->code;
}

PreparedShaderInvocation::PreparedShaderInvocation(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle) : request(request), handle(handle) {}

std::optional<PreparedShaderInvocation> PreparedShaderInvocation::TryCreate(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle) {
    struct PreparedKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, PreparedKeyStorage>();
    BuildPreparedShaderKey(request, key);
    return TryCreate(request, handle, key);
}

std::optional<PreparedShaderInvocation> PreparedShaderInvocation::TryCreate(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle, std::span<const std::uint64_t> key) {
    if (handle == nullptr || !MatchesPreparedShader(request, *handle, key)) return std::nullopt;
    ValidatePreparedInputs(request);
    return PreparedShaderInvocation(request, handle);
}

std::shared_ptr<const ResourceCapture> PreparedShaderInvocation::Capture(const SrtRuntime& runtime) const {
    auto capture = std::make_shared<ResourceCapture>();
    capture->source = handle->source;
    capture->plan = handle->source->plan;
    materializeCapture(*capture, runtime);
    return capture;
}

std::shared_ptr<const RecompileResult> PreparedShaderInvocation::Materialize(const ResourceCapture& capture, bool* memoHit) const {
    if (memoHit != nullptr) *memoHit = false;
    if (capture.source != handle->source || capture.plan != handle->source->plan) throw std::runtime_error("ShaderRecompiler: resource capture belongs to another prepared shader");
    if (request.useCache && ResultMemo()) return materializePreparedMemoized(*handle->source, handle->artifact, request, capture.snapshot, true, memoHit);
    auto result = std::make_shared<RecompileResult>(materializeResult(*handle->artifact, request, capture.snapshot));
    result->cacheHit = true;
    return result;
}

const CompiledShaderArtifact& GetPreparedArtifact(const SourceHandle& handle) {
    if (handle.artifact == nullptr) throw std::runtime_error("ShaderRecompiler: prepared artifact is missing");
    return handle.artifact->artifact;
}

std::shared_ptr<const RecompileResult> MaterializeShader(const RecompileRequest& request, const ResourceCapture& capture, const SourceHandle& handle) {
    if (!MatchesPreparedShader(request, handle)) throw std::runtime_error("ShaderRecompiler: prepared artifact does not match the static ABI");
    if (capture.source != handle.source || capture.plan != handle.source->plan) throw std::runtime_error("ShaderRecompiler: resource capture belongs to another prepared shader");
    if (request.useCache && ResultMemo()) return materializePreparedMemoized(*handle.source, handle.artifact, request, capture.snapshot, true, nullptr);
    auto result = std::make_shared<RecompileResult>(materializeResult(*handle.artifact, request, capture.snapshot));
    result->cacheHit = true;
    return result;
}

std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime, const SourceHandle& handle) {
    if (handle.source == nullptr) throw std::runtime_error("ShaderRecompiler: source handle is missing");
    if (handle.artifact != nullptr && !MatchesPreparedShader(request, handle)) throw std::runtime_error("ShaderRecompiler: prepared artifact does not match the static ABI");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // The whole vertex family (Vertex, Local, TC, TE, Mesh) validates V# fields the memo key does not cover.
    if (request.shader.stage != ShaderStage::Compute && request.shader.stage != ShaderStage::Fragment) ValidateRequestInputs(request);
    auto capture = std::make_shared<ResourceCapture>();
    capture->source = handle.source;
    capture->plan = handle.source->plan;
    if (profile) capture->sourceNanoseconds = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
    materializeCapture(*capture, runtime);
    return capture;
}

RecompileResult Recompile(const RecompileRequest& request) {
    return recompileReporting(request, [&] { return RecompileImpl(request); });
}

std::shared_ptr<const RecompileResult> Recompile(const RecompileRequest& request, const ResourceCapture& capture, bool* memoHit) {
    if (memoHit != nullptr) *memoHit = false;
    return recompileReporting(request, [&] { return RecompileImpl(request, capture, memoHit); });
}

std::shared_ptr<const UserDataKey> UserDataKeyFor(const SourceHandle& handle) {
    if (handle.source == nullptr) return nullptr;
    auto& source = *handle.source;
    std::lock_guard lock(source.mutex);
    if (source.plan == nullptr) return nullptr;
    if (source.userData == nullptr) source.userData = analyzeUserData(*source.plan);
    return source.userData->key;
}

std::shared_ptr<const std::vector<UserDataPatch>> UserDataPatchesFor(const SourceHandle& handle, const RecompileResult& result) {
    if (handle.source == nullptr || result.variantId == 0) return nullptr;
    auto& source = *handle.source;
    std::lock_guard lock(source.mutex);
    if (source.plan == nullptr) return nullptr;
    if (source.userData == nullptr) source.userData = analyzeUserData(*source.plan);
    // By the result's binding plan, whose bindings are the result's: one plan, one set of patches.
    if (const auto* bindings = static_cast<const DescriptorBindingPlan*>(result.bindingPlan.get())) {
        for (const auto& [plan, patches] : source.planPatches) {
            if (plan == bindings) return patches;
        }
        if (result.bindings.size() != bindings->bindings.size()) return nullptr;
        const auto make = [&](const CompiledVariant& variant) {
            auto patches = makePlanUserDataPatches(*source.userData, variant, *bindings, source.plan->userDataCount);
            source.planPatches.emplace_back(bindings, patches);
            return patches;
        };
        if (handle.artifact != nullptr && handle.artifact->artifact.variantId == result.variantId) return make(*handle.artifact);
        for (const auto& variant : source.variants) {
            if (variant->artifact.variantId == result.variantId) return make(*variant);
        }
        return nullptr;
    }
    // Otherwise by the result's pipeline variant: the specializations of one variant may bind differently.
    const auto id = result.PipelineVariantId();
    for (const auto& [variantId, patches] : source.patches) {
        if (variantId == id) return patches;
    }
    const auto make = [&](const CompiledVariant& variant) -> std::shared_ptr<const std::vector<UserDataPatch>> {
        // The patches name the bindings in layout order: a result its specialization took bindings
        // out of numbers them otherwise.
        if (result.bindings.size() != variant.bindings.layout.descriptors.size()) return nullptr;
        auto patches = makeUserDataPatches(*source.userData, variant, source.plan->userDataCount);
        source.patches.emplace_back(id, patches);
        return patches;
    };
    if (handle.artifact != nullptr && handle.artifact->artifact.variantId == result.variantId) return make(*handle.artifact);
    for (const auto& variant : source.variants) {
        if (variant->artifact.variantId == result.variantId) return make(*variant);
    }
    return nullptr;
}

std::shared_ptr<const std::vector<MemoryCopy>> MemoryCopiesFor(const SourceHandle& handle, const RecompileResult& result) {
    if (handle.source == nullptr || result.variantId == 0) return nullptr;
    auto& source = *handle.source;
    std::lock_guard lock(source.mutex);
    if (source.plan == nullptr) return nullptr;
    if (source.userData == nullptr) source.userData = analyzeUserData(*source.plan);
    if (source.userData->copyNodes == nullptr) return nullptr;
    // By the result's binding plan, as UserDataPatchesFor.
    if (const auto* bindings = static_cast<const DescriptorBindingPlan*>(result.bindingPlan.get())) {
        for (const auto& [plan, copies] : source.planMemoryCopies) {
            if (plan == bindings) return copies;
        }
        if (result.bindings.size() != bindings->bindings.size()) return nullptr;
        const bool own = (handle.artifact != nullptr && handle.artifact->artifact.variantId == result.variantId) || std::ranges::any_of(source.variants, [&](const auto& variant) { return variant->artifact.variantId == result.variantId; });
        if (!own) return nullptr;
        auto copies = makePlanMemoryCopies(*source.userData, *bindings, *source.plan);
        auto shared = copies.empty() ? nullptr : std::make_shared<const std::vector<MemoryCopy>>(std::move(copies));
        source.planMemoryCopies.emplace_back(bindings, shared);
        return shared;
    }
    // Otherwise by the result's pipeline variant, as UserDataPatchesFor.
    const auto id = result.PipelineVariantId();
    for (const auto& [variantId, copies] : source.memoryCopies) {
        if (variantId == id) return copies;
    }
    const auto make = [&](const CompiledVariant& variant) -> std::shared_ptr<const std::vector<MemoryCopy>> {
        if (result.bindings.size() != variant.bindings.layout.descriptors.size()) return nullptr;
        auto copies = makeMemoryCopies(*source.userData, variant, *source.plan);
        auto shared = copies.empty() ? nullptr : std::make_shared<const std::vector<MemoryCopy>>(std::move(copies));
        source.memoryCopies.emplace_back(id, shared);
        return shared;
    };
    if (handle.artifact != nullptr && handle.artifact->artifact.variantId == result.variantId) return make(*handle.artifact);
    for (const auto& variant : source.variants) {
        if (variant->artifact.variantId == result.variantId) return make(*variant);
    }
    return nullptr;
}

void PatchOverUserData(const RecompileRequest& request, RecompileResult& result, const std::vector<UserDataPatch>& patches) {
    recompileReporting(request, [&] {
        // As a capture does: the vertex family's inputs carry V# fields no key covers.
        if (request.shader.stage != ShaderStage::Compute && request.shader.stage != ShaderStage::Fragment) ValidateRequestInputs(request);
        const auto userData = request.context.userData;
        for (const auto& patch : patches) {
            if (patch.userWord >= userData.size()) throw std::runtime_error("user data patch exceeds the request's user data");
            const auto value = userData[patch.userWord];
            if (patch.binding == UserDataPatch::PushConstants) {
                if ((static_cast<std::size_t>(patch.word) + 1u) * sizeof(value) > result.pushConstants.size()) throw std::runtime_error("user data patch exceeds the push constants");
                std::memcpy(result.pushConstants.data() + static_cast<std::size_t>(patch.word) * sizeof(value), &value, sizeof(value));
            } else {
                result.bindings.at(patch.binding).guestDescriptor.at(patch.word) = value;
            }
        }
        for (auto& attribute : result.vertexAttributes) {
            if (!request.context.vertex || attribute.location >= request.context.vertex->resourcesNum) throw std::runtime_error("Shader cache: invalid vertex attribute metadata");
            attribute.resource = request.context.vertex->resources[attribute.location];
        }
        result.cacheHit = true;
    });
}

}
