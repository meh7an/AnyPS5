#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include "CacheKey.hpp"
#include "CompiledVariant.hpp"
#include "ShaderDiskCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include <list>
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
#include "Optimization/include/Optimization/DescriptorBindingBuilder.hpp"
#include "Optimization/include/Optimization/MaskedSelectEliminator.hpp"
#include "Optimization/include/Optimization/ReadLaneEliminator.hpp"
#include "Optimization/include/Optimization/RequestMemoryView.hpp"
#include "Optimization/include/Optimization/ResourceMaterializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/include/Optimization/ResourceTracker.hpp"
#include "Optimization/include/Optimization/ShaderInfoCollector.hpp"
#include "Optimization/include/Optimization/SrtWalker.hpp"
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

ShaderStageInputInfo RequestInputInfo(const RecompileRequest& request) {
    const auto* mesh = request.graphics && request.graphics->mesh ? &*request.graphics->mesh : nullptr;
    return BuildShaderStageInputInfo(toShaderStageKind(request.shader.stage), request.context, HostSubgroupSize(request), mesh);
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
    ValidateVertexStageInputs(stage, request.context, mesh);
}

}

IrProgram PrepareResourceProgram(const RecompileRequest& request) {
    const auto stageKind = toShaderStageKind(request.shader.stage);
    const auto inputInfo = RequestInputInfo(request);

    constexpr RdnaInstructionDecoder decoder;
    const auto decoded = decoder.Decode(request.shader.code);

    constexpr GraphBuilder graphBuilder;
    auto cfg = graphBuilder.Build(decoded);

    constexpr Structurizer structurizer;
    structurizer.Structurize(cfg);

    TranslateOptions translateOptions {};
    translateOptions.stage = stageKind;
    translateOptions.shaderHash = request.shader.codeAddress;
    translateOptions.waveSize = request.context.waveSize;
    translateOptions.userDataBaseRegister = request.context.userDataBaseRegister;
    translateOptions.userDataCount = static_cast<std::uint32_t>(request.context.userData.size());
    translateOptions.scratchDwords = request.context.compute.has_value() ? request.context.compute->scratchDwords : 0u;
    translateOptions.embeddedFetch = nullptr;
    translateOptions.fragmentShaderBarycentricEnabled = request.target.fragmentShaderBarycentricEnabled;
    translateOptions.inputInfo = inputInfo;

    constexpr InstructionTranslator translator;

    EmbeddedFetchPlan embeddedFetch;
    if ((stageKind == ShaderStageKind::Vertex || stageKind == ShaderStageKind::Local) && inputInfo.vertex != nullptr && inputInfo.vertex->fetchEmbedded) {
        constexpr EmbeddedVertexFetchAnalyzer embeddedFetchAnalyzer;
        embeddedFetch = embeddedFetchAnalyzer.Analyze(decoded, inputInfo.vertex->fetchAttribReg, inputInfo.vertex->fetchBufferReg, request.context.userDataBaseRegister, static_cast<std::uint32_t>(request.context.userData.size()), request.context.waveSize);
    }
    translateOptions.embeddedFetch = embeddedFetch.loads.empty() ? nullptr : &embeddedFetch;

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
    dumpIr("translate");

    constexpr SsaBuilder ssaBuilder;
    ssaBuilder.Rewrite(program);
    dumpIr("ssa");

    constexpr ConstantFolder constantFolder;
    constexpr DeadCodeEliminator deadCodeEliminator;

    constantFolder.Fold(program);
    ResolveControlFlowIdentities(program);
    deadCodeEliminator.RemoveIdentities(program);
    deadCodeEliminator.Eliminate(program);
    dumpIr("fold");

    constexpr ReadLaneEliminator readLaneEliminator;
    const auto readLaneStats = readLaneEliminator.Eliminate(program, translateOptions.waveSize);
    if (readLaneStats.rewrittenReads != 0u) {
        constantFolder.Fold(program);
        ResolveControlFlowIdentities(program);
        deadCodeEliminator.RemoveIdentities(program);
        deadCodeEliminator.Eliminate(program);
    }

    constexpr MaskedSelectEliminator maskedSelectEliminator;
    if (maskedSelectEliminator.Eliminate(program).removedSelects != 0u) {
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
};

struct EmissionFailure {
    std::uint64_t codeAddress;
    BindingLayout layout;
    ResourceSpecialization specialization;
    std::exception_ptr failure;
};

struct SourceEntry {
    AgcDriver::Mutex mutex;
    // The plan's user data analysis (UserDataKeyFor) and the user data copies of each variant's
    // results by variant id (UserDataPatchesFor), made on first use (under mutex).
    std::shared_ptr<const UserDataAnalysis> userData;
    std::vector<std::pair<std::uint64_t, std::shared_ptr<const std::vector<UserDataPatch>>>> patches;
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
    explicit ResourceProgram(const RecompileRequest& request) : program(std::make_unique<IrProgram>(PrepareResourceProgram(request))), plan(std::make_shared<const IrResourcePlan>(ResourceMaterializer{}.ExtractPlan(*program))) {}

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

std::shared_ptr<SourceEntry> getSource(const RecompileRequest& request) {
    static std::shared_mutex mutex;
    // Entries whose code hashes alike share a bucket; the code comparison picks the right one.
    static std::unordered_map<std::vector<std::uint64_t>, std::vector<std::shared_ptr<SourceEntry>>, SourceKeyHash> sources;
    struct SourceKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, SourceKeyStorage>();
    RecompileCacheKey::Build(request, key);
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
                ResourceProgram resource(request);
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
// slot's read); `unknown` when one is read outside the plan's user data.
void markUserWords(const IrResourcePlan& plan, IrValue* root, std::vector<std::uint8_t>& read, std::unordered_set<const IrValue*>& visited, bool& unknown) {
    std::vector<IrValue*> stack;
    if (root != nullptr) stack.push_back(root);
    while (!stack.empty()) {
        auto* value = stack.back();
        stack.pop_back();
        if (value == nullptr) continue;
        value = value->Resolve();
        if (value == nullptr || !visited.insert(value).second) continue;
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
    const auto mark = [&](IrValue* root) { markUserWords(plan, root, read, visited, unknown); };
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
    analysis->key = std::move(key);
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

std::uint64_t nextVariantId() {
    static std::atomic<std::uint64_t> variants{0};
    return variants.fetch_add(1, std::memory_order_relaxed) + 1;
}

CompiledVariant compileVariant(const RecompileRequest& request, IrProgram program, const ResourceSnapshot& resourceSnapshot, const ResourceSpecialization& resourceSpecialization, std::exception_ptr* emissionFailure = nullptr) {
    const auto inputInfo = RequestInputInfo(request);
    constexpr DeadCodeEliminator deadCodeEliminator;
    constexpr ResourceMaterializer resourceMaterializer;
    resourceMaterializer.Apply(program, resourceSpecialization);

    deadCodeEliminator.RemoveIdentities(program);
    deadCodeEliminator.Eliminate(program);

    constexpr ShaderInfoCollector shaderInfoCollector;
    shaderInfoCollector.Collect(program, inputInfo);

    constexpr BindingAllocator bindingAllocator;
    auto bindings = bindingAllocator.Allocate(program, request.layout);

    constexpr DescriptorBindingBuilder descriptorBindingBuilder;
    descriptorBindingBuilder.Populate(bindings, program, resourceSnapshot, partialThreads(request));

    SpirvTargetOptions targetOptions {};
    targetOptions.vulkanVersion = request.target.vulkanVersion;
    targetOptions.spirvVersion = request.target.spirvVersion;
    targetOptions.subgroupSize = request.target.subgroupSize;
    targetOptions.bdaAbiVersion = request.target.bdaAbiVersion;
    targetOptions.supportedCapabilities = request.target.supportedCapabilities;
    targetOptions.supportedExtensions = request.target.supportedExtensions;
    targetOptions.nonConstantImageOffsets = request.target.nonConstantImageOffsets;

    constexpr SpirvEmitter spirvEmitter;
    RecompileResult result;
    result.variantId = nextVariantId();
    try {
        result.spirv = spirvEmitter.Emit(program, inputInfo, bindings, targetOptions);
#if ANYPS5_ENABLE_SPIRV_TOOLS
        result.spirv = ValidateAndOptimizeSpirv(result.spirv, request.target.vulkanVersion, request.target.spirvVersion, request.target.nonConstantImageOffsets);
#endif
    } catch (const std::bad_alloc&) {
        throw;
    } catch (...) {
        if (emissionFailure != nullptr) *emissionFailure = std::current_exception();
        throw;
    }

    result.bdaAbiVersion = program.Info().usesDma ? request.target.bdaAbiVersion : 0u;
    result.memoryOffsetDword = bindings.layout.memoryOffsetDword;
    result.hostSubgroupSize = HostSubgroupSize(request);
    result.vertexOffsetSgpr = program.Info().vertexOffsetSgpr;
    result.instanceOffsetSgpr = program.Info().instanceOffsetSgpr;
    result.vertexOffsetShared = program.Info().vertexOffsetShared;
    result.instanceOffsetShared = program.Info().instanceOffsetShared;
    result.vertexOffsetConflict = program.Info().vertexOffsetConflict;
    result.instanceOffsetConflict = program.Info().instanceOffsetConflict;
    for (const auto& output : program.Info().outputs) {
        if (output.kind == StageOutputKind::Parameter) result.parameterExports.push_back(output.location);
    }
    if (request.shader.stage == ShaderStage::Fragment) result.fragmentParameters = DescribeFragmentParameters(program, inputInfo);
    if (request.shader.stage == ShaderStage::Vertex || request.shader.stage == ShaderStage::Local) {
        if (inputInfo.vertex == nullptr) throw std::runtime_error("vertex input metadata is missing");
        for (const auto& input : program.Info().inputs) {
            if (input.kind != StageInputKind::Parameter) continue;
            if (input.location >= static_cast<std::uint32_t>(inputInfo.vertex->resourcesNum)) throw std::runtime_error("vertex attribute location exceeds resource count");
            result.vertexAttributes.push_back({input.location, input.componentCount, {inputInfo.vertex->resources[input.location].fields}, inputInfo.vertex->resourcesDst[input.location].fetchIndex});
        }
    }

    result.bindings.clear();
    result.pushConstants.clear();
    for (auto& attribute : result.vertexAttributes) attribute.resource = {};
    bindings.bindings.clear();
    bindings.pushConstants.clear();
    return {resourceSpecialization, request.layout, std::move(program).TakeCompiledInfo(), std::move(bindings), std::move(result)};
}

RecompileResult materializeResult(const CompiledVariant& variant, const RecompileRequest& request, const ResourceSnapshot& snapshot) {
    auto result = variant.result;
    DescriptorBindingBuilder{}.Populate(variant.bindings.layout, variant.info.info, variant.info.stage, variant.info.userDataBase, snapshot, partialThreads(request), result.bindings, result.pushConstants);
    for (auto& attribute : result.vertexAttributes) {
        if (!request.context.vertex || attribute.location >= request.context.vertex->resourcesNum) throw std::runtime_error("Shader cache: invalid vertex attribute metadata");
        attribute.resource = request.context.vertex->resources[attribute.location];
    }
    return result;
}

bool sameLayout(const BindingLayout& left, const BindingLayout& right) {
    return left.descriptorSet == right.descriptorSet && left.firstBinding == right.firstBinding && left.pushConstantOffsetBytes == right.pushConstantOffsetBytes && left.pushConstantSizeBytes == right.pushConstantSizeBytes;
}

std::shared_ptr<const CompiledVariant> findOrCompileVariant(SourceEntry& source, const RecompileRequest& request, const ResourceSnapshot& snapshot, const ResourceSpecialization& specialization, bool& cacheHit) {
    for (const auto& candidate : source.variants) {
        if (sameLayout(candidate->layout, request.layout) && candidate->specialization == specialization) {
            cacheHit = true;
            return candidate;
        }
    }
    for (const auto& failure : source.emissionFailures) {
        if (failure.codeAddress == request.shader.codeAddress && sameLayout(failure.layout, request.layout) && failure.specialization == specialization) std::rethrow_exception(failure.failure);
    }
    cacheHit = false;
    const bool disk = ShaderDiskCache::Enabled() && !DebugProbeActive();
    std::vector<std::byte> diskKey;
    std::shared_ptr<const CompiledVariant> variant;
    if (disk) {
        ShaderDiskCache::BuildKey(request, HostSubgroupSize(request), specialization, diskKey);
        CompiledVariant loaded;
        if (ShaderDiskCache::Load(diskKey, loaded)) {
            loaded.specialization = specialization;
            loaded.layout = request.layout;
            loaded.result.variantId = nextVariantId();
            variant = std::make_shared<const CompiledVariant>(std::move(loaded));
        }
    }
    if (variant == nullptr) {
        auto program = source.program != nullptr ? std::move(*source.program) : PrepareResourceProgram(request);
        source.program.reset();
        std::exception_ptr emissionFailure;
        try {
            variant = std::make_shared<const CompiledVariant>(compileVariant(request, std::move(program), snapshot, specialization, &emissionFailure));
        } catch (...) {
            if (emissionFailure != nullptr && FailureMemo()) source.emissionFailures.push_back({request.shader.codeAddress, request.layout, specialization, emissionFailure});
            throw;
        }
        if (disk) ShaderDiskCache::Store(std::move(diskKey), variant);
    }
    source.program.reset();
    source.variants.push_back(variant);
    return variant;
}

// The cached variant of `source` for the specialization, compiled on first use.
RecompileResult materializeVariant(SourceEntry& source, const RecompileRequest& request, const ResourceSnapshot& snapshot, const ResourceSpecialization& specialization) {
    std::shared_ptr<const CompiledVariant> variant;
    bool cacheHit = false;
    {
        std::lock_guard lock(source.mutex);
        variant = findOrCompileVariant(source, request, snapshot, specialization, cacheHit);
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
    ResourceSpecialization specialization;
    constexpr ResourceMaterializer materializer;
    if (!request.useCache) {
        auto program = PrepareResourceProgram(request);
        const auto plan = materializer.ExtractPlan(program);
        materializer.Materialize(plan, runtime, snapshot, specialization);
        const auto variant = compileVariant(request, std::move(program), snapshot, specialization);
        return materializeResult(variant, request, snapshot);
    }
    const auto source = getSource(request);
    materializer.Materialize(*source->plan, runtime, snapshot, specialization);
    return materializeVariant(*source, request, snapshot, specialization);
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
    if (request.context.vertex) {
        const auto& vertex = *request.context.vertex;
        const auto count = std::min<std::uint32_t>(vertex.resourcesNum, ShaderVertexStageInfo::MaxResources);
        mix(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            for (const auto field : vertex.resources[i].fields) mix(field);
        }
    } else {
        mix(1ull << 32u);
    }
    return hash;
}

// The memo'd result of `source`'s variant for the snapshot (design13 R5): a hit returns the shared
// object, a miss materializes outside the source mutex and inserts (a concurrent miss's object is
// as good). `memoHit` reports the hit.
std::shared_ptr<const RecompileResult> materializeMemoized(SourceEntry& source, const RecompileRequest& request, const ResourceSnapshot& snapshot, const ResourceSpecialization& specialization, bool* memoHit) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    std::shared_ptr<const CompiledVariant> variant;
    bool cacheHit = false;
    const auto hash = snapshotHash(request, snapshot);
    std::uint64_t index = 0;
    bool insert = true;
    auto& counters = resultMemoCounters();
    {
        std::lock_guard lock(source.mutex);
        variant = findOrCompileVariant(source, request, snapshot, specialization, cacheHit);
        index = (variant->result.variantId * 0x9e3779b97f4a7c15ull) ^ hash;
        if (auto* set = memoSet(source, index)) {
            for (std::size_t way = 0; way < ResultMemoWays; ++way) {
                auto& entry = set[way];
                if (entry.result == nullptr || entry.variantId != variant->result.variantId || entry.hash != hash) continue;
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
            if (entry.result != nullptr && entry.variantId == variant->result.variantId && entry.hash == hash) {
                entry.used = ++source.memoClock;
                shared = entry.result;
                present = true;
            } else if (entry.used < victim->used) {
                victim = &entry;
            }
        }
        if (!present) {
            if (victim->result != nullptr) counters.evictions.fetch_add(1, std::memory_order_relaxed);
            *victim = {variant->result.variantId, hash, shared, ++source.memoClock};
        }
    }
    reportResultMemo();
    return shared;
}

// The capture already resolved the source entry (stage input validation included) and materialized
// the request over exactly the words the driver captured, so neither is repeated here.
std::shared_ptr<const RecompileResult> RecompileImpl(const RecompileRequest& request, const ResourceCapture& capture, bool* memoHit) {
    if (!request.useCache || capture.source == nullptr) {
        auto program = PrepareResourceProgram(request);
        const auto variant = compileVariant(request, std::move(program), capture.snapshot, capture.specialization);
        return std::make_shared<const RecompileResult>(materializeResult(variant, request, capture.snapshot));
    }
    if (!ResultMemo()) return std::make_shared<const RecompileResult>(materializeVariant(*capture.source, request, capture.snapshot, capture.specialization));
    return materializeMemoized(*capture.source, request, capture.snapshot, capture.specialization, memoHit);
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

// The capture's materialization; with pure flat slots in the plan the walk's read addresses are
// traced into the capture (ResourceCapture::readTrace).
void materializeCapture(ResourceCapture& capture, const SrtRuntime& runtime) {
    const auto& plan = *capture.plan;
    if (std::none_of(plan.pureFlatSlots.begin(), plan.pureFlatSlots.end(), [](std::uint8_t pure) { return pure != 0u; })) {
        ResourceMaterializer{}.Materialize(plan, runtime, capture.snapshot, capture.specialization);
        return;
    }
    SrtRuntime traced = runtime;
    traced.readTrace = &capture.readTrace;
    ResourceMaterializer{}.Materialize(plan, traced, capture.snapshot, capture.specialization);
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

std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime, const SourceHandle& handle) {
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
    for (const auto& [variantId, patches] : source.patches) {
        if (variantId == result.variantId) return patches;
    }
    for (const auto& variant : source.variants) {
        if (variant->result.variantId != result.variantId) continue;
        auto patches = makeUserDataPatches(*source.userData, *variant, source.plan->userDataCount);
        source.patches.emplace_back(result.variantId, patches);
        return patches;
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
