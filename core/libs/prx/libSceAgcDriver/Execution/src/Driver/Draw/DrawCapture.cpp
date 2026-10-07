#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>

namespace AgcDriver::DriverDetail {

namespace {

// APS5_PROFILE_DRAW_PHASES: compileDrawStage's own parts per stage, every 10 s.
struct StageTimes {
    enum Part { Handle, Capture, Regions, RecompileHit, RecompileMiss, Count };
    std::array<double, Count> us{};
    std::array<std::uint64_t, Count> calls{};
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

StageTimes& stageTimes() {
    static StageTimes times;
    return times;
}

void reportStageTimes() {
    auto& times = stageTimes();
    const auto now = std::chrono::steady_clock::now();
    if (now - times.lastReport < std::chrono::seconds(10)) return;
    times.lastReport = now;
    const auto avg = [&](StageTimes::Part part) { return times.calls[part] != 0 ? times.us[part] / static_cast<double>(times.calls[part]) : 0.0; };
    std::fprintf(stderr, "[stage-compile] %llu stages (10 s), us per stage: source handle %.2f, capture %.2f, regions %.2f, recompile %.2f on %llu memo hits / %.2f on %llu misses\n", static_cast<unsigned long long>(times.calls[StageTimes::Handle]), avg(StageTimes::Handle), avg(StageTimes::Capture), avg(StageTimes::Regions), avg(StageTimes::RecompileHit), static_cast<unsigned long long>(times.calls[StageTimes::RecompileHit]), avg(StageTimes::RecompileMiss), static_cast<unsigned long long>(times.calls[StageTimes::RecompileMiss]));
    times.us = {};
    times.calls = {};
}

// The stage memo (StageMemoEntry). APS5_NO_STAGE_MEMO=1 captures and recompiles every stage;
// APS5_VERIFY_STAGE_MEMO=1 captures and recompiles the stages it would serve too and compares the
// results (a difference is reported, and the capture's result used); APS5_STAGE_MEMO_ENTRIES caps
// the entries (default 8192, the least recently used go first).
bool StageMemoEnabled() {
    static const bool enabled = std::getenv("APS5_NO_STAGE_MEMO") == nullptr;
    return enabled;
}

bool VerifyStageMemo() {
    static const bool verify = std::getenv("APS5_VERIFY_STAGE_MEMO") != nullptr;
    return verify;
}

std::size_t StageMemoCapacity() {
    static const std::size_t capacity = [] {
        const char* text = std::getenv("APS5_STAGE_MEMO_ENTRIES");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 8192ull;
        return static_cast<std::size_t>(std::max<unsigned long long>(value, 1ull));
    }();
    return capacity;
}

// APS5_PROFILE_DRAW: the memo's outcomes, every 10 s on a [stage-memo] line.
struct StageMemoCounters {
    std::atomic<std::uint64_t> lookups{0}, hits{0}, absent{0}, mappings{0}, changed{0}, pending{0};
    std::atomic<std::uint64_t> inserts{0}, refusedPatches{0}, refusedMismatch{0}, refusedUnwatched{0}, refusedUnstable{0}, evictions{0};
    std::atomic<std::uint64_t> verified{0}, differed{0};
    std::atomic<std::int64_t> lastReport{0};
};

StageMemoCounters& stageMemoCounters() {
    static StageMemoCounters counters;
    return counters;
}

void reportStageMemo(std::size_t entries) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& counters = stageMemoCounters();
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load(std::memory_order_relaxed);
    if (now - last < 10000 || !counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const auto take = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    const auto lookups = take(counters.lookups);
    const auto hits = take(counters.hits);
    std::fprintf(stderr, "[stage-memo] %llu lookups (10 s): %llu hits (%.1f%%); misses: no entry %llu, mappings changed %llu, words changed %llu, pending results %llu; inserts %llu, refused: no patches %llu, patch mismatch %llu, unwatched %llu, unstable %llu; evictions %llu, %zu entries; verify: %llu compared, %llu differed\n", lookups, hits, lookups != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(lookups) : 0.0, take(counters.absent), take(counters.mappings), take(counters.changed), take(counters.pending), take(counters.inserts), take(counters.refusedPatches), take(counters.refusedMismatch), take(counters.refusedUnwatched), take(counters.refusedUnstable), take(counters.evictions), entries, take(counters.verified), take(counters.differed));
}

// A stage's memo key: the user data size, each user word's bits the source reads other than as a
// copy, each inline V#'s base zero-ness; hashed with the source and the push offset.
std::uint64_t stageMemoKey(const ShaderRecompiler::UserDataKey& key, const void* source, std::uint32_t pushOffset, std::span<const std::uint32_t> userData, std::vector<std::uint32_t>& words) {
    words.clear();
    words.reserve(key.keepBits.size() + key.bufferBases.size() + 1u);
    words.push_back(static_cast<std::uint32_t>(userData.size()));
    const auto word = [&](std::size_t k) { return k < userData.size() ? userData[k] : 0u; };
    for (std::size_t k = 0; k < key.keepBits.size(); ++k) words.push_back(word(k) & key.keepBits[k]);
    for (const auto& [low, high] : key.bufferBases) words.push_back(word(low) == 0u && (word(high) & 0xffffu) == 0u ? 1u : 0u);
    std::uint64_t hash = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= 0x100000001b3ull;
    };
    mix(reinterpret_cast<std::uintptr_t>(source));
    mix(pushOffset);
    for (const auto value : words) mix(value);
    return hash;
}

enum class StageMemoMiss { None, Absent, Mappings, Changed, Pending };

// Whether an entry's captured words are still the guest's: no mapping changed since before the
// capture, no write stamped over them since (CPU stores collected first; recorded GPU stores stamp
// when recorded), and no storage image results pending and no unit shadow live over them (bytes the
// memory does not hold yet).
StageMemoMiss stageMemoCurrent(const StageMemoEntry& entry) {
    if (entry.mappings != GuestAllocations::GuestAllocationsGeneration_nid_postfix()) return StageMemoMiss::Mappings;
    if (entry.runs.empty()) return StageMemoMiss::None;
    struct QueriesTag {};
    auto& queries = ThreadScratch<std::vector<GuestMemory::UnchangedQuery>, QueriesTag>();
    queries.clear();
    for (const auto& [begin, end] : entry.runs) {
        const auto bytes = static_cast<std::size_t>(end - begin);
        GuestMemory::CollectWrites(begin, bytes);
        queries.push_back({begin, bytes, entry.generation});
    }
    if (!GuestMemory::UnchangedSinceAll(queries)) return StageMemoMiss::Changed;
    if (Graphics::StorageTexture::AnyPendingOverlaps(entry.runs) || Graphics::AnyShadowedOverlaps(entry.runs)) return StageMemoMiss::Pending;
    return StageMemoMiss::None;
}

// The first difference between a memo result and a capture's result of the same stage (empty when
// they are equal), for APS5_VERIFY_STAGE_MEMO.
std::string stageMemoDifference(const ShaderRecompiler::RecompileResult& memo, const ShaderRecompiler::RecompileResult& captured) {
    char text[200];
    if (memo.variantId != captured.variantId) {
        std::snprintf(text, sizeof(text), "variant %llu, capture %llu", static_cast<unsigned long long>(memo.variantId), static_cast<unsigned long long>(captured.variantId));
        return text;
    }
    if (memo.bindings.size() != captured.bindings.size()) return "binding count";
    for (std::size_t b = 0; b < memo.bindings.size(); ++b) {
        const auto& left = memo.bindings[b];
        const auto& right = captured.bindings[b];
        if (left.role != right.role || left.kind != right.kind || left.count != right.count || left.guestDescriptor.size() != right.guestDescriptor.size()) {
            std::snprintf(text, sizeof(text), "binding %zu shape", b);
            return text;
        }
        for (std::size_t w = 0; w < left.guestDescriptor.size(); ++w) {
            if (left.guestDescriptor[w] == right.guestDescriptor[w]) continue;
            std::snprintf(text, sizeof(text), "binding %zu (role %d) word %zu: memo 0x%08x, capture 0x%08x", b, static_cast<int>(left.role), w, left.guestDescriptor[w], right.guestDescriptor[w]);
            return text;
        }
    }
    if (memo.pushConstants != captured.pushConstants) return "push constants";
    if (memo.vertexAttributes.size() != captured.vertexAttributes.size()) return "vertex attribute count";
    for (std::size_t a = 0; a < memo.vertexAttributes.size(); ++a) {
        if (memo.vertexAttributes[a].resource.fields == captured.vertexAttributes[a].resource.fields && memo.vertexAttributes[a].location == captured.vertexAttributes[a].location) continue;
        std::snprintf(text, sizeof(text), "vertex attribute %zu", a);
        return text;
    }
    return {};
}

}

std::shared_ptr<const ShaderRecompiler::RecompileResult> Driver::compileDrawStage(std::size_t i, std::uint32_t pushOffset, const QueueState& queue, const Submission& submission, const std::vector<DrawProgram>& programs, const Graphics::State& graphics, const ShaderRecompiler::ShaderPixelStageInfo& pixel, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, std::vector<ShaderRecompiler::MemoryRegion>& memory, const std::vector<ShaderRecompiler::LinkedProgram>& linked, const Pm4::DrawParameters& drawParameters, const std::shared_ptr<VulkanDevice>& localDevice, ShaderMemory& shaderMemory, std::vector<StageCapture>& stageCaptures, std::vector<bool>& recompiled, bool drawHit, bool wantRegions, const std::vector<std::shared_ptr<DispatchVariant>>& matched, const std::vector<std::vector<ShaderRecompiler::MemoryRegion>>& matchedRegions, bool profile, std::uint64_t dumpTarget, std::uint64_t dumpSlot1, std::uint64_t& captures, DrawPhaseTiming& phaseTiming, std::array<double, DrawDriverPhaseCount>& phaseMs, std::string& rejected) {
    using Stage = ShaderRecompiler::ShaderStage;
    phaseTiming.Phase(DrawRowVectors);
    const auto& program = programs[i];
    const auto waveSize = program.binary.stage == Stage::Fragment ? graphics.stages.fragmentWaveSize : graphics.stages.vertexWaveSize;
    ShaderRecompiler::RecompileRequest request{
        program.binary,
        {waveSize, program.firstUserSgpr, program.userData, std::nullopt, program.binary.stage == Stage::Fragment ? std::optional(pixel) : std::nullopt, vertexInfos[i], memory},
        localDevice->Target(),
        {0, 0, pushOffset, (graphics.stages.mesh ? ShaderRecompiler::MeshDrawPushOffsetBytes : Graphics::PipelinePushConstantBytes) - pushOffset},
        ShaderRecompiler::GraphicsCompileContext{program.firstUserSgpr, linked, graphics.stages.mesh, graphics.stages.tessellation, {drawParameters.indexAddress, drawParameters.indexCount, drawParameters.indexSize, drawParameters.instanceCount}}
    };
    const auto waitedBefore = traceCapSync() || profile ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
    static const bool subPhases = std::getenv("APS5_PROFILE_DRAW_PHASES") != nullptr;
    auto lap = subPhases ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto part = [&](StageTimes::Part which) {
        if (!subPhases) return;
        const auto now = std::chrono::steady_clock::now();
        stageTimes().us[which] += std::chrono::duration<double, std::micro>(now - lap).count();
        ++stageTimes().calls[which];
        lap = now;
    };
    const std::string* poisoned = nullptr;
    const auto handle = SourceHandleFor(*program.snapshot, program.codeOffset, localDevice->Serial(), request, false, FailureMemo() ? &poisoned : nullptr);
    part(StageTimes::Handle);
    if (handle == nullptr && poisoned != nullptr) {
        rejected = *poisoned;
        return nullptr;
    }
    auto& stageCapture = stageCaptures[i];
    stageCapture.forgetSerial = GuestMemory::ForgetSerial();
    stageCapture.pushOffset = pushOffset;
    stageCapture.memo.reset();
    const auto regionsDone = [&] {
        recompiled[i] = true;
        shaderMemory.Regions(memory);
        part(StageTimes::Regions);
        if (drawHit) {
            for (std::size_t j = 0; j < programs.size(); ++j) {
                if (matched[j] != nullptr && !recompiled[j]) memory.insert(memory.end(), matchedRegions[j].begin(), matchedRegions[j].end());
            }
        }
        request.context.memory = memory;
    };

    // The stage memo (StageMemoEntry): a key that repeats over unchanged captured words takes the
    // remembered result with this stage's user words copied in. Its words enter the draw's shader
    // memory as this stage's reads, so the draw's regions are those a capture would have made.
    static const bool memoize = StageMemoEnabled();
    std::shared_ptr<const ShaderRecompiler::UserDataKey> memoKey;
    if (memoize && handle != nullptr && dumpTarget == 0 && dumpSlot1 == 0 && !ShaderRecompiler::DebugProbeActive()) {
        // Per thread by source (a source entry lives as long as the process): no lock per stage.
        struct MemoKeysTag {};
        auto& keys = ThreadScratch<std::unordered_map<const void*, std::shared_ptr<const ShaderRecompiler::UserDataKey>>, MemoKeysTag>();
        auto& known = keys[handle->source.get()];
        if (known == nullptr) known = ShaderRecompiler::UserDataKeyFor(*handle);
        memoKey = known;
    }
    struct MemoKeyTag {};
    auto& memoWords = ThreadScratch<std::vector<std::uint32_t>, MemoKeyTag>();
    std::uint64_t memoHash = 0;
    const auto mappings = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    std::shared_ptr<const ShaderRecompiler::RecompileResult> memoResult;
    // Whether a capture of this stage is stored: always for a new key, at powers of two of a key's
    // misses in a row otherwise.
    bool memoStore = true;
    if (memoKey != nullptr) {
        auto& counters = stageMemoCounters();
        counters.lookups.fetch_add(1, std::memory_order_relaxed);
        memoHash = stageMemoKey(*memoKey, handle->source.get(), pushOffset, program.userData, memoWords);
        std::shared_ptr<const StageMemoEntry> entry;
        std::shared_ptr<ShaderRecompiler::RecompileResult> served;
        std::size_t entries = 0;
        auto miss = StageMemoMiss::Absent;
        {
            std::lock_guard lock(stageMemoMutex);
            entries = stageMemo.size();
            const auto found = stageMemo.find(memoHash);
            if (found != stageMemo.end() && found->second.entry->handle->source == handle->source && found->second.entry->pushOffset == pushOffset && found->second.entry->key == memoWords) {
                auto& slot = found->second;
                entry = slot.entry;
                stageMemoOrder.splice(stageMemoOrder.end(), stageMemoOrder, slot.order);
                miss = stageMemoCurrent(*entry);
                if (miss == StageMemoMiss::None) {
                    slot.misses = 0;
                    // Held by nothing but the slot: no draw is using it any more.
                    if (slot.served != nullptr && slot.served.use_count() == 1) served = slot.served;
                } else {
                    ++slot.misses;
                    memoStore = (slot.misses & (slot.misses - 1u)) == 0;
                }
            }
        }
        switch (miss) {
            case StageMemoMiss::None: break;
            case StageMemoMiss::Absent: counters.absent.fetch_add(1, std::memory_order_relaxed); break;
            case StageMemoMiss::Mappings: counters.mappings.fetch_add(1, std::memory_order_relaxed); break;
            case StageMemoMiss::Changed: counters.changed.fetch_add(1, std::memory_order_relaxed); break;
            case StageMemoMiss::Pending: counters.pending.fetch_add(1, std::memory_order_relaxed); break;
        }
        if (miss == StageMemoMiss::None) {
            const bool fresh = served == nullptr;
            if (fresh) served = std::make_shared<ShaderRecompiler::RecompileResult>(*entry->result);
            ShaderRecompiler::PatchOverUserData(request, *served, *entry->patches);
            if (fresh) {
                std::lock_guard lock(stageMemoMutex);
                const auto found = stageMemo.find(memoHash);
                if (found != stageMemo.end() && found->second.entry == entry && found->second.served == nullptr) found->second.served = served;
            }
            memoResult = served;
            if (!VerifyStageMemo()) {
                counters.hits.fetch_add(1, std::memory_order_relaxed);
                shaderMemory.Seed(entry->regions);
                if (wantRegions) {
                    stageCapture.regions.clear();
                    for (const auto& [address, words] : entry->regions) stageCapture.regions.push_back({address, std::as_bytes(std::span(words))});
                    stageCapture.memo = entry;
                }
                regionsDone();
                stageCapture.compiled = std::move(memoResult);
                phaseTiming.Phase(DrawRowRecompile);
                reportStageMemo(entries);
                return stageCapture.compiled;
            }
        }
        reportStageMemo(entries);
    }

    const auto capture = [&] {
        const SampledReadScope sampling(evidenceReads);
        return shaderMemory.Capture(request, handle.get());
    }();
    part(StageTimes::Capture);

    // This stage's own reads: the draw cache's and the memo's.
    auto recent = wantRegions || memoKey != nullptr ? shaderMemory.TakeRecentRegions() : std::vector<ShaderRecompiler::MemoryRegion>{};
    if (wantRegions) stageCapture.regions = recent;
    regionsDone();
    if (traceCapSync()) traceCapture("draw-capture", program.binary.codeAddress, submission.queue, memory, Graphics::Recorder::ThreadWaitedMs() - waitedBefore);
    if (profile) {
        ++captures;
        phaseTiming.Phase(DrawRowCapture);

        const auto waited = std::min(Graphics::Recorder::ThreadWaitedMs() - waitedBefore, phaseMs[DrawRowCapture]);
        phaseMs[DrawRowCapture] -= waited;
        phaseMs[DrawRowCaptureHookWaits] += waited;
    }
    if (dumpTarget != 0) {

        const auto slot0 = (static_cast<std::uint64_t>(readRegister(queue.context, 0x390)) << 40u) | (static_cast<std::uint64_t>(readRegister(queue.context, 0x318)) << 8u);
        if ((graphics.hasColorTarget && graphics.color.address == dumpTarget) || slot0 == dumpTarget) static_cast<void>(dumpRequest(program.binary.codeAddress, request));
    }
    if (dumpSlot1 != 0) {
        const auto value = [&](std::uint32_t offset) -> std::uint64_t { const auto it = queue.context.find(offset); return it == queue.context.end() ? 0u : it->second; };
        const auto slot1 = (value(0x391) << 40u) | (value(0x327) << 8u);
        if (slot1 == dumpSlot1) {
            static_cast<void>(dumpRequest(program.binary.codeAddress, request));
            if (std::FILE* file = std::fopen("draw_slot1.regs", "w")) {
                for (const auto& [offset, value] : queue.context) std::fprintf(file, "context %x %08x\n", offset, value);
                for (const auto& [offset, value] : queue.userConfig) std::fprintf(file, "uconfig %x %08x\n", offset, value);
                for (const auto& [offset, value] : queue.shader) std::fprintf(file, "shader %x %08x\n", offset, value);
                std::fclose(file);
            }
        }
    }

    static const bool reuseCapture = std::getenv("APS5_NO_CAPTURE_REUSE") == nullptr;
    phaseTiming.Phase(DrawRowCapture);

    if (subPhases) lap = std::chrono::steady_clock::now();
    bool memoHit = false;
    stageCapture.compiled = reuseCapture ? ShaderRecompiler::Recompile(request, *capture, &memoHit) : std::make_shared<const ShaderRecompiler::RecompileResult>(ShaderRecompiler::Recompile(request));
    if (subPhases) {
        part(memoHit ? StageTimes::RecompileHit : StageTimes::RecompileMiss);
        reportStageTimes();
    }
    phaseTiming.Phase(DrawRowRecompile);
    if (memoKey != nullptr && (memoStore || memoResult != nullptr)) insertStageMemo(memoHash, handle, pushOffset, memoWords, mappings, stageCapture.compiled, memoResult, request.context.userData, recent, program.binary.codeAddress);
    // Shared, not copied: the result is immutable (a copy cost a few dozen allocations per stage).
    return stageCapture.compiled;
}

void Driver::insertStageMemo(std::uint64_t hash, const std::shared_ptr<const ShaderRecompiler::SourceHandle>& handle, std::uint32_t pushOffset, const std::vector<std::uint32_t>& key, std::uint64_t mappings, const std::shared_ptr<const ShaderRecompiler::RecompileResult>& result, const std::shared_ptr<const ShaderRecompiler::RecompileResult>& memoResult, std::span<const std::uint32_t> userData, std::span<const ShaderRecompiler::MemoryRegion> recent, std::uint64_t codeAddress) {
    auto& counters = stageMemoCounters();
    if (memoResult != nullptr) {
        // APS5_VERIFY_STAGE_MEMO: the memo's result against the capture's.
        counters.verified.fetch_add(1, std::memory_order_relaxed);
        const auto difference = stageMemoDifference(*memoResult, *result);
        if (!difference.empty()) {
            counters.differed.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<int> reports{0};
            if (reports.fetch_add(1) < 32) std::fprintf(stderr, "[stage-memo] verify: program 0x%llx: %s\n", static_cast<unsigned long long>(codeAddress), difference.c_str());
        }
    }
    const auto patches = ShaderRecompiler::UserDataPatchesFor(*handle, *result);
    if (patches == nullptr) {
        counters.refusedPatches.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // Every copy the analysis names must hold in this result (an image the materializer cleared as
    // invalid, say, holds zeros where the user words are not): the key would let the user words move
    // where the result does not follow them.
    for (const auto& patch : *patches) {
        if (patch.userWord >= userData.size()) {
            counters.refusedMismatch.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const auto value = userData[patch.userWord];
        std::uint32_t held = 0;
        if (patch.binding == ShaderRecompiler::UserDataPatch::PushConstants) {
            if ((static_cast<std::size_t>(patch.word) + 1u) * sizeof(held) > result->pushConstants.size()) {
                counters.refusedMismatch.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            std::memcpy(&held, result->pushConstants.data() + static_cast<std::size_t>(patch.word) * sizeof(held), sizeof(held));
        } else {
            if (patch.binding >= result->bindings.size() || patch.word >= result->bindings[patch.binding].guestDescriptor.size()) {
                counters.refusedMismatch.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            held = result->bindings[patch.binding].guestDescriptor[patch.word];
        }
        if (held != value) {
            counters.refusedMismatch.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    auto entry = std::make_shared<StageMemoEntry>();
    entry->handle = handle;
    entry->pushOffset = pushOffset;
    entry->key = key;
    entry->result = result;
    entry->patches = patches;
    entry->mappings = mappings;
    entry->regions.reserve(recent.size());
    for (const auto& region : recent) {
        std::vector<std::uint32_t> words(region.bytes.size() / sizeof(std::uint32_t));
        std::memcpy(words.data(), region.bytes.data(), words.size() * sizeof(std::uint32_t));
        const auto begin = region.guestAddress;
        const auto end = begin + region.bytes.size();
        if (!entry->runs.empty() && begin >= entry->runs.back().first && begin <= entry->runs.back().second + 65536u) entry->runs.back().second = std::max(entry->runs.back().second, end);
        else entry->runs.emplace_back(begin, end);
        entry->regions.emplace_back(begin, std::move(words));
    }
    std::uint64_t generation = 0;
    for (const auto& [begin, end] : entry->runs) generation = std::max(generation, GuestMemory::CollectWrites(begin, static_cast<std::size_t>(end - begin)));
    if (!entry->runs.empty() && generation == 0) {
        counters.refusedUnwatched.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    entry->generation = generation;
    {
        // A store landing between the capture's read and the collect above is stamped no later than
        // the generation: the words must still be the guest's now.
        const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
        if (!recent.empty() && !captureStable(recent)) {
            counters.refusedUnstable.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    counters.inserts.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(stageMemoMutex);
    auto [slot, inserted] = stageMemo.try_emplace(hash);
    if (inserted) {
        stageMemoOrder.push_back(hash);
        slot->second.order = std::prev(stageMemoOrder.end());
    } else {
        stageMemoOrder.splice(stageMemoOrder.end(), stageMemoOrder, slot->second.order);
    }
    // The served result belongs to the entry replaced; the misses in a row go on counting.
    slot->second.entry = std::move(entry);
    slot->second.served.reset();
    while (stageMemo.size() > StageMemoCapacity() && !stageMemoOrder.empty()) {
        stageMemo.erase(stageMemoOrder.front());
        stageMemoOrder.pop_front();
        counters.evictions.fetch_add(1, std::memory_order_relaxed);
    }
}

void Driver::cacheDrawStages(bool useDrawEntries, bool drawHit, const Pm4::DrawParameters& drawParameters, const std::optional<Graphics::IndirectDrawPath>& indirectCpu, const std::vector<DrawProgram>& programs, const std::vector<StageCapture>& stageCaptures, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, const std::vector<std::vector<Graphics::DecodeRead>>& decodeReads, bool verifyHit, const std::vector<std::shared_ptr<DispatchVariant>>& matched, std::vector<std::shared_ptr<DispatchVariant>>& fresh, std::uint64_t drawKey, bool registerKey, const std::shared_ptr<const DrawDecode>& decode, DrawPhaseTiming& phaseTiming) {
    if (useDrawEntries && !drawHit && !(drawParameters.indirect && indirectCpu)) {
        phaseTiming.Phase(DrawRowVectors);
        std::uint64_t unstable = 0, mismatches = 0;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            const auto& stageCapture = stageCaptures[i];
            if (stageCapture.compiled == nullptr) continue;
            auto variant = std::make_shared<DispatchVariant>();
            variant->compiled = stageCapture.compiled;
            variant->shader = programs[i].snapshot;
            variant->forgetSerial = stageCapture.forgetSerial;
            variant->pushOffset = stageCapture.pushOffset;
            if (vertexInfos[i]) variant->vertexInfo = std::make_shared<const ShaderRecompiler::ShaderVertexStageInfo>(*vertexInfos[i]);

            std::vector<ShaderRecompiler::MemoryRegion> regions(stageCapture.regions.begin(), stageCapture.regions.end());
            for (const auto& read : decodeReads[i]) regions.push_back({read.address, read.Bytes()});
            std::stable_sort(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress < b.guestAddress; });
            for (const auto& region : regions) {
                variant->runs.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
                const auto count = region.bytes.size() / sizeof(std::uint32_t);
                const auto offset = variant->words.size();
                variant->words.resize(offset + count);
                std::memcpy(variant->words.data() + offset, region.bytes.data(), count * sizeof(std::uint32_t));
            }
            if (verifyHit && matched[i] != nullptr && (matched[i]->runs != variant->runs || matched[i]->words != variant->words)) {
                ++mismatches;
                static std::atomic<std::uint64_t> reports{0};
                if (reports.fetch_add(1) < 20) std::fprintf(stderr, "[draw-cache] verify: stage %zu (program 0x%llx) of a hit captured differently: %zu runs / %zu words matched, %zu / %zu fresh\n", i, static_cast<unsigned long long>(programs[i].binary.codeAddress), matched[i]->runs.size(), matched[i]->words.size(), variant->runs.size(), variant->words.size());
            }
            if (insertCompare()) {
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
                if (!captureStable(stageCapture.regions)) {
                    ++unstable;
                    continue;
                }
            }
            fresh[i] = std::move(variant);
        }
        insertDrawEntry(drawKey, fresh, registerKey ? decode : nullptr);
        if (unstable != 0 || mismatches != 0) {
            std::lock_guard cacheLock(drawCacheMutex);
            drawEntryCounters.unstable += unstable;
            drawEntryCounters.verifyMismatches += mismatches;
        }
        phaseTiming.Phase(DrawRowKeyLookupValidate);
    }
}

}
