#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawThread.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/PreparedDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <cstdlib>

namespace AgcDriver::DriverDetail {

namespace {

struct FrontDrawThreadTag;

// Waits until `ready()`: spins about 50 us first, as the other thread usually answers within a few,
// then sleeps on `word` with `sleeping` set, which has the other thread wake it after its next
// change of `word` (set before `word` is read, so one of the two always sees the other).
template<typename TReady>
void awaitWord(const std::atomic<std::uint32_t>& word, std::atomic<bool>& sleeping, TReady&& ready) {
    if (ready()) return;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t spin = 1;; ++spin) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
        if (ready()) return;
        if ((spin & 63u) == 0 && std::chrono::steady_clock::now() - start > std::chrono::microseconds(50)) break;
    }
    for (;;) {
        sleeping.store(true, std::memory_order_seq_cst);
        const auto seen = word.load(std::memory_order_seq_cst);
        if (ready()) break;
        WaitWhile(word, seen);
    }
    sleeping.store(false, std::memory_order_relaxed);
}

// Whether a draw's stages may write guest memory besides its render targets: a buffer element not
// proved read-only, a written or atomic storage image, GDS, or an address-based (BDA) binding. A
// worker waits for such a draw on its draw thread before it prepares the next one, whose captures
// and validation read what recording it registers (pending writes, write stamps, pending images).
bool writesGuestMemory(std::span<const Graphics::CompiledShader> stages) {
    using Role = ShaderRecompiler::DescriptorRole;
    const auto any = [](const ShaderRecompiler::ElementFlags& flags) {
        for (const bool flag : flags) {
            if (flag) return true;
        }
        return false;
    };
    for (const auto& stage : stages) {
        for (const auto& binding : stage.program->bindings) {
            switch (binding.role) {
                case Role::GuestBuffers:
                    if (binding.bufferWritten.size() < binding.count || any(binding.bufferWritten)) return true;
                    break;
                case Role::GuestImages:
                    if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage && (binding.imageWritten.size() < binding.count || any(binding.imageWritten) || any(binding.imageAtomic))) return true;
                    break;
                case Role::Gds:
                case Role::BdaPagetable:
                    return true;
                default:
                    break;
            }
        }
    }
    return false;
}

struct PreparedDrawTag;

// The thread's PreparedDraw for one draw, emptied when the draw returns: the compiled stages and
// variants in it must not outlive the draw. A draw entered again on the thread while one runs (a
// capture retry) gets a record of its own.
class PreparedDrawLease {
public:
    PreparedDrawLease() {
        auto& shared = ThreadScratch<PreparedDraw, PreparedDrawTag>();
        if (shared.busy) {
            owned = std::make_unique<PreparedDraw>();
            prepared = owned.get();
        } else {
            prepared = &shared;
        }
        prepared->busy = true;
    }

    ~PreparedDrawLease() {
        prepared->Clear();
        prepared->busy = false;
    }

    PreparedDrawLease(const PreparedDrawLease&) = delete;
    PreparedDrawLease& operator=(const PreparedDrawLease&) = delete;

    PreparedDraw& operator*() const { return *prepared; }

private:
    std::unique_ptr<PreparedDraw> owned;
    PreparedDraw* prepared = nullptr;
};

}

DrawThread::DrawThread(std::uint32_t queue) : queue(queue) {
    for (auto& slot : slots) slot = std::make_unique<PreparedDraw>();
}

DrawThread::~DrawThread() = default;

bool DrawThread::Enabled() {
    static const bool enabled = std::getenv("APS5_NO_DRAW_THREAD") == nullptr && std::getenv("APS5_LOCKED_DRAW_PREPARE") == nullptr;
    return enabled;
}

bool DrawThread::Push() {
    const auto count = pushed.load(std::memory_order_relaxed);
    if (recorded.load(std::memory_order_acquire) != count) Drain(Wait::Push);
    if (abandoned.load(std::memory_order_acquire)) return false;
    pushed.store(count + 1, std::memory_order_seq_cst);
    doorbell.fetch_add(1, std::memory_order_seq_cst);
    if (backSleeping.load(std::memory_order_seq_cst)) WakeAll(doorbell);
    ++pushes;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (profile) report();
    return true;
}

void DrawThread::Drain(Wait why) {
    const auto count = pushed.load(std::memory_order_relaxed);
    if (recorded.load(std::memory_order_acquire) == count) return;
    // The draw thread needs the mutex to record what it waits for.
    require(!GuestMemory::GpuMutex().HeldByThisThread(), "a queue worker waited for its draw thread holding the GPU mutex");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // A draw thread that left early records nothing more.
    awaitWord(recorded, frontSleeping, [&] { return recorded.load(std::memory_order_acquire) == count || abandoned.load(std::memory_order_acquire); });
    const auto index = static_cast<std::size_t>(why);
    ++waits[index];
    if (profile) waitedMs[index] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void DrawThread::AttachFront() {
    GuestMemory::SetGpuLockFront(CurrentThreadToken(), [](void* back) { return static_cast<DrawThread*>(back)->Busy(); }, [](void* back) { static_cast<DrawThread*>(back)->Drain(Wait::Lock); }, this);
    HostThreadSlot<DrawThread*, FrontDrawThreadTag>::Set(this);
}

void DrawThread::Stop() {
    stopping.store(true, std::memory_order_seq_cst);
    doorbell.fetch_add(1, std::memory_order_seq_cst);
    WakeAll(doorbell);
    if (thread.joinable()) thread.join();
    GuestMemory::SetGpuLockFront(nullptr, nullptr, nullptr, nullptr);
    HostThreadSlot<DrawThread*, FrontDrawThreadTag>::Set(nullptr);
}

bool DrawThread::AwaitRecord(std::uint32_t taken) {
    awaitWord(doorbell, backSleeping, [&] { return pushed.load(std::memory_order_acquire) != taken || stopping.load(std::memory_order_acquire); });
    return pushed.load(std::memory_order_acquire) != taken;
}

void DrawThread::Recorded(std::uint32_t count) {
    recorded.store(count, std::memory_order_seq_cst);
    if (frontSleeping.load(std::memory_order_seq_cst)) WakeAll(recorded);
}

void DrawThread::Abandon() {
    abandoned.store(true, std::memory_order_seq_cst);
    Recorded(pushed.load(std::memory_order_acquire));
}

void DrawThread::report() {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    const auto seconds = std::chrono::duration<double>(now - lastReport).count();
    lastReport = now;
    const auto at = [&](Wait why) { return static_cast<std::size_t>(why); };
    std::fprintf(stderr, "[draw-thread] queue 0x%x (%.0f s): %llu draws handed over; the worker waited for the draw thread before a hand-off %llu times (%.0f ms), at its GPU mutex %llu (%.0f ms), before a packet %llu (%.0f ms), after a memory-writing draw %llu (%.0f ms), at the end of a submission %llu (%.0f ms)\n", queue, seconds, static_cast<unsigned long long>(pushes), static_cast<unsigned long long>(waits[at(Wait::Push)]), waitedMs[at(Wait::Push)], static_cast<unsigned long long>(waits[at(Wait::Lock)]), waitedMs[at(Wait::Lock)], static_cast<unsigned long long>(waits[at(Wait::Packet)]), waitedMs[at(Wait::Packet)], static_cast<unsigned long long>(waits[at(Wait::Writer)]), waitedMs[at(Wait::Writer)], static_cast<unsigned long long>(waits[at(Wait::End)]), waitedMs[at(Wait::End)]);
    pushes = 0;
    waits.fill(0);
    waitedMs.fill(0);
}

DrawThread* FrontDrawThread() {
    return HostThreadSlot<DrawThread*, FrontDrawThreadTag>::Get();
}

DrawVerdict Driver::drawnVerdict(DrawPhaseTiming& phaseTiming, std::uint64_t captures) {
    phaseTiming.Phase(DrawRowVectors);
    if (!phaseTiming.profile) return DrawVerdict::Drawn;
    auto& pending = pendingDrawPhases();
    pending.phases = true;
    pending.captures = captures;
    pending.ms = phaseTiming.phaseMs;
    pending.tailAt = phaseTiming.phaseLap;
    return DrawVerdict::Drawn;
}

DrawVerdict Driver::draw(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::string& rejected) {
    PerformanceTimer timing("Driver.Draw");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr || std::getenv("APS5_PROFILE_DRAW_PHASES") != nullptr;
    std::array<double, DrawDriverPhaseCount> phaseMs{};
    std::uint64_t captures = 0;
    auto phaseLap = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    DrawPhaseTiming phaseTiming{profile, phaseMs, phaseLap};
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[DrawRowPrologue] = std::chrono::duration<double, std::milli>(phaseLap - packetStartedAt()).count();
    // Once taken, held until the draw returns: the record is emptied under it.
    std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::defer_lock);
    if (auto* back = FrontDrawThread(); back != nullptr && back->FrontReady()) {
        auto& slot = back->Slot();
        // Emptied here unless handed over: the draw thread empties what it records.
        struct Preparing {
            DrawThread& back;
            PreparedDraw& slot;
            bool handed = false;
            ~Preparing() {
                if (!handed) slot.Clear();
                back.preparing = false;
            }
        } preparing{*back, slot};
        back->preparing = true;
        if (const auto verdict = prepareDraw(slot, queue, packet, submission, rejected, gpuLock, timing, phaseTiming, captures)) return *verdict;
        // The worker's deferred labels come before this draw in the stream: recorded now, as the
        // serial draw's own lock records them (the draw thread's sees only its own, empty list).
        if (!deferredLabels().labels.empty()) {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
            std::lock_guard labelLock(GuestMemory::GpuMutex());
            recordLabelsForPacket(slot.device.get(), submission.queue);
        }
        const bool writes = writesGuestMemory(slot.stages);
        slot.collectEpoch = GuestMemory::ThreadCollectEpoch();
        preparing.handed = back->Push();
        if (!preparing.handed) recordPrepared(slot, gpuLock, timing, phaseTiming);
        else if (writes) back->Drain(DrawThread::Wait::Writer);
        return drawnVerdict(phaseTiming, captures);
    }
    const PreparedDrawLease prepared;
    if (const auto verdict = prepareDraw(*prepared, queue, packet, submission, rejected, gpuLock, timing, phaseTiming, captures)) return *verdict;
    recordPrepared(*prepared, gpuLock, timing, phaseTiming);
    return drawnVerdict(phaseTiming, captures);
}

std::optional<DrawVerdict> Driver::prepareDraw(PreparedDraw& prepared, QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::string& rejected, std::unique_lock<GuestMemory::GpuMutexType>& gpuLock, PerformanceTimer& timing, DrawPhaseTiming& phaseTiming, std::uint64_t& captures) {
    const bool profile = phaseTiming.profile;
    auto& phaseMs = phaseTiming.phaseMs;
    auto drawParameters = Pm4::ResolveDraw(packet, queue);
    bool traceIndirect = false;
    if (const auto verdict = precheckDraw(queue, submission, packet, drawParameters, rejected, traceIndirect)) return *verdict;
    phaseTiming.Phase(DrawRowPrecheck);
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;

    static const std::uint64_t dumpTarget = [] { const char* text = std::getenv("APS5_DUMP_DRAW_SHADERS"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();

    static const std::uint64_t dumpSlot1 = [] { const char* text = std::getenv("APS5_DUMP_DRAW_SLOT1"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();

    static const bool lockedPrepare = std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr;
    prepared.queue = submission.queue;
    auto& localDevice = prepared.device;
    if (lockedPrepare) {

        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        gpuLock.lock();
        timing.Mark("gpu_mutex_wait");
        phaseTiming.Phase(DrawRowLockWait);
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(DrawRowLabels);
    } else if ((localDevice = device.Load()) == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        std::lock_guard createLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    timing.Mark("device_setup");
    phaseTiming.Phase(DrawRowVectors);

    const bool useDrawEntries = drawEntries() && !ShaderRecompiler::DebugProbeActive() && dumpTarget == 0 && dumpSlot1 == 0 && drawCacheActive();
    static const bool traceRelocation = std::getenv("APS5_TRACE_DRAW_RELOC") != nullptr;
    static const bool probeStageMemo = std::getenv("APS5_TRACE_STAGE_MEMO") != nullptr;
    const bool wantRegions = useDrawEntries || traceRelocation || probeStageMemo;
    const bool registerKey = useDrawEntries && registerKeyEnabled();
    auto& drawKey = prepared.drawKey;
    std::shared_ptr<DrawEntry> entry;
    auto& decode = prepared.decode;
    if (registerKey) {
        const auto keyStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        drawKey = drawRegisterKey(queue, *submission.shaders, localDevice->Serial());
        std::lock_guard cacheLock(drawCacheMutex);
        ++drawEntryCounters.lookups;
        ++drawEntryCounters.registerKeyLookups;
        if (profile) drawEntryCounters.keyUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - keyStart).count();
        const auto found = drawCache.find(drawKey);
        if (found != drawCache.end()) {
            entry = found->second;
            decode = entry->decode;
        } else {
            ++drawEntryCounters.absent;
        }
        static const bool traceKey = std::getenv("APS5_TRACE_DRAWKEY") != nullptr;
        if (traceKey) traceDrawKey(queue, *submission.shaders, localDevice->Serial(), found != drawCache.end());
    }
    phaseTiming.Phase(DrawRowKeyLookupValidate);

    resolveDrawDecode(queue, submission, decode, registerKey, drawKey, profile);
    const auto& graphics = decode->state;
    const auto& pixel = decode->pixel;
    // The decode's programs, read in place. Only a mesh draw (its index buffer descriptor) and an
    // indirect draw (each record's patches) write user words; they work on a copy, made here
    // before anything refers into the programs.
    const bool writesPrograms = graphics.stages.mesh || drawParameters.indirect;
    std::vector<DrawProgram> writablePrograms;
    if (writesPrograms) writablePrograms = decode->programs;
    const std::vector<DrawProgram>& programs = writesPrograms ? writablePrograms : decode->programs;
    const auto setMeshIndexBuffer = [&](const Pm4::DrawParameters& parameters) {
        if (!graphics.stages.mesh) return;
        auto& words = writablePrograms.front().userData;
        require(programs.front().firstUserSgpr == 0 && words.size() >= ShaderRecompiler::MeshIndexBufferUserWord + 4, "mesh program lacks the hidden user words");
        const auto descriptor = Graphics::MeshIndexBufferDescriptor(parameters);
        std::copy(descriptor.begin(), descriptor.end(), words.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    };
    if (!drawParameters.indirect) setMeshIndexBuffer(drawParameters);
    else if (graphics.stages.mesh) setMeshIndexBuffer(Pm4::DrawParameters{drawParameters.indexAddress, std::max(drawParameters.indexCount, 1u), drawParameters.indexSize, 1, 0, drawParameters.indexed});
    const std::vector<Role>& roles = decode->roles;
    phaseTiming.Phase(DrawRowDecode);

    const auto locate = [&](std::uint32_t location) -> std::optional<std::pair<std::size_t, std::size_t>> {
        if (location == 0x280u) return std::nullopt;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (roles[i] == Role::Fragment || roles[i] == Role::GeometryBack || location < programs[i].userDataBase) continue;
            const auto word = location - programs[i].userDataBase + (8u - programs[i].firstUserSgpr);
            if (word < programs[i].userData.size()) return std::make_pair(i, static_cast<std::size_t>(word));
        }
        return std::nullopt;
    };
    if (drawParameters.indirect) {
        auto& indirect = *drawParameters.indirect;
        const auto sgprOf = [&](std::uint32_t location) -> std::int32_t {
            const auto word = locate(location);
            if (!word || word->first != 0) return -1;
            return static_cast<std::int32_t>(programs.front().firstUserSgpr + word->second);
        };
        indirect.baseVertexSgpr = sgprOf(indirect.baseVertexLocation);
        indirect.startInstanceSgpr = sgprOf(indirect.startInstanceLocation);
        indirect.drawIndexSgpr = sgprOf(indirect.drawIndexLocation);
    }
    // The regions the stages read: two per program here, then the captures' and the decode reads.
    // Reserved once: grown a push at a time, the list was reallocated several times per draw.
    auto& memory = prepared.memory;
    memory.reserve(2 * programs.size() + 32);
    auto& linked = prepared.linked;
    linked.reserve(programs.size());
    for (std::size_t i = 0; i < programs.size(); ++i) {
        const auto& program = programs[i];
        memory.insert(memory.end(), program.memory.begin(), program.memory.end());
        linked.push_back({roles[i], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
    }
    timing.Mark("prepare");
    phaseTiming.Phase(DrawRowProgramPrepare);

    auto& vertexInfos = prepared.vertexInfos;
    vertexInfos.resize(programs.size());
    auto& requests = prepared.requests;
    if (requests.size() < programs.size()) requests.resize(programs.size());
    auto& decodeReads = prepared.decodeReads;
    decodeReads.resize(programs.size());
    const auto decodeVertexInfo = [&](std::size_t i) {
        const auto& program = programs[i];
        if (program.binary.stage == Stage::Fragment || roles[i] == Role::GeometryBack) return;
        decodeReads[i].clear();
        vertexInfos[i] = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, &decodeReads[i]);
    };
    if (!registerKey) {
        for (std::size_t i = 0; i < programs.size(); ++i) decodeVertexInfo(i);
        phaseTiming.Phase(DrawRowDecode);
    }
    auto& shaderMemory = prepared.shaderMemory.emplace(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter());
    // The compiled stages of a miss (shared with the capture, never copied) and the rect-list's
    // generated stages; `stages` points into both.
    auto& results = prepared.results;
    auto& rectStages = prepared.rectStages;
    auto& stages = prepared.stages;
    results.reserve(programs.size());
    stages.reserve(programs.size() + (graphics.rectList ? 2u : 0u));
    std::uint32_t pushCursorBytes = 0;

    auto& programResults = prepared.programResults;
    programResults.assign(programs.size(), nullptr);

    auto& stageCaptures = prepared.stageCaptures;
    stageCaptures.resize(programs.size());
    auto& matched = prepared.matched;
    matched.resize(programs.size());
    auto& matchedRegions = prepared.matchedRegions;
    matchedRegions.resize(programs.size());

    auto& fresh = prepared.fresh;
    fresh.resize(programs.size());

    auto& recompiled = prepared.recompiled;
    recompiled.assign(programs.size(), false);
    bool drawHit = false;
    bool verifyHit = false;
    lookupDraw(submission, localDevice, graphics, pixel, programs, roles, vertexInfos, useDrawEntries, registerKey, profile, drawKey, entry, matched, matchedRegions, drawHit, verifyHit, phaseTiming, phaseMs);
    if (useDrawEntries) noteDrawCacheLookup(drawHit || verifyHit);

    if (registerKey) {

        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (matched[i] != nullptr && drawHit && !verifyDrawRecipe()) {
                if (matched[i]->vertexInfo != nullptr) vertexInfos[i] = *matched[i]->vertexInfo;
                continue;
            }
            decodeVertexInfo(i);
            if (matched[i] != nullptr && verifyDrawRecipe() && (vertexInfos[i].has_value() != (matched[i]->vertexInfo != nullptr) || (vertexInfos[i] && !sameVertexInfo(*vertexInfos[i], *matched[i]->vertexInfo)))) {
                static std::atomic<std::uint64_t> reports{0};
                if (reports.fetch_add(1) < 20) std::fprintf(stderr, "[draw-cache] verify: stage %zu (program 0x%llx) of a hit has a vertex stage info unlike its variant's\n", i, static_cast<unsigned long long>(programs[i].binary.codeAddress));
                std::lock_guard cacheLock(drawCacheMutex);
                ++drawEntryCounters.verifyDecodeMismatches;
            }
        }
        phaseTiming.Phase(DrawRowDecode);
    }

    static const bool indxOffsetSkipFold = std::getenv("APS5_INDX_OFFSET_SKIP_FOLD") != nullptr;
    static const bool indexedOffsetFold = std::getenv("APS5_NO_INDEXED_OFFSET_FOLD") == nullptr;
    const auto fold = [&](const ShaderRecompiler::RecompileResult& main, Pm4::DrawParameters& parameters) {
        if (parameters.indexed && !indexedOffsetFold) return;
        if (main.vertexOffsetSgpr >= 0 && (parameters.firstVertex == 0 || !indxOffsetSkipFold)) {
            const auto offset = drawUserWord(programs.front(), main.vertexOffsetSgpr);
            require(offset <= std::numeric_limits<std::uint32_t>::max() - parameters.firstVertex, "draw vertex offset overflow");
            parameters.firstVertex += offset;
        }
        if (main.instanceOffsetSgpr >= 0) parameters.firstInstance = drawUserWord(programs.front(), main.instanceOffsetSgpr);
    };

    std::optional<Graphics::IndirectDrawPath> indirectCpu;
    auto& pushOffsets = prepared.pushOffsets;
    pushOffsets.assign(programs.size(), 0);
    auto& resultIndex = prepared.resultIndex;
    resultIndex.assign(programs.size(), 0);
    for (std::size_t i = 0; i < programs.size(); ++i) {
        if (roles[i] == Role::GeometryBack) continue;
        const auto& program = programs[i];
        pushOffsets[i] = pushCursorBytes;
        if (drawHit) {

            programResults[i] = matched[i]->compiled.get();
            memory.insert(memory.end(), matchedRegions[i].begin(), matchedRegions[i].end());
        } else {
            resultIndex[i] = results.size();
            results.push_back(compileDrawStage(i, pushCursorBytes, queue, submission, programs, graphics, pixel, vertexInfos, requests, memory, linked, drawParameters, localDevice, shaderMemory, stageCaptures, recompiled, drawHit, wantRegions, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected));
            if (!rejected.empty()) return DrawVerdict::Rejected;
            programResults[i] = results.back().get();
        }
        const auto& result = *programResults[i];
        if (i == 0 && drawParameters.indirect) {
            indirectCpu = classifyIndirectDraw(result, graphics, programs.front(), localDevice, drawParameters, traceIndirect);
        } else if (i == 0) {
            fold(result, drawParameters);
        }
        require(result.pushConstants.size() <= Graphics::PipelinePushConstantBytes - pushCursorBytes, "stage push constants exceed the pipeline push constant block");
        stages.push_back({program.binary.stage, &result, result.pushConstants.empty() ? 0u : pushCursorBytes});
        pushCursorBytes += static_cast<std::uint32_t>(result.pushConstants.size());
    }

    if (traceRelocation && !drawHit) traceDrawRelocation(structuralDrawKey(queue, *submission.shaders, localDevice->Serial()), programs, stageCaptures);
    if (probeStageMemo && !drawHit) traceStageMemo(structuralDrawKey(queue, *submission.shaders, localDevice->Serial()), programs, stageCaptures, programResults, vertexInfos);
    cacheDrawStages(useDrawEntries, drawHit, drawParameters, indirectCpu, programs, stageCaptures, vertexInfos, decodeReads, verifyHit, matched, fresh, drawKey, registerKey, decode, phaseTiming);
    timing.Mark("shader_compile_and_link");

    for (const auto& reads : decodeReads) {
        for (const auto& read : reads) memory.push_back({read.address, read.Bytes()});
    }

    if (recordQueuedLabelsAfterCapture(submission.queue, memory)) return draw(queue, packet, submission, rejected);

    const auto buildRectList = [&] {
        phaseTiming.Phase(DrawRowVectors);
        require(programs.size() == 2 && programResults[0] != nullptr && programResults[1] != nullptr, "rect-list requires vertex and fragment programs");
        auto rectangle = ShaderRecompiler::BuildRectListShaders(*programResults[0], *programResults[1], localDevice->Target());
        rectStages[0] = std::move(rectangle.control);
        rectStages[1] = std::move(rectangle.evaluation);
        if (prepared.rectList) {
            phaseTiming.Phase(DrawRowRectList);
            return;
        }
        require(stages.size() == 2, "rect-list requires vertex and fragment programs");
        stages.insert(stages.begin() + 1, {{Stage::TessellationControl, &rectStages[0], 0}, {Stage::TessellationEvaluation, &rectStages[1], 0}});
        prepared.rectList = true;
        phaseTiming.Phase(DrawRowRectList);
    };
    if (graphics.rectList) buildRectList();
    auto& snapshots = prepared.snapshots;
    const auto snapshot = [&] {
        snapshots.clear();
        snapshots.reserve(memory.size());
        for (const auto& region : memory) snapshots.push_back({region.guestAddress, region.bytes});
    };
    snapshot();
    timing.Mark("post_compile_prepare");
    if (drawParameters.indirect && indirectCpu) {

        const auto indirect = *drawParameters.indirect;
        if (drawHit) {

            // The variants' results, shared: a record's patch replaces its program's entry below.
            for (std::size_t i = 0; i < programs.size(); ++i) {
                if (programResults[i] == nullptr) continue;
                resultIndex[i] = results.size();
                results.push_back(matched[i]->compiled);
            }
        }
        recordQueuedLabelsBeforeRead(submission.queue);
        const auto readStart = std::chrono::steady_clock::now();
        const auto count = std::min(indirect.countIndirect ? Pm4::ReadDrawCount(indirect) : indirect.count, indirect.count);
        std::vector<Pm4::DrawArguments> records;
        for (std::uint32_t record = 0; record < count; ++record) records.push_back(Pm4::ReadDrawArguments(indirect, record));
        Graphics::CountIndirectDraw(*indirectCpu, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        const auto baseVertexWord = locate(indirect.baseVertexLocation);
        const auto startInstanceWord = locate(indirect.startInstanceLocation);
        const auto drawIndexWord = indirect.drawIndexEnabled ? locate(indirect.drawIndexLocation) : std::nullopt;
        for (std::uint32_t record = 0; record < records.size(); ++record) {
            const auto& arguments = records[record];
            if (traceIndirect) std::fprintf(stderr, "[draw]   record %u: count %u instances %u first %u vertexOffset %u startInstance %u\n", record, arguments.count, arguments.instances, arguments.firstVertexOrIndex, arguments.vertexOffset, arguments.firstInstance);
            if (arguments.count == 0 || arguments.instances == 0) continue;
            std::set<std::size_t> patched;
            const auto patch = [&](const std::optional<std::pair<std::size_t, std::size_t>>& word, std::uint32_t value) {
                if (!word) return;
                writablePrograms[word->first].userData[word->second] = value;
                patched.insert(word->first);
            };
            patch(baseVertexWord, indirect.recordBytes == 20 ? arguments.vertexOffset : arguments.firstVertexOrIndex);
            patch(startInstanceWord, arguments.firstInstance);
            patch(drawIndexWord, record);
            Pm4::DrawParameters direct{0, arguments.count, 0, arguments.instances, drawParameters.flags, drawParameters.indexed, 0, 0};
            if (drawParameters.indexed) {

                if (arguments.firstVertexOrIndex >= drawParameters.indexCount) continue;
                direct.indexAddress = drawParameters.indexAddress + static_cast<std::uint64_t>(arguments.firstVertexOrIndex) * drawParameters.indexSize;
                direct.indexCount = std::min(arguments.count, drawParameters.indexCount - arguments.firstVertexOrIndex);
                direct.indexSize = drawParameters.indexSize;
            } else {
                direct.firstVertex = indirect.indxOffset;
            }
            if (graphics.stages.mesh) {
                setMeshIndexBuffer(direct);
                patched.insert(0);
            }
            for (const auto programIndex : patched) {
                auto& result = results[resultIndex[programIndex]];
                const auto* previous = result.get();
                const auto pushBytes = result->pushConstants.size();
                decodeVertexInfo(programIndex);
                result = compileDrawStage(programIndex, pushOffsets[programIndex], queue, submission, programs, graphics, pixel, vertexInfos, requests, memory, linked, drawParameters, localDevice, shaderMemory, stageCaptures, recompiled, drawHit, wantRegions, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected);
                if (!rejected.empty()) return DrawVerdict::Rejected;
                require(result->pushConstants.size() == pushBytes, "patched program changed its push constant layout");
                // The stages name the program's result by address: repoint them at the new one.
                for (auto& stage : stages) {
                    if (stage.program == previous) stage.program = result.get();
                }
                programResults[programIndex] = result.get();
            }
            fold(*programResults[0], direct);
            if (graphics.rectList && patched.contains(0)) buildRectList();
            snapshot();
            if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
                rejected = std::move(*known);
                return DrawVerdict::Rejected;
            }
            // Drawn before the next record's patches: they change the stages in place.
            prepared.parameters = direct;
            recordPrepared(prepared, gpuLock, timing, phaseTiming);
        }
        return drawnVerdict(phaseTiming, captures);
    }

    auto& recipeStages = prepared.recipeStages;
    if (registerKey && !drawParameters.indirect && Graphics::DrawRecipes()) {
        recipeStages.reserve(programs.size());
        for (std::size_t i = 0; i < programs.size(); ++i) recipeStages.push_back(drawHit ? matched[i] : fresh[i]);
        if (std::all_of(recipeStages.begin(), recipeStages.end(), [](const std::shared_ptr<DispatchVariant>& variant) { return variant == nullptr; })) recipeStages.clear();
    }
    auto& recipe = prepared.recipe;
    if (drawHit && !recipeStages.empty()) {
        recipe = findDrawRecipe(drawKey, recipeStages);
        if (recipe == nullptr) VulkanDevice::NoteDrawRecipeMiss(VulkanDevice::DrawRecipePrecheck::NoRecipe);
    }
    if (recipe == nullptr) {
        if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
            rejected = std::move(*known);
            return DrawVerdict::Rejected;
        }
    }
    prepared.parameters = drawParameters;
    return std::nullopt;
}

void Driver::recordPrepared(PreparedDraw& prepared, std::unique_lock<GuestMemory::GpuMutexType>& gpuLock, PerformanceTimer& timing, DrawPhaseTiming& phaseTiming) {
    if (!gpuLock.owns_lock()) {
        phaseTiming.Phase(DrawRowVectors);
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        gpuLock.lock();
        timing.Mark("gpu_mutex_wait");
        phaseTiming.Phase(DrawRowLockWait);

        if (auto current = device.Load(); current != nullptr && current != prepared.device) {
            static std::atomic<std::uint64_t> replaced{0};
            std::fprintf(stderr, "[draw] device replaced during unlocked preparation (%llu)\n", static_cast<unsigned long long>(++replaced));
            prepared.device = std::move(current);
        }

        recordLabelsForPacket(prepared.device.get(), prepared.queue);
        phaseTiming.Phase(DrawRowLabels);
    }
    noteDrawWriters(prepared.stages, prepared.queue);
    phaseTiming.Phase(DrawRowVectors);
    const auto& graphics = prepared.decode->state;
    if (prepared.recipe != nullptr) {
        if (prepared.device->DrawFromRecipe(graphics, prepared.parameters, prepared.stages, prepared.snapshots, prepared.recipe) == RecipeOutcome::Recorded) {
            phaseTiming.Phase(DrawRowGraphics);
            timing.Mark("draw_and_resource_release");
            return;
        }

        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
    }
    std::shared_ptr<const DrawRecipe> built;
    prepared.device->Draw(graphics, prepared.parameters, prepared.stages, prepared.snapshots, prepared.recipeStages.empty() ? nullptr : &built);
    phaseTiming.Phase(DrawRowGraphics);
    if (built != nullptr) attachDrawRecipe(prepared.drawKey, prepared.recipeStages, std::move(built));
    timing.Mark("draw_and_resource_release");
}

void Driver::runDrawThread(DrawThread& back) noexcept {
    onWorkerThread() = true;
    {
        char role[32];
        std::snprintf(role, sizeof role, "draw thread 0x%x", back.queue);
        PinWorkerThread(role);
    }
    GuestMemory::TagGpuLockThread(back.queue);
    if (back.collectMemo != nullptr) GuestMemory::AdoptCollectMemo(back.collectMemo);
    std::array<double, DrawDriverPhaseCount> phaseMs{};
    std::chrono::steady_clock::time_point phaseLap{};
    DrawPhaseTiming phaseTiming{false, phaseMs, phaseLap};
    try {
        for (std::uint32_t taken = 0; back.AwaitRecord(taken); ++taken) {
            auto& prepared = back.Record(taken);
            GuestMemory::AdoptCollectEpoch(prepared.collectEpoch);
            PerformanceTimer timing("Driver.Draw");
            std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::defer_lock);
            try {
                recordPrepared(prepared, gpuLock, timing, phaseTiming);
            } catch (const std::exception& error) {
                // The worker's packet has moved on: reported without its register dump.
                CaptureTrace::Log("draw-error queue=%x reason=%.256s (draw thread)", back.queue, error.what());
                reportSkip("draw", std::string(error.what()) + " [draw thread]");
            }
            // Emptied under the mutex, as draw() empties its record: the last reference to a device
            // ends its recorder, which needs it.
            if (!gpuLock.owns_lock()) gpuLock.lock();
            prepared.Clear();
            gpuLock.unlock();
            back.Recorded(taken + 1);
        }
    } catch (...) {
        back.Abandon();
    }
}

}
