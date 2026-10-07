#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestArena.hpp"
#ifdef _WIN32
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

namespace {

constexpr std::uint64_t UnitBytes = 65536;
constexpr std::size_t ReasonCount = static_cast<std::size_t>(PublishReason::Count);
constexpr const char* ReasonNames[ReasonCount] = {"hook", "region", "indirect", "copy-src", "copy-dst", "fill", "fill-clear", "label", "keys", "scanout", "validate", "upload", "tail", "evict", "retire", "teardown"};

std::uint64_t BudgetBytes() {
    static const std::uint64_t budget = [] {
        const char* text = std::getenv("APS5_UNIT_SHADOW_MIB");
        return (text != nullptr ? std::strtoull(text, nullptr, 10) : 1024ull) << 20u;
    }();
    return budget;
}

std::uint32_t SlabUnits() {
    static const std::uint32_t units = [] {
        const char* text = std::getenv("APS5_UNIT_SHADOW_SLAB_MIB");
        const auto mib = text != nullptr ? std::strtoull(text, nullptr, 10) : 8ull;
        return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(mib, 1, 1024) * 16);
    }();
    return units;
}

// APS5_TRACE_SHADOW=<n>: the first n publishes, the first 64 seeds, every evict and retire.
std::atomic<std::int64_t> tracePublishesLeft{[] {
    const char* text = std::getenv("APS5_TRACE_SHADOW");
    return text != nullptr ? static_cast<std::int64_t>(std::strtoll(text, nullptr, 10)) : std::int64_t{0};
}()};
std::atomic<std::int64_t> traceSeedsLeft{64};

bool TraceEnabled() {
    static const bool enabled = std::getenv("APS5_TRACE_SHADOW") != nullptr;
    return enabled;
}

struct Statistics {
    std::atomic<std::uint64_t> retiled{0}, retiledBytes{0}, refused{0}, refusedBytes{0}, seeds{0}, seedBytes{0};
    std::atomic<std::uint64_t> detiledShadow{0}, detiledShadowBytes{0}, detiledImport{0}, detiledImportBytes{0}, tailPublishes{0};
    std::array<std::atomic<std::uint64_t>, ReasonCount> published{}, publishedBytes{};
    std::atomic<std::uint64_t> staleDroppedCpu{0}, staleDroppedDriver{0}, skippedInCompletion{0}, evictions{0}, lostOnRetire{0}, droppedOnRetire{0}, slabLost{0}, verified{0}, mismatches{0};
    // Buffer shadows: made (and their bytes), refused, uses bound, seeds recorded, writes marked.
    std::atomic<std::uint64_t> buffersMade{0}, buffersMadeBytes{0}, buffersRefused{0}, bufferUses{0}, bufferSeeds{0}, bufferSeedBytes{0}, bufferWrites{0};
};

Statistics& Stats() {
    static Statistics statistics;
    return statistics;
}

struct UnitShadow {
    Context context;
    // The import's range and buffer (taken at creation: a publish under the import table's mutex
    // must not look them up), and the unit base (the import's base rounded down to a unit).
    std::uint64_t importBase = 0;
    std::uint64_t importBytes = 0;
    VkBuffer importBuffer = VK_NULL_HANDLE;
    std::uint64_t base = 0;
    // Per unit: 0 = never shadowed, else the tracker generation the slab bytes are current at;
    // whether a slab -> import copy was recorded at that generation.
    std::vector<std::uint64_t> generation;
    std::vector<std::uint8_t> published;
    // Slab k covers units [k * SlabUnits(), (k + 1) * SlabUnits()), made on demand; a failed
    // allocation is not retried before the noted present.
    std::vector<std::shared_ptr<ShadowSlab>> slabs;
    std::vector<std::uint64_t> slabFailedUntil;
    // Buffer shadows (BufferShadowFor): slabs spanning a bound range's units whole, disjoint, a few
    // per import. A unit inside one lives there, whatever fixed slab lies over it.
    std::vector<std::shared_ptr<ShadowSlab>> bufferSlabs;
    std::uint32_t liveUnits = 0;

    const std::shared_ptr<ShadowSlab>* BufferSlabOf(std::uint64_t unit) const {
        for (const auto& slab : bufferSlabs) {
            if (unit >= slab->firstUnit && unit - slab->firstUnit < slab->units) return &slab;
        }
        return nullptr;
    }
    // The slab a unit's shadowed bytes live in: its buffer slab, else the fixed slab over it.
    std::shared_ptr<ShadowSlab> SlabOf(std::uint64_t unit) const {
        if (const auto* slab = BufferSlabOf(unit)) return *slab;
        return slabs[static_cast<std::size_t>(unit / SlabUnits())];
    }
    std::uint64_t Units() const { return generation.size(); }
    std::uint64_t UnitBegin(std::uint64_t unit) const { return base + unit * UnitBytes; }
    std::uint64_t UnitOf(std::uint64_t address) const { return (address - base) / UnitBytes; }
    std::uint64_t ImportEnd() const { return importBase + importBytes; }
    // The unit's bytes inside the import.
    std::pair<std::uint64_t, std::uint64_t> UnitRange(std::uint64_t unit) const { return {std::max(UnitBegin(unit), importBase), std::min(UnitBegin(unit + 1), ImportEnd())}; }
    bool Live(std::uint64_t unit) const { return generation[unit] != 0 && published[unit] == 0; }
    void Recount() {
        std::uint32_t live = 0;
        for (std::uint64_t unit = 0; unit < Units(); ++unit) live += Live(unit) ? 1u : 0u;
        liveUnits = live;
    }
};

struct Shadows {
    // A cache line of its own: every draw locks it, and counters other threads write sat beside it.
    alignas(64) AgcDriver::Mutex mutex;
    std::map<std::uint64_t, std::shared_ptr<UnitShadow>> byBase;
    std::uint64_t liveBytes = 0;
    std::uint64_t peakBytes = 0;
    std::uint32_t liveSlabs = 0;
    // Counted for the [storage] segment: shadows retired and the slab bytes they returned.
    std::uint64_t retired = 0;
};

Shadows& Registry() {
    static Shadows shadows;
    return shadows;
}

VkDeviceSize SlabBytes(const ShadowSlab& slab) {
    return static_cast<VkDeviceSize>(slab.units) * UnitBytes;
}

// A shadow leaves the registry: its slabs (fixed and buffer) no longer count against the budget.
// They live on wherever a batch or a build kept them.
void forgetSlabs(Shadows& registry, const UnitShadow& shadow) {
    for (const auto* list : {&shadow.slabs, &shadow.bufferSlabs}) {
        for (const auto& slab : *list) {
            if (slab == nullptr) continue;
            registry.liveBytes -= SlabBytes(*slab);
            --registry.liveSlabs;
        }
    }
}

// The shadows overlapping [address, end): imports never overlap, so at most a few.
template<typename TVisit>
bool anyOverlapping(const Shadows& registry, std::uint64_t address, std::uint64_t end, TVisit&& visit) {
    if (end <= address || registry.byBase.empty()) return false;
    auto it = registry.byBase.upper_bound(address);
    if (it != registry.byBase.begin()) --it;
    for (; it != registry.byBase.end() && it->first < end; ++it) {
        const auto& shadow = it->second;
        if (address < shadow->ImportEnd() && shadow->importBase < end && visit(shadow)) return true;
    }
    return false;
}

std::vector<std::shared_ptr<UnitShadow>> overlapping(const Shadows& registry, std::uint64_t address, std::uint64_t end) {
    std::vector<std::shared_ptr<UnitShadow>> result;
    anyOverlapping(registry, address, end, [&](const std::shared_ptr<UnitShadow>& shadow) {
        result.push_back(shadow);
        return false;
    });
    return result;
}

std::shared_ptr<UnitShadow> find(const Shadows& registry, const HostImport& import) {
    const auto found = registry.byBase.find(import.base);
    if (found == registry.byBase.end()) return nullptr;
    const auto& shadow = found->second;
    return shadow->importBuffer == import.buffer && shadow->importBytes == import.bytes ? shadow : nullptr;
}

std::shared_ptr<UnitShadow> findOrCreate(Shadows& registry, const Context& context, const HostImport& import) {
    if (auto shadow = find(registry, import); shadow != nullptr && shadow->context.device == context.device) return shadow;
    // A stale entry (an import replaced without a retire, a replaced device): dropped; its slabs
    // live on wherever a batch kept them.
    if (const auto found = registry.byBase.find(import.base); found != registry.byBase.end()) {
        forgetSlabs(registry, *found->second);
        registry.byBase.erase(found);
    }
    auto shadow = std::make_shared<UnitShadow>();
    shadow->context = context;
    shadow->importBase = import.base;
    shadow->importBytes = import.bytes;
    shadow->importBuffer = import.buffer;
    shadow->base = import.base & ~(UnitBytes - 1);
    const auto units = (import.base + import.bytes - shadow->base + UnitBytes - 1) / UnitBytes;
    shadow->generation.assign(static_cast<std::size_t>(units), 0);
    shadow->published.assign(static_cast<std::size_t>(units), 0);
    const auto slabs = (units + SlabUnits() - 1) / SlabUnits();
    shadow->slabs.assign(static_cast<std::size_t>(slabs), nullptr);
    shadow->slabFailedUntil.assign(static_cast<std::size_t>(slabs), 0);
    registry.byBase.emplace(import.base, shadow);
    return shadow;
}

// One tracker query over units [first, last]: `changed[k]` (k = unit - first) whether a block
// stamp exceeds the unit's generation (a generation of 0 reads as changed), `cpu[k]` whether a
// collect made it (a CPU store).
void changedUnits(const UnitShadow& shadow, std::uint64_t first, std::uint64_t last, std::vector<std::uint8_t>& changed, std::vector<std::uint8_t>& cpu) {
    const auto count = static_cast<std::size_t>(last - first + 1);
    changed.assign(count, 1);
    cpu.assign(count, 1);
    const auto begin = std::max(shadow.UnitBegin(first), shadow.importBase);
    const auto end = std::min(shadow.UnitBegin(last + 1), shadow.ImportEnd());
    if (end <= begin) return;
    GuestMemory::ChangedBlocks(begin, static_cast<std::size_t>(end - begin), std::span(shadow.generation).subspan(static_cast<std::size_t>(first), count), changed, cpu);
}

std::shared_ptr<ShadowSlab> makeSlab(const Context& context, std::uint64_t firstUnit, std::uint32_t units) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = static_cast<VkDeviceSize>(units) * UnitBytes;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (context.Function<PFN_vkCreateBuffer>("vkCreateBuffer")(context.device, &info, nullptr, &buffer) != VK_SUCCESS) return nullptr;
    VkMemoryRequirements requirements{};
    context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    try {
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory) != VK_SUCCESS) memory = VK_NULL_HANDLE;
        if (memory != VK_NULL_HANDLE && context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(context.device, buffer, memory, 0) != VK_SUCCESS) {
            context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    } catch (const std::exception&) {
        memory = VK_NULL_HANDLE;
    }
    if (memory == VK_NULL_HANDLE) {
        context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
        return nullptr;
    }
    return std::make_shared<ShadowSlab>(context, buffer, memory, firstUnit, units);
}

struct SlabCopies {
    std::shared_ptr<ShadowSlab> slab;
    std::vector<VkBufferCopy> regions;
};

// Records the copies as one command group into the active recorder's open batch (a waited batch
// without one). `ranges` are the guest ranges copied, for the note and the hazard tracker.
void recordCopies(const UnitShadow& shadow, const std::vector<SlabCopies>& copies, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges, std::uint64_t bytes) {
    const auto& context = shadow.context;
    using CommandClass = Recorder::CommandClass;
    auto* recorder = Recorder::Active();
    std::unique_ptr<CommandBatch> batch;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkAccessFlags covered = 0;
    auto timing = Recorder::NoTiming;
    if (recorder != nullptr) {
        commands = recorder->Commands(&covered);
        timing = recorder->BeginGpuTiming(CommandClass::ShadowPublish);
        if (Recorder::BarrierValidate()) recorder->NoteAccess(CommandClass::ShadowPublish, Recorder::Access{{}, ranges, {}, VK_PIPELINE_STAGE_TRANSFER_BIT});
    } else {
        batch = std::make_unique<CommandBatch>(context);
        commands = batch->Handle();
    }
    // The retile that filled the slab (its trailing barrier covers transfer reads and writes) and
    // every earlier import writer precede the copies; the CopyBuffer merge rule.
    constexpr VkAccessFlags transferAccess = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    if (recorder != nullptr && (covered & transferAccess) == transferAccess && Recorder::MergeBarriers()) {
        Recorder::CountMerged(CommandClass::ShadowPublish);
    } else {
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, transferAccess);
        Recorder::CountBarriers(CommandClass::ShadowPublish);
    }
    const auto copyBuffer = context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer");
    for (const auto& entry : copies) copyBuffer(commands, entry.slab->buffer, shadow.importBuffer, static_cast<std::uint32_t>(entry.regions.size()), entry.regions.data());
    // The published bytes are visible to everything recorded after and to the host, as after a fill.
    constexpr VkAccessFlags publishedAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, publishedAccess);
    Recorder::CountBarriers(CommandClass::ShadowPublish);
    if (batch) {
        batch->SubmitAndWait();
        return;
    }
    recorder->EndGpuTiming(timing, bytes);
    recorder->MarkCovered(publishedAccess);
    for (const auto& entry : copies) recorder->Keep(entry.slab);
    // CPU readers reach the import's bytes only after the hook's SyncThrough waits by this note.
    recorder->NotePendingWrites(ranges);
}

// APS5_SHADOW_VERIFY=1: the published bytes read back from the slabs equal the guest bytes.
void verifyPublished(const UnitShadow& shadow, const std::vector<SlabCopies>& copies) {
    const auto& context = shadow.context;
    if (auto* recorder = Recorder::Active()) {
        Recorder::CountSync(4);
        recorder->Sync();
    }
    for (const auto& entry : copies) {
        for (const auto& region : entry.regions) {
            Buffer host(context, static_cast<std::size_t>(region.size), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            CommandBatch batch(context);
            CopyBuffer(context, batch.Handle(), entry.slab->buffer, region.srcOffset, host.Handle(), 0, region.size);
            RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            batch.SubmitAndWait();
            host.Invalidate();
            const auto address = shadow.importBase + region.dstOffset;
            Stats().verified.fetch_add(1, std::memory_order_relaxed);
            if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), static_cast<std::size_t>(region.size)) || std::memcmp(reinterpret_cast<const void*>(address), host.Bytes().data(), static_cast<std::size_t>(region.size)) != 0) {
                Stats().mismatches.fetch_add(1, std::memory_order_relaxed);
                static std::atomic<int> reports{0};
                if (reports.fetch_add(1) < 8) std::fprintf(stderr, "[shadow] verify: published bytes differ from the slab at 0x%llx+0x%llx\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(region.size));
            }
        }
    }
}

// The publish of one shadow: `selected[k]` (k = unit - first) names the units the scope chose.
std::size_t publishUnits(const std::shared_ptr<UnitShadow>& shadow, std::uint64_t first, std::uint64_t last, const std::vector<std::uint8_t>& selected, PublishReason reason) {
    auto& registry = Registry();
    std::vector<SlabCopies> copies;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::uint64_t bytes = 0;
    std::size_t units = 0;
    {
        // The stamps judged below see every CPU store into the live selected units up to now: a
        // store no collect walked yet would read as fresh and the copy would land over it. Every
        // caller holds the GpuMutex, so the live set is stable across the walk.
        std::uint64_t walkFirst = last + 1, walkLast = first;
        {
            std::lock_guard lock(registry.mutex);
            for (auto unit = first; unit <= last; ++unit) {
                if (!selected[static_cast<std::size_t>(unit - first)] || !shadow->Live(unit)) continue;
                walkFirst = std::min(walkFirst, unit);
                walkLast = std::max(walkLast, unit);
            }
        }
        if (walkFirst > walkLast) return 0;
        const auto walkBegin = std::max(shadow->UnitBegin(walkFirst), shadow->importBase);
        const auto walkEnd = std::min(shadow->UnitBegin(walkLast + 1), shadow->ImportEnd());
        if (walkEnd > walkBegin) GuestMemory::CollectWritesUncached(walkBegin, static_cast<std::size_t>(walkEnd - walkBegin));
    }
    {
        std::lock_guard lock(registry.mutex);
        std::vector<std::uint8_t> changed, cpu;
        changedUnits(*shadow, first, last, changed, cpu);
        for (auto unit = first; unit <= last; ++unit) {
            const auto k = static_cast<std::size_t>(unit - first);
            if (!selected[k] || !shadow->Live(unit)) continue;
            if (changed[k] == GuestMemory::BlockWritten) {
                // Newer bytes reached the import since the retile: the slab's are dead.
                shadow->generation[unit] = 0;
                (cpu[k] != 0 ? Stats().staleDroppedCpu : Stats().staleDroppedDriver).fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            const auto slab = shadow->SlabOf(unit);
            if (slab == nullptr) {
                shadow->generation[unit] = 0;
                continue;
            }
            const auto [begin, end] = shadow->UnitRange(unit);
            if (end <= begin) continue;
            const VkBufferCopy region{(unit - slab->firstUnit) * UnitBytes + (begin - shadow->UnitBegin(unit)), begin - shadow->importBase, end - begin};
            if (!copies.empty() && copies.back().slab == slab && copies.back().regions.back().dstOffset + copies.back().regions.back().size == region.dstOffset) {
                copies.back().regions.back().size += region.size;
                ranges.back().second = end;
            } else {
                if (copies.empty() || copies.back().slab != slab) copies.push_back({slab, {}});
                copies.back().regions.push_back(region);
                ranges.emplace_back(begin, end);
            }
            shadow->published[unit] = 1;
            slab->lastUse = Recorder::Presents();
            bytes += end - begin;
            ++units;
        }
        shadow->Recount();
    }
    if (units == 0) return 0;
    Stats().published[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    Stats().publishedBytes[static_cast<std::size_t>(reason)].fetch_add(bytes, std::memory_order_relaxed);
    if (tracePublishesLeft.load(std::memory_order_relaxed) > 0 && tracePublishesLeft.fetch_sub(1, std::memory_order_relaxed) > 0) {
        const auto packet = GuestMemory::CurrentPacket();
        std::fprintf(stderr, "[shadow] publish %s: 0x%llx+0x%llx, %zu units, %.2f MiB, %zu slabs (packet 0x%x queue 0x%x)\n", ReasonNames[static_cast<std::size_t>(reason)], static_cast<unsigned long long>(ranges.front().first), static_cast<unsigned long long>(ranges.back().second - ranges.front().first), units, bytes / 1048576.0, copies.size(), packet.opcode, packet.queue);
    }
    recordCopies(*shadow, copies, ranges, bytes);
    if (ShadowVerify()) verifyPublished(*shadow, copies);
    return units;
}

std::size_t publishRange(const std::shared_ptr<UnitShadow>& shadow, std::uint64_t address, std::uint64_t end, PublishScope scope, PublishReason reason) {
    const auto begin = std::max(address, shadow->importBase);
    const auto stop = std::min(end, shadow->ImportEnd());
    if (stop <= begin || shadow->liveUnits == 0) return 0;
    const auto first = shadow->UnitOf(begin);
    const auto last = shadow->UnitOf(stop - 1);
    std::vector<std::uint8_t> selected(static_cast<std::size_t>(last - first + 1), 0);
    bool any = false;
    for (auto unit = first; unit <= last; ++unit) {
        const auto [unitBegin, unitEnd] = shadow->UnitRange(unit);
        const bool whole = address <= unitBegin && end >= unitEnd;
        if (scope == PublishScope::PartialUnits && whole) continue;
        selected[static_cast<std::size_t>(unit - first)] = 1;
        any = true;
    }
    return any ? publishUnits(shadow, first, last, selected, reason) : 0;
}

// Frees the least recently used unpinned slab of any shadow after publishing its fresh units; false
// when nothing can be freed (a pinned slab is a destination of the write-back in progress: its
// copies are recorded into it after this decision).
bool evictOne(Shadows& registry) {
    std::shared_ptr<UnitShadow> victim;
    std::shared_ptr<ShadowSlab> victimSlab;
    {
        std::lock_guard lock(registry.mutex);
        std::uint64_t oldest = ~0ull;
        const auto consider = [&](const std::shared_ptr<UnitShadow>& shadow, const std::shared_ptr<ShadowSlab>& slab) {
            if (slab == nullptr || slab->pins.load(std::memory_order_relaxed) != 0 || slab->lastUse >= oldest) return;
            oldest = slab->lastUse;
            victim = shadow;
            victimSlab = slab;
        };
        for (const auto& [base, shadow] : registry.byBase) {
            for (const auto& slab : shadow->slabs) consider(shadow, slab);
            for (const auto& slab : shadow->bufferSlabs) consider(shadow, slab);
        }
    }
    if (victim == nullptr) return false;
    const auto firstUnit = victimSlab->firstUnit;
    const auto lastUnit = std::min<std::uint64_t>(firstUnit + victimSlab->units, victim->Units()) - 1;
    const auto published = publishRange(victim, victim->UnitBegin(firstUnit), victim->UnitBegin(lastUnit + 1), PublishScope::Whole, PublishReason::Evict);
    std::lock_guard lock(registry.mutex);
    // Gone meanwhile (a retire, or a buffer slab replaced): nothing left to free.
    const auto buffer = std::find(victim->bufferSlabs.begin(), victim->bufferSlabs.end(), victimSlab);
    const bool isBuffer = buffer != victim->bufferSlabs.end();
    if (!isBuffer && victim->slabs[static_cast<std::size_t>(firstUnit / SlabUnits())] != victimSlab) return true;
    if (TraceEnabled()) std::fprintf(stderr, "[shadow] evict %s slab at unit %llu of import 0x%llx (%.1f MiB, %zu units published)\n", isBuffer ? "buffer" : "fixed", static_cast<unsigned long long>(firstUnit), static_cast<unsigned long long>(victim->importBase), SlabBytes(*victimSlab) / 1048576.0, published);
    // Its units read from the import from now on (the publish put their bytes there). Units of a
    // fixed slab that a buffer slab holds are the buffer slab's.
    for (auto unit = firstUnit; unit <= lastUnit; ++unit) {
        if (!isBuffer && victim->BufferSlabOf(unit) != nullptr) continue;
        victim->generation[unit] = 0;
        victim->published[unit] = 0;
    }
    victim->Recount();
    registry.liveBytes -= SlabBytes(*victimSlab);
    --registry.liveSlabs;
    if (isBuffer) victim->bufferSlabs.erase(buffer);
    else victim->slabs[static_cast<std::size_t>(firstUnit / SlabUnits())].reset();
    Stats().evictions.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// The buffer slab over units [first, last] of `shadow`, made under the budget. Fresh results of
// fixed slabs over those units reach the import first: the buffer slab takes the units over
// unseeded. Null when the budget or the device refuses it, or another buffer slab holds a unit.
std::shared_ptr<ShadowSlab> makeBufferSlab(Shadows& registry, const Context& context, const std::shared_ptr<UnitShadow>& shadow, std::uint64_t first, std::uint64_t last) {
    const auto units = static_cast<std::uint32_t>(last - first + 1);
    const auto needed = static_cast<std::uint64_t>(units) * UnitBytes;
    if (needed > BudgetBytes()) return nullptr;
    for (;;) {
        {
            std::lock_guard lock(registry.mutex);
            if (registry.liveBytes + needed <= BudgetBytes()) break;
        }
        if (!evictOne(registry)) return nullptr;
    }
    publishRange(shadow, shadow->UnitBegin(first), shadow->UnitBegin(last + 1), PublishScope::Whole, PublishReason::Upload);
    auto slab = makeSlab(context, first, units);
    if (slab == nullptr) return nullptr;
    std::lock_guard lock(registry.mutex);
    for (const auto& other : shadow->bufferSlabs) {
        if (other->firstUnit <= last && first < other->firstUnit + other->units) return nullptr;
    }
    for (auto unit = first; unit <= last; ++unit) {
        shadow->generation[unit] = 0;
        shadow->published[unit] = 0;
    }
    shadow->Recount();
    const auto position = std::find_if(shadow->bufferSlabs.begin(), shadow->bufferSlabs.end(), [&](const std::shared_ptr<ShadowSlab>& other) { return other->firstUnit > first; });
    shadow->bufferSlabs.insert(position, slab);
    registry.liveBytes += needed;
    registry.peakBytes = std::max(registry.peakBytes, registry.liveBytes);
    ++registry.liveSlabs;
    Stats().buffersMade.fetch_add(1, std::memory_order_relaxed);
    Stats().buffersMadeBytes.fetch_add(needed, std::memory_order_relaxed);
    if (TraceEnabled()) std::fprintf(stderr, "[shadow] buffer slab 0x%llx+0x%llx of import 0x%llx (%u units)\n", static_cast<unsigned long long>(shadow->UnitBegin(first)), static_cast<unsigned long long>(needed), static_cast<unsigned long long>(shadow->importBase), units);
    return slab;
}

// Records the seed copies (import -> buffer slab) into the recorder's open batch, after every
// earlier writer of the import and every earlier user of the slab, before the work that binds it.
// `ranges` are the guest ranges copied.
void recordSeeds(const UnitShadow& shadow, Recorder& recorder, const std::shared_ptr<ShadowSlab>& slab, const std::vector<VkBufferCopy>& seeds, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges, std::uint64_t bytes) {
    const auto& context = shadow.context;
    using CommandClass = Recorder::CommandClass;
    const auto commands = recorder.Commands();
    const auto timing = recorder.BeginGpuTiming(CommandClass::StagingIn);
    if (Recorder::BarrierValidate()) recorder.NoteAccess(CommandClass::StagingIn, Recorder::Access{ranges, {}, {}, VK_PIPELINE_STAGE_TRANSFER_BIT});
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    Recorder::CountBarriers(CommandClass::StagingIn);
    context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, shadow.importBuffer, slab->buffer, static_cast<std::uint32_t>(seeds.size()), seeds.data());
    // The seeded bytes are visible to the shaders bound to the slab, which may also store into it.
    constexpr VkAccessFlags seededAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, seededAccess | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    Recorder::CountBarriers(CommandClass::StagingIn);
    recorder.MarkCovered(seededAccess);
    recorder.EndGpuTiming(timing, bytes);
    // A CPU store into the import must not land before the copy read it.
    recorder.NotePendingReads(ranges, Recorder::ReadKind::GpuCopy);
    recorder.Keep(slab);
    Stats().bufferSeeds.fetch_add(1, std::memory_order_relaxed);
    Stats().bufferSeedBytes.fetch_add(bytes, std::memory_order_relaxed);
    if (traceSeedsLeft.load(std::memory_order_relaxed) > 0 && traceSeedsLeft.fetch_sub(1, std::memory_order_relaxed) > 0) std::fprintf(stderr, "[shadow] buffer seed 0x%llx..0x%llx from the import (%.2f MiB in %zu copies)\n", static_cast<unsigned long long>(ranges.front().first), static_cast<unsigned long long>(ranges.back().second), bytes / 1048576.0, seeds.size());
}

}

bool UnitShadowEnabled() {
    static const bool enabled = std::getenv("APS5_NO_UNIT_SHADOW") == nullptr && GuestMemory::WriteWatched();
    return enabled;
}

bool ShadowVerify() {
    static const bool verify = std::getenv("APS5_SHADOW_VERIFY") != nullptr;
    return verify;
}

void NoteShadowSeed(std::uint64_t begin, std::uint64_t end) {
    Stats().seeds.fetch_add(1, std::memory_order_relaxed);
    Stats().seedBytes.fetch_add(end - begin, std::memory_order_relaxed);
    if (traceSeedsLeft.load(std::memory_order_relaxed) > 0 && traceSeedsLeft.fetch_sub(1, std::memory_order_relaxed) > 0) std::fprintf(stderr, "[shadow] seed 0x%llx+0x%llx from the import\n", static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end - begin));
}

PublishReason PublishReasonFor(const char* flushReason) {
    const auto is = [&](const char* name) { return flushReason != nullptr && std::strcmp(flushReason, name) == 0; };
    if (is("memory access")) return GuestMemory::CurrentReadSite() == GuestMemory::ReadSite::Scanout ? PublishReason::Scanout : PublishReason::Hook;
    if (is("imported buffer region") || is("copied buffer region") || is("sampled texture")) return PublishReason::Region;
    if (is("indirect dispatch arguments") || is("indirect draw arguments")) return PublishReason::Indirect;
    if (is("buffer copy source")) return PublishReason::CopySource;
    if (is("buffer copy destination")) return PublishReason::CopyDestination;
    if (is("buffer fill")) return PublishReason::Fill;
    if (is("packet store")) return PublishReason::Label;
    if (is("buffer copy count") || is("dispatch-cache entry")) return PublishReason::Validate;
    if (is("storage image creation") || is("storage refresh")) return PublishReason::Upload;
    return PublishReason::Hook;
}

ShadowSlab::ShadowSlab(const Context& context, VkBuffer buffer, VkDeviceMemory memory, std::uint64_t firstUnit, std::uint32_t units) : context(context), buffer(buffer), memory(memory), firstUnit(firstUnit), units(units) {}

ShadowSlab::~ShadowSlab() {
    if (buffer != VK_NULL_HANDLE) context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
    if (memory != VK_NULL_HANDLE) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
}

ShadowSlabPin::ShadowSlabPin(std::shared_ptr<ShadowSlab> slab) : slab(std::move(slab)) {
    this->slab->pins.fetch_add(1, std::memory_order_relaxed);
}

ShadowSlabPin::~ShadowSlabPin() {
    slab->pins.fetch_sub(1, std::memory_order_relaxed);
}

std::size_t PublishShadow(std::uint64_t address, std::size_t bytes, PublishScope scope, PublishReason reason) {
    if (!UnitShadowEnabled() || scope == PublishScope::None || bytes == 0) return 0;
    if (Recorder::InCompletion() && GuestMemory::CurrentReadSite() == GuestMemory::ReadSite::Store) {
        // The store's own stamp makes the unit stale; a publish recorded now would land over the
        // store when the batch runs (the completion is not waited for).
        Stats().skippedInCompletion.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const auto end = bytes > std::numeric_limits<std::uint64_t>::max() - address ? std::numeric_limits<std::uint64_t>::max() : address + bytes;
    std::vector<std::shared_ptr<UnitShadow>> shadows;
    {
        auto& registry = Registry();
        std::lock_guard lock(registry.mutex);
        shadows = overlapping(registry, address, end);
    }
    std::size_t units = 0;
    for (const auto& shadow : shadows) units += publishRange(shadow, address, end, scope, reason);
    return units;
}

bool AnyShadowedOverlaps(std::uint64_t address, std::size_t bytes) {
    if (!UnitShadowEnabled() || bytes == 0) return false;
    const auto end = bytes > std::numeric_limits<std::uint64_t>::max() - address ? std::numeric_limits<std::uint64_t>::max() : address + bytes;
    auto& registry = Registry();
    std::lock_guard lock(registry.mutex);
    return anyOverlapping(registry, address, end, [&](const std::shared_ptr<UnitShadow>& shadow) {
        if (shadow->liveUnits == 0) return false;
        const auto begin = std::max(address, shadow->importBase);
        const auto stop = std::min(end, shadow->ImportEnd());
        for (auto unit = shadow->UnitOf(begin); unit <= shadow->UnitOf(stop - 1); ++unit) {
            if (shadow->Live(unit)) return true;
        }
        return false;
    });
}

bool AnyShadowedOverlaps(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    if (!UnitShadowEnabled()) return false;
    for (const auto& [begin, end] : ranges) {
        if (end > begin && AnyShadowedOverlaps(begin, static_cast<std::size_t>(end - begin))) return true;
    }
    return false;
}

std::uint64_t SlabBoundary(const HostImport& import, std::uint64_t address) {
    const auto base = import.base & ~(UnitBytes - 1);
    const auto unit = (address - base) / UnitBytes;
    return base + (unit / SlabUnits() + 1) * SlabUnits() * UnitBytes;
}

VkDeviceSize SlabOffset(const HostImport& import, const ShadowSlab& slab, std::uint64_t address) {
    const auto base = import.base & ~(UnitBytes - 1);
    return (address - base) - slab.firstUnit * UnitBytes;
}

std::optional<ShadowDestination> ShadowDestinationFor(const Context& context, const HostImport& import, std::uint64_t begin, std::uint64_t end) {
    if (!UnitShadowEnabled() || end <= begin || begin < import.base || end > import.base + import.bytes) return std::nullopt;
    if (!GuestMemory::Watched(begin, static_cast<std::size_t>(end - begin))) return std::nullopt;
    auto& registry = Registry();
    std::shared_ptr<UnitShadow> shadow;
    std::size_t slabIndex = 0;
    std::uint64_t first = 0, last = 0;
    bool missing = false;
    {
        std::lock_guard lock(registry.mutex);
        shadow = findOrCreate(registry, context, import);
        first = shadow->UnitOf(begin);
        last = shadow->UnitOf(end - 1);
        slabIndex = static_cast<std::size_t>(first / SlabUnits());
        if (last / SlabUnits() != slabIndex) return std::nullopt;
        // A buffer shadow's units take no retile: the write-back stores to the import, whose stamp
        // makes those units stale until the shadow's next use seeds them.
        for (auto unit = first; unit <= last; ++unit) {
            if (shadow->BufferSlabOf(unit) == nullptr) continue;
            Stats().refused.fetch_add(1, std::memory_order_relaxed);
            Stats().refusedBytes.fetch_add(end - begin, std::memory_order_relaxed);
            return std::nullopt;
        }
        missing = shadow->slabs[slabIndex] == nullptr;
        if (missing && shadow->slabFailedUntil[slabIndex] > Recorder::Presents()) return std::nullopt;
    }
    if (missing) {
        const auto firstUnit = static_cast<std::uint64_t>(slabIndex) * SlabUnits();
        const auto units = static_cast<std::uint32_t>(std::min<std::uint64_t>(SlabUnits(), shadow->Units() - firstUnit));
        const auto needed = static_cast<std::uint64_t>(units) * UnitBytes;
        for (;;) {
            {
                std::lock_guard lock(registry.mutex);
                if (registry.liveBytes + needed <= BudgetBytes()) break;
            }
            if (!evictOne(registry)) {
                Stats().refused.fetch_add(1, std::memory_order_relaxed);
                Stats().refusedBytes.fetch_add(end - begin, std::memory_order_relaxed);
                return std::nullopt;
            }
        }
        auto slab = makeSlab(context, firstUnit, units);
        std::lock_guard lock(registry.mutex);
        if (slab == nullptr) {
            shadow->slabFailedUntil[slabIndex] = Recorder::Presents() + 1;
            Stats().refused.fetch_add(1, std::memory_order_relaxed);
            Stats().refusedBytes.fetch_add(end - begin, std::memory_order_relaxed);
            return std::nullopt;
        }
        if (shadow->slabs[slabIndex] == nullptr) {
            shadow->slabs[slabIndex] = slab;
            registry.liveBytes += SlabBytes(*slab);
            registry.peakBytes = std::max(registry.peakBytes, registry.liveBytes);
            ++registry.liveSlabs;
        }
    }
    // A unit the copy covers partly keeps its other bytes from the slab when fresh, else from a
    // seed of the import: the stamps must see the CPU's stores into it first (the write-back's own
    // walk covers the stored span only, not the unit's remainder outside the surface).
    for (auto unit = first; unit <= last; unit = last) {
        const auto [unitBegin, unitEnd] = shadow->UnitRange(unit);
        if (!(begin <= unitBegin && end >= unitEnd) && unitEnd > unitBegin) GuestMemory::CollectWritesUncached(unitBegin, static_cast<std::size_t>(unitEnd - unitBegin));
        if (unit == last) break;
    }
    std::lock_guard lock(registry.mutex);
    const auto& slab = shadow->slabs[slabIndex];
    if (slab == nullptr) return std::nullopt;
    slab->lastUse = Recorder::Presents();
    ShadowDestination destination{slab->buffer, SlabOffset(import, *slab, begin), slab, {}, std::make_shared<ShadowSlabPin>(slab)};
    std::vector<std::uint8_t> changed, cpu;
    bool queried = false;
    for (auto unit = first; unit <= last; ++unit) {
        const auto [unitBegin, unitEnd] = shadow->UnitRange(unit);
        if (begin <= unitBegin && end >= unitEnd) continue;
        if (!queried) {
            changedUnits(*shadow, first, last, changed, cpu);
            queried = true;
        }
        const auto k = static_cast<std::size_t>(unit - first);
        if (shadow->generation[unit] != 0 && changed[k] == 0) continue;
        destination.seedUnits.emplace_back(unitBegin, unitEnd);
    }
    return destination;
}

std::vector<ShadowRun> ShadowSources(const Context& context, const HostImport& import, std::uint64_t surfaceBase, std::span<const std::pair<std::uint64_t, std::uint64_t>> runs, std::span<const std::pair<std::uint64_t, std::uint64_t>> tailBlocks, bool countReads) {
    std::vector<ShadowRun> result;
    if (runs.empty()) return result;
    const auto importPiece = [&](std::uint64_t begin, std::uint64_t end) {
        result.push_back({begin, end, import.buffer, surfaceBase + begin - import.base, false, nullptr});
        if (countReads) Stats().detiledImportBytes.fetch_add(end - begin, std::memory_order_relaxed);
    };
    std::shared_ptr<UnitShadow> shadow;
    if (UnitShadowEnabled()) {
        auto& registry = Registry();
        std::lock_guard lock(registry.mutex);
        shadow = find(registry, import);
        if (shadow != nullptr && (shadow->context.device != context.device || (shadow->liveUnits == 0 && std::none_of(shadow->generation.begin(), shadow->generation.end(), [](std::uint64_t generation) { return generation != 0; })))) shadow = nullptr;
    }
    if (shadow == nullptr) {
        for (const auto& [begin, end] : runs) importPiece(begin, end);
        if (countReads) Stats().detiledImport.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    const auto first = shadow->UnitOf(surfaceBase + runs.front().first);
    const auto last = shadow->UnitOf(surfaceBase + runs.back().second - 1);
    std::vector<std::uint8_t> fresh(static_cast<std::size_t>(last - first + 1), 0);
    // The slab of each fresh unit (fixed or buffer), taken under the lock.
    std::vector<std::shared_ptr<ShadowSlab>> unitSlabs(fresh.size());
    {
        auto& registry = Registry();
        std::lock_guard lock(registry.mutex);
        std::vector<std::uint8_t> changed, cpu;
        changedUnits(*shadow, first, last, changed, cpu);
        for (auto unit = first; unit <= last; ++unit) {
            const auto k = static_cast<std::size_t>(unit - first);
            auto slab = shadow->SlabOf(unit);
            fresh[k] = shadow->generation[unit] != 0 && changed[k] == 0 && slab != nullptr ? 1 : 0;
            if (fresh[k] != 0) unitSlabs[k] = std::move(slab);
        }
    }
    // A tail block is moved whole by one window: its units take one source.
    for (const auto& [tailBegin, tailEnd] : tailBlocks) {
        if (tailEnd <= tailBegin) continue;
        const auto tailFirst = shadow->UnitOf(surfaceBase + tailBegin);
        const auto tailLast = shadow->UnitOf(surfaceBase + tailEnd - 1);
        if (tailFirst < first || tailLast > last) continue;
        bool anyFresh = false, anyImport = false;
        for (auto unit = tailFirst; unit <= tailLast; ++unit) (fresh[static_cast<std::size_t>(unit - first)] != 0 ? anyFresh : anyImport) = true;
        if (!anyFresh || !anyImport) continue;
        PublishShadow(surfaceBase + tailBegin, static_cast<std::size_t>(tailEnd - tailBegin), PublishScope::Whole, PublishReason::TailMip);
        Stats().tailPublishes.fetch_add(1, std::memory_order_relaxed);
        for (auto unit = tailFirst; unit <= tailLast; ++unit) fresh[static_cast<std::size_t>(unit - first)] = 0;
    }
    std::uint64_t shadowBytes = 0, importBytes = 0;
    const auto present = Recorder::Presents();
    for (const auto& [runBegin, runEnd] : runs) {
        if (runEnd <= runBegin) continue;
        const auto guestBegin = surfaceBase + runBegin;
        const auto guestEnd = surfaceBase + runEnd;
        for (auto unit = shadow->UnitOf(guestBegin); unit <= shadow->UnitOf(guestEnd - 1); ++unit) {
            const auto pieceBegin = std::max(guestBegin, shadow->UnitBegin(unit)) - surfaceBase;
            const auto pieceEnd = std::min(guestEnd, shadow->UnitBegin(unit + 1)) - surfaceBase;
            if (pieceEnd <= pieceBegin) continue;
            const bool isFresh = fresh[static_cast<std::size_t>(unit - first)] != 0;
            const auto& slab = isFresh ? unitSlabs[static_cast<std::size_t>(unit - first)] : nullptr;
            if (!result.empty() && result.back().end == pieceBegin && result.back().shadow == isFresh && result.back().slab == slab) {
                result.back().end = pieceEnd;
            } else if (isFresh) {
                slab->lastUse = present;
                result.push_back({pieceBegin, pieceEnd, slab->buffer, SlabOffset(import, *slab, surfaceBase + pieceBegin), true, slab});
            } else {
                result.push_back({pieceBegin, pieceEnd, import.buffer, surfaceBase + pieceBegin - import.base, false, nullptr});
            }
            (isFresh ? shadowBytes : importBytes) += pieceEnd - pieceBegin;
        }
    }
    if (!countReads) shadowBytes = importBytes = 0;
    if (shadowBytes != 0) {
        Stats().detiledShadow.fetch_add(1, std::memory_order_relaxed);
        Stats().detiledShadowBytes.fetch_add(shadowBytes, std::memory_order_relaxed);
    }
    if (importBytes != 0) {
        Stats().detiledImport.fetch_add(1, std::memory_order_relaxed);
        Stats().detiledImportBytes.fetch_add(importBytes, std::memory_order_relaxed);
    }
    if (ShadowVerify()) {
        std::vector<std::uint8_t> changed, cpu;
        std::lock_guard lock(Registry().mutex);
        changedUnits(*shadow, first, last, changed, cpu);
        for (const auto& run : result) {
            if (!run.shadow) continue;
            for (auto unit = shadow->UnitOf(surfaceBase + run.begin); unit <= shadow->UnitOf(surfaceBase + run.end - 1); ++unit) {
                Stats().verified.fetch_add(1, std::memory_order_relaxed);
                if (changed[static_cast<std::size_t>(unit - first)] != 0) Stats().mismatches.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    return result;
}

void MarkShadowed(const HostImport& import, std::span<const ShadowedRange> ranges, std::uint64_t generation) {
    if (!UnitShadowEnabled() || ranges.empty() || generation == 0) return;
    auto& registry = Registry();
    std::uint64_t bytes = 0;
    std::uint64_t lost = 0;
    {
        std::lock_guard lock(registry.mutex);
        const auto shadow = find(registry, import);
        if (shadow == nullptr) return;
        const auto present = Recorder::Presents();
        for (const auto& [begin, end, slab] : ranges) {
            if (end <= begin) continue;
            bytes += end - begin;
            for (auto unit = shadow->UnitOf(begin); unit <= shadow->UnitOf(end - 1); ++unit) {
                const auto current = shadow->SlabOf(unit);
                if (current == nullptr || current != slab) {
                    // The copies went into a slab the shadow no longer holds: the import keeps
                    // whatever it had, as after a refused piece.
                    ++lost;
                    continue;
                }
                shadow->generation[unit] = generation;
                shadow->published[unit] = 0;
                current->lastUse = present;
            }
        }
        shadow->Recount();
    }
    if (lost != 0) {
        Stats().slabLost.fetch_add(lost, std::memory_order_relaxed);
        static std::atomic<int> reports{0};
        if (reports.fetch_add(1) < 4) std::fprintf(stderr, "[shadow] %llu retiled units of import 0x%llx lost their slab before they were marked\n", static_cast<unsigned long long>(lost), static_cast<unsigned long long>(import.base));
    }
    Stats().retiled.fetch_add(1, std::memory_order_relaxed);
    Stats().retiledBytes.fetch_add(bytes, std::memory_order_relaxed);
    // The resource cache's pending memos (fastRevalidate) must see the change of the surface's
    // source, as they see a write-back.
    StorageTexture::BumpPendingSerial();
}

void RetireShadow(const Context& context, const HostImport& import, const std::function<bool(std::uint64_t, std::uint64_t)>& registered) {
    if (!UnitShadowEnabled()) return;
    auto& registry = Registry();
    std::shared_ptr<UnitShadow> shadow;
    std::vector<std::uint8_t> selected;
    bool any = false;
    std::uint64_t dropped = 0;
    {
        std::lock_guard lock(registry.mutex);
        shadow = find(registry, import);
        if (shadow == nullptr) return;
        // The memory the title took back holds its bytes now: only a unit still inside a readable
        // registered range receives its results.
        selected.assign(static_cast<std::size_t>(shadow->Units()), 0);
        for (std::uint64_t unit = 0; unit < shadow->Units(); ++unit) {
            if (!shadow->Live(unit)) continue;
            const auto [begin, end] = shadow->UnitRange(unit);
            if (end > begin && registered(begin, end)) {
                selected[static_cast<std::size_t>(unit)] = 1;
                any = true;
            } else {
                shadow->generation[unit] = 0;
                ++dropped;
            }
        }
        shadow->Recount();
    }
    if (dropped != 0) Stats().droppedOnRetire.fetch_add(dropped, std::memory_order_relaxed);
    std::size_t published = 0;
    if (any) {
        if (GuestMemory::GpuMutex().HeldByThisThread() && shadow->context.device == context.device) {
            published = publishUnits(shadow, 0, shadow->Units() - 1, selected, PublishReason::Retire);
        } else {
            Stats().lostOnRetire.fetch_add(shadow->liveUnits, std::memory_order_relaxed);
            static std::atomic<int> reports{0};
            if (reports.fetch_add(1) < 4) std::fprintf(stderr, "[shadow] import 0x%llx+0x%llx retired outside the device lock: %u shadowed units lost\n", static_cast<unsigned long long>(import.base), static_cast<unsigned long long>(import.bytes), shadow->liveUnits);
        }
    }
    std::lock_guard lock(registry.mutex);
    const auto found = registry.byBase.find(import.base);
    if (found == registry.byBase.end() || found->second != shadow) return;
    forgetSlabs(registry, *shadow);
    ++registry.retired;
    if (TraceEnabled()) std::fprintf(stderr, "[shadow] retire import 0x%llx+0x%llx: %zu units published, %llu dropped (memory no longer registered)\n", static_cast<unsigned long long>(import.base), static_cast<unsigned long long>(import.bytes), published, static_cast<unsigned long long>(dropped));
    registry.byBase.erase(found);
}

void PublishAllShadows(const Context& context, PublishReason reason) {
    if (!UnitShadowEnabled()) return;
    std::vector<std::shared_ptr<UnitShadow>> shadows;
    {
        auto& registry = Registry();
        std::lock_guard lock(registry.mutex);
        for (const auto& [base, shadow] : registry.byBase) {
            if (shadow->context.device == context.device) shadows.push_back(shadow);
        }
    }
    for (const auto& shadow : shadows) publishRange(shadow, shadow->importBase, shadow->ImportEnd(), PublishScope::Whole, reason);
}

void DestroyShadows(VkDevice device) {
    auto& registry = Registry();
    std::lock_guard lock(registry.mutex);
    for (auto it = registry.byBase.begin(); it != registry.byBase.end();) {
        if (it->second->context.device != device) {
            ++it;
            continue;
        }
        forgetSlabs(registry, *it->second);
        it = registry.byBase.erase(it);
    }
}

bool BufferShadowEnabled() {
    static const bool enabled = std::getenv("APS5_BUFFER_SHADOW") != nullptr && UnitShadowEnabled();
    return enabled;
}

std::uint64_t BufferShadowMinBytes() {
    static const std::uint64_t bytes = [] {
        const char* text = std::getenv("APS5_BUFFER_SHADOW_MIN_MIB");
        return (text != nullptr ? std::strtoull(text, nullptr, 10) : 8ull) << 20u;
    }();
    return bytes;
}

std::optional<BufferShadowBinding> BufferShadowFor(const Context& context, const HostImport& import, std::uint64_t begin, std::uint64_t end, bool make) {
    auto* recorder = Recorder::Active();
    if (!BufferShadowEnabled() || recorder == nullptr || end <= begin || begin < import.base || end > import.base + import.bytes) return std::nullopt;
    const auto bytes = static_cast<std::size_t>(end - begin);
    if (!GuestMemory::Watched(begin, bytes)) return std::nullopt;
    auto& registry = Registry();
    std::shared_ptr<UnitShadow> shadow;
    std::shared_ptr<ShadowSlab> slab;
    std::uint64_t first = 0, last = 0;
    {
        std::lock_guard lock(registry.mutex);
        shadow = make ? findOrCreate(registry, context, import) : find(registry, import);
        if (shadow == nullptr || shadow->context.device != context.device) return std::nullopt;
        first = shadow->UnitOf(begin);
        last = shadow->UnitOf(end - 1);
        for (const auto& candidate : shadow->bufferSlabs) {
            if (candidate->firstUnit <= first && last - candidate->firstUnit < candidate->units) {
                slab = candidate;
                break;
            }
            // A buffer slab over part of the range cannot serve it.
            if (candidate->firstUnit <= last && first < candidate->firstUnit + candidate->units) return std::nullopt;
        }
        if (slab == nullptr && !make) return std::nullopt;
    }
    if (slab == nullptr) {
        slab = makeBufferSlab(registry, context, shadow, first, last);
        if (slab == nullptr) {
            Stats().buffersRefused.fetch_add(1, std::memory_order_relaxed);
            return std::nullopt;
        }
    }
    // Pinned before anything below can evict (a seed's publish of another shadow could).
    auto pin = make ? std::make_shared<ShadowSlabPin>(slab) : nullptr;
    // Queued stores over the range land in the import first (in queue order they precede this use),
    // and the stamps then see every CPU store since the units' generations.
    recorder->FlushKeyStoresOverlapping(begin, bytes);
    recorder->FlushStoresOverlapping(begin, bytes);
    const auto unitsBegin = std::max(shadow->UnitBegin(first), shadow->importBase);
    const auto unitsEnd = std::min(shadow->UnitBegin(last + 1), shadow->ImportEnd());
    GuestMemory::CollectWritesUncached(unitsBegin, static_cast<std::size_t>(unitsEnd - unitsBegin));
    std::vector<VkBufferCopy> seeds;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> seeded;
    std::uint64_t seedBytes = 0;
    std::vector<std::uint64_t> seedUnits;
    std::uint64_t generation = 0;
    {
        std::lock_guard lock(registry.mutex);
        const auto* held = shadow->BufferSlabOf(first);
        if (held == nullptr || *held != slab) return std::nullopt;
        std::vector<std::uint8_t> changed, cpu;
        changedUnits(*shadow, first, last, changed, cpu);
        // Every stamp the walk above made is at most this generation: the seeded units are current
        // at it.
        generation = GuestMemory::TrackerGeneration();
        for (auto unit = first; unit <= last; ++unit) {
            const auto k = static_cast<std::size_t>(unit - first);
            if (shadow->generation[unit] != 0 && changed[k] == GuestMemory::BlockUnchanged) continue;
            // Never seeded, or the import was written since: results the slab still held lose to
            // the newer import bytes, as a publish would drop them.
            if (shadow->Live(unit)) (cpu[k] != 0 ? Stats().staleDroppedCpu : Stats().staleDroppedDriver).fetch_add(1, std::memory_order_relaxed);
            seedUnits.push_back(unit);
            const auto [unitBegin, unitEnd] = shadow->UnitRange(unit);
            if (unitEnd <= unitBegin) continue;
            const VkBufferCopy region{unitBegin - shadow->importBase, (unit - slab->firstUnit) * UnitBytes + (unitBegin - shadow->UnitBegin(unit)), unitEnd - unitBegin};
            if (!seeds.empty() && seeds.back().srcOffset + seeds.back().size == region.srcOffset && seeds.back().dstOffset + seeds.back().size == region.dstOffset) {
                seeds.back().size += region.size;
                seeded.back().second = unitEnd;
            } else {
                seeds.push_back(region);
                seeded.emplace_back(unitBegin, unitEnd);
            }
            seedBytes += region.size;
        }
        slab->lastUse = Recorder::Presents();
    }
    if (!seeds.empty()) recordSeeds(*shadow, *recorder, slab, seeds, seeded, seedBytes);
    if (!seedUnits.empty()) {
        // Only once the copies are recorded: a throw above leaves the units unseeded.
        std::lock_guard lock(registry.mutex);
        for (const auto unit : seedUnits) {
            shadow->generation[unit] = generation;
            shadow->published[unit] = 1;
        }
        shadow->Recount();
    }
    Stats().bufferUses.fetch_add(1, std::memory_order_relaxed);
    return BufferShadowBinding{slab->buffer, shadow->UnitBegin(slab->firstUnit), slab, std::move(pin)};
}

void MarkBufferShadowWritten(const HostImport& import, const ShadowSlab& slab, std::uint64_t begin, std::uint64_t end, std::uint64_t generation) {
    if (!BufferShadowEnabled() || end <= begin || generation == 0) return;
    auto& registry = Registry();
    std::uint64_t lost = 0;
    {
        std::lock_guard lock(registry.mutex);
        const auto shadow = find(registry, import);
        if (shadow == nullptr) return;
        const auto from = std::max(begin, shadow->importBase);
        const auto to = std::min(end, shadow->ImportEnd());
        if (to <= from) return;
        for (auto unit = shadow->UnitOf(from); unit <= shadow->UnitOf(to - 1); ++unit) {
            const auto* held = shadow->BufferSlabOf(unit);
            if (held == nullptr || held->get() != &slab) {
                // The slab is not the shadow's any more (a retire, a replaced import): the writer's
                // pin kept it from eviction, so nothing else takes it away.
                ++lost;
                continue;
            }
            shadow->generation[unit] = generation;
            shadow->published[unit] = 0;
        }
        shadow->Recount();
    }
    Stats().bufferWrites.fetch_add(1, std::memory_order_relaxed);
    if (lost != 0) {
        Stats().slabLost.fetch_add(lost, std::memory_order_relaxed);
        static std::atomic<int> reports{0};
        if (reports.fetch_add(1) < 4) std::fprintf(stderr, "[shadow] %llu written units of import 0x%llx lost their buffer slab before they were marked\n", static_cast<unsigned long long>(lost), static_cast<unsigned long long>(import.base));
    }
    // Builds reading the range in place see the change through their pending memos (fastRevalidate
    // and Revalidate's epoch gate), as after a retile: they publish it, or rebuild to bind the slab.
    StorageTexture::BumpPendingSerial();
}

bool BufferShadowServes(std::uint64_t begin, std::uint64_t end) {
    if (!BufferShadowEnabled() || end <= begin) return false;
    auto& registry = Registry();
    std::lock_guard lock(registry.mutex);
    return anyOverlapping(registry, begin, end, [&](const std::shared_ptr<UnitShadow>& shadow) {
        if (begin < shadow->importBase || end > shadow->ImportEnd()) return false;
        const auto first = shadow->UnitOf(begin);
        const auto last = shadow->UnitOf(end - 1);
        return std::any_of(shadow->bufferSlabs.begin(), shadow->bufferSlabs.end(), [&](const std::shared_ptr<ShadowSlab>& slab) { return slab->firstUnit <= first && last - slab->firstUnit < slab->units; });
    });
}

std::string ShadowReport() {
    auto& statistics = Stats();
    const auto take = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    const auto mib = [](unsigned long long bytes) { return bytes / 1048576.0; };
    char text[256];
    std::string line;
    if (!UnitShadowEnabled()) return "; shadow: off";
    std::snprintf(text, sizeof(text), "; shadow: retiled %llu/%.1f (refused %llu/%.1f to import), seeds %llu/%.1f, detiled from shadow %llu/%.1f, from import %llu/%.1f, tail publishes %llu, published", take(statistics.retiled), mib(take(statistics.retiledBytes)), take(statistics.refused), mib(take(statistics.refusedBytes)), take(statistics.seeds), mib(take(statistics.seedBytes)), take(statistics.detiledShadow), mib(take(statistics.detiledShadowBytes)), take(statistics.detiledImport), mib(take(statistics.detiledImportBytes)), take(statistics.tailPublishes));
    line += text;
    unsigned long long publishedTotal = 0, publishedBytesTotal = 0;
    std::string reasons;
    for (std::size_t i = 0; i < ReasonCount; ++i) {
        const auto count = take(statistics.published[i]);
        const auto bytes = take(statistics.publishedBytes[i]);
        publishedTotal += count;
        publishedBytesTotal += bytes;
        if (count == 0) continue;
        std::snprintf(text, sizeof(text), " %s %llu/%.1f", ReasonNames[i], count, mib(bytes));
        reasons += text;
    }
    const auto cpuDropped = take(statistics.staleDroppedCpu), driverDropped = take(statistics.staleDroppedDriver);
    unsigned long long liveSlabs = 0, liveBytes = 0, peakBytes = 0, retired = 0;
    {
        auto& registry = Registry();
        std::lock_guard lock(registry.mutex);
        liveSlabs = registry.liveSlabs;
        liveBytes = registry.liveBytes;
        peakBytes = registry.peakBytes;
        retired = registry.retired;
    }
    std::snprintf(text, sizeof(text), " %llu/%.1f by reason {%s }, stale-dropped %llu (cpu %llu, driver %llu), skipped-in-completion %llu, slabs %llu live %.1f MiB / peak %.1f MiB, evictions %llu, slab lost %llu, retired %llu, dropped on retire %llu, lost on retire %llu", publishedTotal, mib(publishedBytesTotal), reasons.c_str(), cpuDropped + driverDropped, cpuDropped, driverDropped, take(statistics.skippedInCompletion), liveSlabs, mib(liveBytes), mib(peakBytes), take(statistics.evictions), take(statistics.slabLost), retired, take(statistics.droppedOnRetire), take(statistics.lostOnRetire));
    line += text;
    if (BufferShadowEnabled()) {
        std::snprintf(text, sizeof(text), ", buffers made %llu/%.1f (refused %llu), uses %llu, seeds %llu/%.1f, writes %llu", take(statistics.buffersMade), mib(take(statistics.buffersMadeBytes)), take(statistics.buffersRefused), take(statistics.bufferUses), take(statistics.bufferSeeds), mib(take(statistics.bufferSeedBytes)), take(statistics.bufferWrites));
        line += text;
    }
    if (ShadowVerify()) {
        std::snprintf(text, sizeof(text), ", verified %llu, mismatches %llu", take(statistics.verified), take(statistics.mismatches));
        line += text;
    }
    return line;
}

namespace {

// APS5_BUFFER_SHADOW_OBSERVE: what the guards saw since the last [shadow-guard] line. The fault
// handler records into it, so a plain leaf mutex guards it and nothing is resolved under it.
struct GuardTouch {
    std::uint32_t thread;
    std::uintptr_t instruction;
    std::uintptr_t address;
    bool write;
    // The touching code's callers (the handler's own frames and the exception dispatch left out, or
    // the whole captured stack when the faulting frame is not found in it).
    std::array<std::uintptr_t, 16> callers{};
};

struct GuardObservation {
    std::mutex mutex;
    std::uint64_t writes = 0, guardedBytes = 0, refused = 0, refusedBytes = 0, touches = 0, touchWrites = 0;
    std::vector<GuardTouch> first;
    std::map<std::uintptr_t, std::uint64_t> byInstruction;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

GuardObservation& Observation() {
    static GuardObservation observation;
    return observation;
}

#ifdef _WIN32
// The guard handler: counts the touch and opens its whole unit (its neighbours' bytes are as
// current as the touched one's) until the next write guards it again.
void observeTouch(std::uintptr_t address, bool write, std::uintptr_t instruction) {
    auto& observation = Observation();
    {
        std::lock_guard lock(observation.mutex);
        ++observation.touches;
        if (write) ++observation.touchWrites;
        if (observation.first.size() < 12) {
            GuardTouch touch{static_cast<std::uint32_t>(GetCurrentThreadId()), instruction, address, write};
            // This stack runs from the handler through the exception dispatch to the faulting frame:
            // the frames after the one returning into the faulting function are its callers.
            std::array<void*, 40> frames{};
            const auto count = RtlCaptureStackBackTrace(0, static_cast<DWORD>(frames.size()), frames.data(), nullptr);
            USHORT start = 0;
            for (USHORT i = 0; i < count; ++i) {
                if (reinterpret_cast<std::uintptr_t>(frames[i]) == instruction) start = static_cast<USHORT>(i + 1);
            }
            for (std::size_t next = 0; start + next < count && next < touch.callers.size(); ++next) touch.callers[next] = reinterpret_cast<std::uintptr_t>(frames[start + next]);
            observation.first.push_back(touch);
        }
        if (observation.byInstruction.size() < 64 || observation.byInstruction.contains(instruction)) ++observation.byInstruction[instruction];
    }
    const auto unit = address & ~(UnitBytes - 1);
    GuestArena::GuestArenaUnguard_nid_postfix(reinterpret_cast<void*>(unit), UnitBytes);
}

// "module+0xoffset" for a code address, or the bare address.
std::string describeCode(std::uintptr_t address) {
    HMODULE module = nullptr;
    char path[MAX_PATH] = "";
    char text[MAX_PATH + 32];
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module) && GetModuleFileNameA(module, path, sizeof(path)) != 0) {
        const char* name = std::strrchr(path, '\\');
        std::snprintf(text, sizeof(text), "%s+0x%llx", name != nullptr ? name + 1 : path, static_cast<unsigned long long>(address - reinterpret_cast<std::uintptr_t>(module)));
    } else {
        std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(address));
    }
    return text;
}

std::string threadName(std::uint32_t id) {
    std::string name;
    if (HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, id)) {
        PWSTR description = nullptr;
        if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
            char text[128];
            if (WideCharToMultiByte(CP_UTF8, 0, description, -1, text, sizeof(text), nullptr, nullptr) > 0) name = text;
            LocalFree(description);
        }
        CloseHandle(thread);
    }
    return name;
}

// The [shadow-guard] line, every 10 s with anything to say.
void reportObservation() {
    auto& observation = Observation();
    std::uint64_t writes = 0, guardedBytes = 0, refused = 0, refusedBytes = 0, touches = 0, touchWrites = 0;
    std::vector<GuardTouch> first;
    std::map<std::uintptr_t, std::uint64_t> byInstruction;
    {
        std::lock_guard lock(observation.mutex);
        const auto now = std::chrono::steady_clock::now();
        if (now - observation.lastReport < std::chrono::seconds(10)) return;
        observation.lastReport = now;
        writes = std::exchange(observation.writes, 0);
        guardedBytes = std::exchange(observation.guardedBytes, 0);
        refused = std::exchange(observation.refused, 0);
        refusedBytes = std::exchange(observation.refusedBytes, 0);
        touches = std::exchange(observation.touches, 0);
        touchWrites = std::exchange(observation.touchWrites, 0);
        first.swap(observation.first);
        byInstruction.swap(observation.byInstruction);
    }
    if (writes == 0 && refused == 0 && touches == 0) return;
    std::vector<std::pair<std::uint64_t, std::uintptr_t>> hot;
    for (const auto& [instruction, count] : byInstruction) hot.emplace_back(count, instruction);
    std::sort(hot.rbegin(), hot.rend());
    std::string byCode;
    for (std::size_t i = 0; i < hot.size() && i < 8; ++i) byCode += " " + describeCode(hot[i].second) + " x" + std::to_string(hot[i].first);
    std::string firstTouches;
    for (std::size_t i = 0; i < first.size(); ++i) {
        const auto& touch = first[i];
        char text[96];
        std::snprintf(text, sizeof(text), " %s of 0x%llx by thread %u '", touch.write ? "write" : "read", static_cast<unsigned long long>(touch.address), touch.thread);
        firstTouches += text + threadName(touch.thread) + "' at " + describeCode(touch.instruction);
        // The callers of the first three.
        for (const auto caller : touch.callers) {
            if (caller != 0 && i < 3) firstTouches += " <- " + describeCode(caller);
        }
        firstTouches += ";";
    }
    std::fprintf(stderr, "[shadow-guard] observe (10 s): %llu writes guarded (%.1f MiB), %llu refused (%.1f MiB); %llu host touches (%llu writes); by code:%s; first:%s\n", static_cast<unsigned long long>(writes), static_cast<double>(guardedBytes) / 1048576.0, static_cast<unsigned long long>(refused), static_cast<double>(refusedBytes) / 1048576.0, static_cast<unsigned long long>(touches), static_cast<unsigned long long>(touchWrites), byCode.c_str(), firstTouches.c_str());
}
#endif

}

bool BufferShadowObserved() {
#ifdef _WIN32
    static const bool observed = std::getenv("APS5_BUFFER_SHADOW_OBSERVE") != nullptr && !BufferShadowEnabled();
    return observed;
#else
    return false;
#endif
}

void ObserveBufferShadowWrite(std::uint64_t begin, std::uint64_t end) {
#ifdef _WIN32
    if (!BufferShadowObserved()) return;
    static const bool installed = [] {
        GuestArena::GuestArenaSetGuardHandler_nid_postfix(&observeTouch);
        return true;
    }();
    static_cast<void>(installed);
    // Whole units only: a partly covered one holds bytes beyond the range.
    const auto first = (begin + UnitBytes - 1) & ~(UnitBytes - 1);
    const auto last = end & ~(UnitBytes - 1);
    if (last > first) {
        const auto bytes = static_cast<std::size_t>(last - first);
        const bool guarded = GuestArena::GuestArenaGuard_nid_postfix(reinterpret_cast<void*>(first), bytes);
        {
            auto& observation = Observation();
            std::lock_guard lock(observation.mutex);
            ++(guarded ? observation.writes : observation.refused);
            (guarded ? observation.guardedBytes : observation.refusedBytes) += bytes;
        }
        static std::atomic<int> refusals{0};
        if (!guarded && refusals.fetch_add(1) < 4) {
            MEMORY_BASIC_INFORMATION memory{};
            VirtualQuery(reinterpret_cast<const void*>(first), &memory, sizeof(memory));
            std::fprintf(stderr, "[shadow-guard] cannot guard 0x%llx+0x%llx: state 0x%lx type 0x%lx protection 0x%lx\n", static_cast<unsigned long long>(first), static_cast<unsigned long long>(bytes), memory.State, memory.Type, memory.Protect);
        }
    }
    reportObservation();
#else
    static_cast<void>(begin);
    static_cast<void>(end);
#endif
}

void OpenShadowGuards(std::uint64_t begin, std::uint64_t end) {
#ifdef _WIN32
    if (!BufferShadowObserved() || end <= begin) return;
    GuestArena::GuestArenaUnguard_nid_postfix(reinterpret_cast<void*>(begin), static_cast<std::size_t>(end - begin));
#else
    static_cast<void>(begin);
    static_cast<void>(end);
#endif
}

}
