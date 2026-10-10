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
#include <map>
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>

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
// the entries (default 32768, the least recently used go first); APS5_NO_STAGE_MEMO_VALUES=1 takes
// a write stamp over an entry's words as a miss without comparing them.
bool StageMemoEnabled() {
    static const bool enabled = std::getenv("APS5_NO_STAGE_MEMO") == nullptr;
    return enabled;
}

bool StageMemoValues() {
    static const bool values = std::getenv("APS5_NO_STAGE_MEMO_VALUES") == nullptr;
    return values;
}

bool VerifyStageMemo() {
    static const bool verify = std::getenv("APS5_VERIFY_STAGE_MEMO") != nullptr;
    return verify;
}

// The served copies a memo slot keeps (StageMemoSlot::served): APS5_STAGE_MEMO_COPIES, 1 to 8.
std::size_t StageMemoCopies() {
    static const std::size_t copies = [] {
        const char* text = std::getenv("APS5_STAGE_MEMO_COPIES");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 8ull;
        return static_cast<std::size_t>(std::clamp<unsigned long long>(value, 1ull, 8ull));
    }();
    return copies;
}

std::size_t StageMemoCapacity() {
    static const std::size_t capacity = [] {
        const char* text = std::getenv("APS5_STAGE_MEMO_ENTRIES");
        // 8192 thrashed on the S2 menu page (about 104k evictions per 10 s).
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 32768ull;
        return static_cast<std::size_t>(std::max<unsigned long long>(value, 1ull));
    }();
    return capacity;
}

// Whether a draw cache variant keeps its own copy of a result the stage memo served
// (cacheDrawStages); APS5_NO_VARIANT_RESULT_COPIES=1 holds the memo's, as before.
bool VariantResultCopies() {
    static const bool copies = std::getenv("APS5_NO_VARIANT_RESULT_COPIES") == nullptr;
    return copies;
}

// APS5_PROFILE_DRAW: the memo's outcomes, every 10 s on a [stage-memo] line.
struct StageMemoCounters {
    std::atomic<std::uint64_t> lookups{0}, hits{0}, revalidated{0}, absent{0}, mappings{0}, changed{0}, pending{0}, skipped{0};
    std::atomic<std::uint64_t> inserts{0}, refusedPatches{0}, refusedMismatch{0}, refusedUnwatched{0}, refusedUnstable{0}, evictions{0};
    std::atomic<std::uint64_t> verified{0}, differed{0};
    // Hits that copied the entry's result: into an empty place of the slot, or because a draw held
    // every served copy.
    std::atomic<std::uint64_t> copiedEmpty{0}, copiedHeld{0};
    // The served copies such a copy found held, by their use count: 2 (one reference besides the
    // slot's, a draw cache variant's), 3 (a draw in flight: its stage capture and its results), more.
    std::atomic<std::uint64_t> heldTwo{0}, heldThree{0}, heldMore{0};
    std::atomic<std::int64_t> lastReport{0};
};

StageMemoCounters& stageMemoCounters() {
    static StageMemoCounters counters;
    return counters;
}

bool StageMemoProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

// A counter of the [stage-memo] line, counted only while the line is printed: a locked add on a
// shared line at every lookup and hit otherwise.
void countStageMemo(std::atomic<std::uint64_t>& counter) {
    if (StageMemoProfiled()) counter.fetch_add(1, std::memory_order_relaxed);
}

void reportStageMemo(std::size_t entries) {
    if (!StageMemoProfiled()) return;
    auto& counters = stageMemoCounters();
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load(std::memory_order_relaxed);
    if (now - last < 10000 || !counters.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const auto take = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    const auto lookups = take(counters.lookups);
    const auto hits = take(counters.hits);
    std::fprintf(stderr, "[stage-memo] %llu lookups (10 s): %llu hits (%.1f%%, %llu after comparing stamped words; result copied by %llu into an empty place, %llu with every served copy held, those held by use count 2/3/more %llu/%llu/%llu); misses: no entry %llu, mappings changed %llu, words changed %llu, pending results %llu, not tried (changing key) %llu; inserts %llu, refused: no patches %llu, patch mismatch %llu, unwatched %llu, unstable %llu; evictions %llu, %zu entries; verify: %llu compared, %llu differed\n", lookups, hits, lookups != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(lookups) : 0.0, take(counters.revalidated), take(counters.copiedEmpty), take(counters.copiedHeld), take(counters.heldTwo), take(counters.heldThree), take(counters.heldMore), take(counters.absent), take(counters.mappings), take(counters.changed), take(counters.pending), take(counters.skipped), take(counters.inserts), take(counters.refusedPatches), take(counters.refusedMismatch), take(counters.refusedUnwatched), take(counters.refusedUnstable), take(counters.evictions), entries, take(counters.verified), take(counters.differed));
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
    // MurmurHash3's finalizer: the memo's table indexes by the low bits, which FNV's last multiply
    // leaves depending on the inputs' low bits alone.
    hash ^= hash >> 33u;
    hash *= 0xff51afd7ed558ccdull;
    hash ^= hash >> 33u;
    hash *= 0xc4ceb9fe1a85ec53ull;
    hash ^= hash >> 33u;
    return hash;
}

// A thread's memo keys by source (ShaderRecompiler::UserDataKeyFor; a source entry lives as long as
// the process, its key with it): found per stage without a lock, through a direct-mapped array in
// front of the map that owns them. A source without a key yet (no plan before its first capture)
// is asked again.
struct StageMemoKeys {
    struct Way {
        const void* source = nullptr;
        const ShaderRecompiler::UserDataKey* key = nullptr;
    };
    std::array<Way, 256> ways{};
    std::unordered_map<const void*, std::shared_ptr<const ShaderRecompiler::UserDataKey>> owned;

    const ShaderRecompiler::UserDataKey* Find(const ShaderRecompiler::SourceHandle& handle) {
        const void* source = handle.source.get();
        auto& way = ways[(reinterpret_cast<std::uintptr_t>(source) >> 4) * 0x9e3779b97f4a7c15ull >> 56];
        if (way.source == source) return way.key;
        auto& known = owned[source];
        if (known == nullptr) known = ShaderRecompiler::UserDataKeyFor(handle);
        if (known != nullptr) way = {source, known.get()};
        return known.get();
    }
};

enum class StageMemoMiss { None, Absent, Mappings, Changed, Pending, Skipped };

// Whether a key with `misses` misses in a row is tried (validated, and stored again after a miss):
// at powers of two of them, then every 64th. A key whose words keep changing stops paying for the
// validation's collects (and their write-watch re-arming), but is found again once it settles.
bool stageMemoTried(std::uint32_t misses) {
    return misses < 64u ? (misses & (misses - 1u)) == 0 : misses % 64u == 0;
}

// Whether an entry's captured words are still the guest's: no mapping changed since before the
// capture, no write stamped over them since `generation` (CPU stores collected first, the newest
// generation the collects returned left in `collected`; recorded GPU stores stamp when recorded),
// and no storage image results pending and no unit shadow live over them (bytes the memory does not
// hold yet).
StageMemoMiss stageMemoCurrent(const StageMemoEntry& entry, std::uint64_t generation, std::uint64_t& collected) {
    if (entry.mappings != GuestAllocations::GuestAllocationsGeneration_nid_postfix()) return StageMemoMiss::Mappings;
    if (entry.runs.empty()) return StageMemoMiss::None;
    struct QueriesTag {};
    auto& queries = ThreadScratch<std::vector<GuestMemory::UnchangedQuery>, QueriesTag>();
    queries.clear();
    for (const auto& [begin, end] : entry.runs) {
        const auto bytes = static_cast<std::size_t>(end - begin);
        collected = std::max(collected, GuestMemory::CollectWrites(begin, bytes));
        queries.push_back({begin, bytes, generation});
    }
    if (!GuestMemory::UnchangedSinceAll(queries)) return StageMemoMiss::Changed;
    if (Graphics::StorageTexture::AnyPendingOverlaps(entry.runs) || Graphics::AnyShadowedOverlaps(entry.runs)) return StageMemoMiss::Pending;
    return StageMemoMiss::None;
}

// APS5_TRACE_MEMO_CHURN=1: what the stage memo's misses change, every 10 s on a [stage-memo-churn]
// line. Each missed stage's capture is compared with the last capture of the same memo key (kept
// whether or not the memo stored it): its regions moved (other addresses or sizes), held the same
// words (the miss cost the capture and recompile for nothing: a stale entry, a backed-off key), or
// changed in place. A change in place is then tried as a memo that let captured words differ would
// serve it: the last capture's result with this stage's user words patched in must have the same
// variant and binding shapes, and every word it still differs in must be a region word that
// changed at the same position from the last capture's value to this one's (a word the result only
// copies: a buffer or image address, flattened data). Profile-only: it copies each missed stage's
// words and computes its patches.
bool MemoChurnTraced() {
    static const bool traced = std::getenv("APS5_TRACE_MEMO_CHURN") != nullptr;
    return traced;
}

struct MemoChurn {
    struct Last {
        std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> regions;
        std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
        std::shared_ptr<const std::vector<ShaderRecompiler::UserDataPatch>> patches;
    };
    AgcDriver::Mutex mutex;
    std::unordered_map<std::uint64_t, Last> last;
    // Misses by the memo's reason (StageMemoMiss's order), then by what the capture changed.
    std::array<std::uint64_t, 6> reasons{};
    std::uint64_t first = 0, moved = 0, identical = 0, changed = 0, served = 0, shapes = 0, noPatches = 0, unexplained = 0;
    std::uint64_t wordsExplained = 0, wordsUnexplained = 0, pushUnexplained = 0;
    // The explained words by the binding's DescriptorRole (the last place: push constants), and
    // within a buffer's V# (4 words) and an image's T# (8 words) by word.
    std::array<std::uint64_t, 9> explainedRoles{};
    std::array<std::uint64_t, 4> explainedBufferWords{};
    std::array<std::uint64_t, 8> explainedImageWords{};
    std::map<std::uint64_t, std::uint64_t> unexplainedPrograms;
    std::vector<std::string> examples;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

void traceMemoChurn(std::uint64_t hash, StageMemoMiss miss, const ShaderRecompiler::SourceHandle& handle, const ShaderRecompiler::RecompileRequest& request, std::span<const ShaderRecompiler::MemoryRegion> recent, const std::shared_ptr<const ShaderRecompiler::RecompileResult>& compiled, std::uint64_t code) {
    static MemoChurn churn;
    std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> regions;
    regions.reserve(recent.size());
    for (const auto& region : recent) {
        std::vector<std::uint32_t> words(region.bytes.size() / sizeof(std::uint32_t));
        std::memcpy(words.data(), region.bytes.data(), words.size() * sizeof(std::uint32_t));
        regions.emplace_back(region.guestAddress, std::move(words));
    }
    auto patches = ShaderRecompiler::UserDataPatchesFor(handle, *compiled);
    std::lock_guard lock(churn.mutex);
    ++churn.reasons[static_cast<std::size_t>(miss)];
    if (churn.last.size() >= 65536) churn.last.clear();
    auto& last = churn.last[hash];
    if (last.result == nullptr) {
        ++churn.first;
    } else {
        bool sameAddresses = last.regions.size() == regions.size();
        for (std::size_t r = 0; sameAddresses && r < regions.size(); ++r) sameAddresses = last.regions[r].first == regions[r].first && last.regions[r].second.size() == regions[r].second.size();
        if (!sameAddresses) {
            ++churn.moved;
        } else if (last.regions == regions) {
            ++churn.identical;
        } else if (last.patches == nullptr) {
            ++churn.changed;
            ++churn.noPatches;
        } else {
            ++churn.changed;
            auto patched = *last.result;
            ShaderRecompiler::PatchOverUserData(request, patched, *last.patches);
            const auto& fresh = *compiled;
            bool sameShapes = patched.variantId == fresh.variantId && patched.bindings.size() == fresh.bindings.size() && patched.pushConstants.size() == fresh.pushConstants.size();
            for (std::size_t b = 0; sameShapes && b < fresh.bindings.size(); ++b) {
                const auto& left = patched.bindings[b];
                const auto& right = fresh.bindings[b];
                sameShapes = left.role == right.role && left.kind == right.kind && left.count == right.count && left.guestDescriptor.size() == right.guestDescriptor.size();
            }
            if (!sameShapes) {
                ++churn.shapes;
            } else {
                // A region word at the same position that went from `before` to `after`.
                const auto explained = [&](std::uint32_t before, std::uint32_t after) {
                    for (std::size_t r = 0; r < regions.size(); ++r) {
                        const auto& old = last.regions[r].second;
                        const auto& now = regions[r].second;
                        for (std::size_t p = 0; p < now.size(); ++p) {
                            if (old[p] == before && now[p] == after) return true;
                        }
                    }
                    return false;
                };
                bool all = true;
                for (std::size_t b = 0; b < fresh.bindings.size(); ++b) {
                    const auto& left = patched.bindings[b].guestDescriptor;
                    const auto& right = fresh.bindings[b].guestDescriptor;
                    for (std::size_t w = 0; w < right.size(); ++w) {
                        if (left[w] == right[w]) continue;
                        if (explained(left[w], right[w])) {
                            ++churn.wordsExplained;
                            const auto role = fresh.bindings[b].role;
                            ++churn.explainedRoles[std::min<std::size_t>(static_cast<std::size_t>(role), 7)];
                            if (role == ShaderRecompiler::DescriptorRole::GuestBuffers) ++churn.explainedBufferWords[w % 4];
                            if (role == ShaderRecompiler::DescriptorRole::GuestImages) ++churn.explainedImageWords[w % 8];
                            continue;
                        }
                        ++churn.wordsUnexplained;
                        if (all && churn.examples.size() < 12) {
                            char text[200];
                            std::snprintf(text, sizeof(text), "program 0x%llx binding %zu (role %d) word %zu: 0x%08x -> 0x%08x", static_cast<unsigned long long>(code), b, static_cast<int>(fresh.bindings[b].role), w, left[w], right[w]);
                            churn.examples.emplace_back(text);
                        }
                        all = false;
                    }
                }
                for (std::size_t offset = 0; offset + sizeof(std::uint32_t) <= fresh.pushConstants.size(); offset += sizeof(std::uint32_t)) {
                    std::uint32_t before = 0, after = 0;
                    std::memcpy(&before, patched.pushConstants.data() + offset, sizeof(before));
                    std::memcpy(&after, fresh.pushConstants.data() + offset, sizeof(after));
                    if (before == after) continue;
                    if (explained(before, after)) {
                        ++churn.wordsExplained;
                        ++churn.explainedRoles[8];
                        continue;
                    }
                    ++churn.pushUnexplained;
                    all = false;
                }
                if (all) {
                    ++churn.served;
                } else {
                    ++churn.unexplained;
                    ++churn.unexplainedPrograms[code];
                }
            }
        }
    }
    last = {std::move(regions), compiled, std::move(patches)};
    const auto now = std::chrono::steady_clock::now();
    if (now - churn.lastReport < std::chrono::seconds(10)) return;
    churn.lastReport = now;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> programs;
    for (const auto& [program, count] : churn.unexplainedPrograms) programs.emplace_back(count, program);
    std::sort(programs.rbegin(), programs.rend());
    std::string top;
    for (std::size_t i = 0; i < programs.size() && i < 6; ++i) {
        char text[48];
        std::snprintf(text, sizeof(text), " 0x%llx x%llu", static_cast<unsigned long long>(programs[i].second), static_cast<unsigned long long>(programs[i].first));
        top += text;
    }
    const auto total = std::accumulate(churn.reasons.begin(), churn.reasons.end(), std::uint64_t{0});
    std::fprintf(stderr, "[stage-memo-churn] %llu misses traced (10 s): words changed %llu, not tried %llu, no entry %llu, mappings %llu, pending %llu; first capture of the key %llu, regions moved %llu, same words %llu, changed in place %llu: served by copied words %llu, variant or shape changed %llu, no patches %llu, unexplained %llu (result words explained %llu, unexplained %llu, push dwords unexplained %llu); unexplained by program:%s\n", static_cast<unsigned long long>(total), static_cast<unsigned long long>(churn.reasons[static_cast<std::size_t>(StageMemoMiss::Changed)]), static_cast<unsigned long long>(churn.reasons[static_cast<std::size_t>(StageMemoMiss::Skipped)]), static_cast<unsigned long long>(churn.reasons[static_cast<std::size_t>(StageMemoMiss::Absent)]), static_cast<unsigned long long>(churn.reasons[static_cast<std::size_t>(StageMemoMiss::Mappings)]), static_cast<unsigned long long>(churn.reasons[static_cast<std::size_t>(StageMemoMiss::Pending)]), static_cast<unsigned long long>(churn.first), static_cast<unsigned long long>(churn.moved), static_cast<unsigned long long>(churn.identical), static_cast<unsigned long long>(churn.changed), static_cast<unsigned long long>(churn.served), static_cast<unsigned long long>(churn.shapes), static_cast<unsigned long long>(churn.noPatches), static_cast<unsigned long long>(churn.unexplained), static_cast<unsigned long long>(churn.wordsExplained), static_cast<unsigned long long>(churn.wordsUnexplained), static_cast<unsigned long long>(churn.pushUnexplained), top.c_str());
    const auto& roles = churn.explainedRoles;
    const auto& buffer = churn.explainedBufferWords;
    const auto& image = churn.explainedImageWords;
    std::fprintf(stderr, "[stage-memo-churn] explained words by binding: buffers %llu (V# words 0-3: %llu/%llu/%llu/%llu), images %llu (T# words 0-7: %llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu), samplers %llu, gds/bda/fault %llu, flattened srt %llu, shader data %llu, push constants %llu\n", static_cast<unsigned long long>(roles[0]), static_cast<unsigned long long>(buffer[0]), static_cast<unsigned long long>(buffer[1]), static_cast<unsigned long long>(buffer[2]), static_cast<unsigned long long>(buffer[3]), static_cast<unsigned long long>(roles[1]), static_cast<unsigned long long>(image[0]), static_cast<unsigned long long>(image[1]), static_cast<unsigned long long>(image[2]), static_cast<unsigned long long>(image[3]), static_cast<unsigned long long>(image[4]), static_cast<unsigned long long>(image[5]), static_cast<unsigned long long>(image[6]), static_cast<unsigned long long>(image[7]), static_cast<unsigned long long>(roles[2]), static_cast<unsigned long long>(roles[3] + roles[4] + roles[5]), static_cast<unsigned long long>(roles[6]), static_cast<unsigned long long>(roles[7]), static_cast<unsigned long long>(roles[8]));
    for (const auto& example : churn.examples) std::fprintf(stderr, "[stage-memo-churn]   %s\n", example.c_str());
    churn.explainedRoles = {};
    churn.explainedBufferWords = {};
    churn.explainedImageWords = {};
    churn.reasons = {};
    churn.first = churn.moved = churn.identical = churn.changed = churn.served = churn.shapes = churn.noPatches = churn.unexplained = 0;
    churn.wordsExplained = churn.wordsUnexplained = churn.pushUnexplained = 0;
    churn.unexplainedPrograms.clear();
    churn.examples.clear();
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

std::shared_ptr<const ShaderRecompiler::RecompileResult> Driver::compileDrawStage(std::size_t i, std::uint32_t pushOffset, const QueueState& queue, const Submission& submission, const std::vector<DrawProgram>& programs, const Graphics::State& graphics, const ShaderRecompiler::ShaderPixelStageInfo& pixel, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, std::vector<ShaderRecompiler::RecompileRequest>& requests, std::vector<ShaderRecompiler::MemoryRegion>& memory, const std::vector<ShaderRecompiler::LinkedProgram>& linked, const Pm4::DrawParameters& drawParameters, const std::shared_ptr<VulkanDevice>& localDevice, ShaderMemory& shaderMemory, std::vector<StageCapture>& stageCaptures, std::vector<bool>& recompiled, bool drawHit, bool wantRegions, const std::vector<std::shared_ptr<DispatchVariant>>& matched, const std::vector<std::vector<ShaderRecompiler::MemoryRegion>>& matchedRegions, bool profile, std::uint64_t dumpTarget, std::uint64_t dumpSlot1, std::uint64_t& captures, DrawPhaseTiming& phaseTiming, std::array<double, DrawDriverPhaseCount>& phaseMs, std::string& rejected) {
    using Stage = ShaderRecompiler::ShaderStage;
    phaseTiming.Phase(DrawRowVectors);
    const auto& program = programs[i];
    const auto waveSize = program.binary.stage == Stage::Fragment ? graphics.stages.fragmentWaveSize : graphics.stages.vertexWaveSize;
    // The stage's request, filled in place (see PreparedDraw::requests), the draw's decoded vertex
    // inputs borrowed rather than copied: vertexInfos outlives every use of it below.
    auto& request = requests[i];
    request.shader = program.binary;
    request.context.waveSize = waveSize;
    request.context.userDataBaseRegister = program.firstUserSgpr;
    request.context.userData = program.userData;
    request.context.compute = std::nullopt;
    if (program.binary.stage == Stage::Fragment) request.context.pixel = pixel;
    else request.context.pixel = std::nullopt;
    request.context.vertex.Borrow(vertexInfos[i] ? &*vertexInfos[i] : nullptr);
    request.context.memory = memory;
    request.target = localDevice->Target();
    request.layout = {0, 0, pushOffset, (graphics.stages.mesh ? ShaderRecompiler::MeshDrawPushOffsetBytes : Graphics::PipelinePushConstantBytes) - pushOffset};
    request.graphics = ShaderRecompiler::GraphicsCompileContext{program.firstUserSgpr, linked, graphics.stages.mesh, graphics.stages.tessellation, {drawParameters.indexAddress, drawParameters.indexCount, drawParameters.indexSize, drawParameters.instanceCount}};
    request.useCache = true;
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
    const ShaderRecompiler::UserDataKey* memoKey = nullptr;
    if (memoize && handle != nullptr && dumpTarget == 0 && dumpSlot1 == 0 && !ShaderRecompiler::DebugProbeActive()) {
        struct MemoKeysTag {};
        memoKey = ThreadScratch<StageMemoKeys, MemoKeysTag>().Find(*handle);
    }
    struct MemoKeyTag {};
    auto& memoWords = ThreadScratch<std::vector<std::uint32_t>, MemoKeyTag>();
    std::uint64_t memoHash = 0;
    const auto mappings = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    std::shared_ptr<const ShaderRecompiler::RecompileResult> memoResult;
    // Whether a capture of this stage is stored: always for a new key, at the misses in a row
    // stageMemoTried picks otherwise.
    bool memoStore = true;
    // Why the memo missed (APS5_TRACE_MEMO_CHURN).
    auto memoMiss = StageMemoMiss::Absent;
    if (memoKey != nullptr) {
        auto& counters = stageMemoCounters();
        countStageMemo(counters.lookups);
        memoHash = stageMemoKey(*memoKey, handle->source.get(), pushOffset, program.userData, memoWords);
        std::shared_ptr<const StageMemoEntry> entry;
        std::shared_ptr<ShaderRecompiler::RecompileResult> served;
        std::size_t entries = 0;
        auto miss = StageMemoMiss::Absent;
        std::uint64_t collected = 0;
        // The slot after the lookup's outcome, under the memo's lock.
        const auto settle = [&](StageMemoSlot& slot) {
            if (miss == StageMemoMiss::None) {
                slot.misses = 0;
                // Held by nothing but the slot: no draw is using it any more.
                // use_count reads relaxed: the fence orders the patch after the last holder's
                // release (a draw thread's record drops its reference there).
                for (std::size_t copy = 0; copy < StageMemoCopies(); ++copy) {
                    if (slot.served[copy] == nullptr || slot.served[copy].use_count() != 1) continue;
                    std::atomic_thread_fence(std::memory_order_acquire);
                    served = slot.served[copy];
                    break;
                }
            } else {
                ++slot.misses;
                memoStore = stageMemoTried(slot.misses);
            }
        };
        {
            std::lock_guard lock(stageMemoMutex);
            entries = stageMemo.size();
            auto* found = stageMemo.find(memoHash);
            if (found != nullptr && found->source == handle->source.get() && found->pushOffset == pushOffset && found->entry->key == memoWords) {
                auto& slot = *found;
                entry = slot.entry;
                stageMemoOrder.splice(stageMemoOrder.end(), stageMemoOrder, slot.order);
                miss = stageMemoTried(slot.misses) ? stageMemoCurrent(*entry, slot.generation, collected) : StageMemoMiss::Skipped;
                if (miss != StageMemoMiss::Changed || !StageMemoValues()) settle(slot);
            }
        }
        // A store stamped the words' 64 KiB blocks (any store near them does): the words decide,
        // compared outside the lock after the collects above, as the insert proves a capture's. Equal,
        // they were the guest's at that collect, and a later store stamps newer.
        if (miss == StageMemoMiss::Changed && StageMemoValues()) {
            if (Graphics::StorageTexture::AnyPendingOverlaps(entry->runs) || Graphics::AnyShadowedOverlaps(entry->runs)) {
                miss = StageMemoMiss::Pending;
            } else {
                struct MemoWordsTag {};
                ThreadScratchLease<std::vector<ShaderRecompiler::MemoryRegion>, MemoWordsTag> words;
                for (const auto& [address, values] : entry->regions) words.value.push_back({address, std::as_bytes(std::span(values))});
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
                if (captureStable(words.value)) {
                    miss = StageMemoMiss::None;
                    countStageMemo(counters.revalidated);
                }
            }
            std::lock_guard lock(stageMemoMutex);
            auto* found = stageMemo.find(memoHash);
            if (found != nullptr && found->entry == entry) {
                if (miss == StageMemoMiss::None) found->generation = std::max(found->generation, collected);
                settle(*found);
            }
        }
        switch (miss) {
            case StageMemoMiss::None: break;
            case StageMemoMiss::Absent: countStageMemo(counters.absent); break;
            case StageMemoMiss::Mappings: countStageMemo(counters.mappings); break;
            case StageMemoMiss::Changed: countStageMemo(counters.changed); break;
            case StageMemoMiss::Pending: countStageMemo(counters.pending); break;
            case StageMemoMiss::Skipped: countStageMemo(counters.skipped); break;
        }
        if (miss == StageMemoMiss::None) {
            const bool fresh = served == nullptr;
            if (fresh) served = std::make_shared<ShaderRecompiler::RecompileResult>(*entry->result);
            ShaderRecompiler::PatchOverUserData(request, *served, *entry->patches);
            if (fresh) {
                std::lock_guard lock(stageMemoMutex);
                auto* found = stageMemo.find(memoHash);
                // The copy takes an empty place, else one whose result a draw still holds.
                if (found != nullptr && found->entry == entry) {
                    const auto copies = std::span(found->served).first(StageMemoCopies());
                    auto place = std::find(copies.begin(), copies.end(), nullptr);
                    countStageMemo(place != copies.end() ? counters.copiedEmpty : counters.copiedHeld);
                    if (place == copies.end() && StageMemoProfiled()) {
                        for (const auto& copy : copies) countStageMemo(copy.use_count() == 2 ? counters.heldTwo : copy.use_count() == 3 ? counters.heldThree : counters.heldMore);
                    }
                    if (place == copies.end()) place = std::find_if(copies.begin(), copies.end(), [](const auto& copy) { return copy.use_count() > 1; });
                    if (place != copies.end()) *place = served;
                }
            }
            memoResult = std::move(served);
            if (!VerifyStageMemo()) {
                countStageMemo(counters.hits);
                shaderMemory.Seed(entry->regions);
                if (wantRegions) {
                    stageCapture.regions.clear();
                    for (const auto& [address, words] : entry->regions) stageCapture.regions.push_back({address, std::as_bytes(std::span(words))});
                    stageCapture.memo = entry;
                }
                regionsDone();
                phaseTiming.Phase(DrawRowRecompile);
                reportStageMemo(entries);
                // Only the draw cache and the traces read the stage capture's result (all under
                // wantRegions): otherwise the result goes to the draw alone, without the capture's
                // reference to drop again on the draw thread.
                if (!wantRegions) return memoResult;
                stageCapture.compiled = std::move(memoResult);
                return stageCapture.compiled;
            }
        }
        memoMiss = miss;
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
    auto compiled = reuseCapture ? ShaderRecompiler::Recompile(request, *capture, &memoHit) : std::make_shared<const ShaderRecompiler::RecompileResult>(ShaderRecompiler::Recompile(request));
    if (subPhases) {
        part(memoHit ? StageTimes::RecompileHit : StageTimes::RecompileMiss);
        reportStageTimes();
    }
    phaseTiming.Phase(DrawRowRecompile);
    if (memoKey != nullptr && MemoChurnTraced()) traceMemoChurn(memoHash, memoMiss, *handle, request, recent, compiled, program.binary.codeAddress);
    if (memoKey != nullptr && (memoStore || memoResult != nullptr)) insertStageMemo(memoHash, handle, pushOffset, memoWords, mappings, compiled, memoResult, request.context.userData, recent, program.binary.codeAddress);
    // Shared, not copied: the result is immutable (a copy cost a few dozen allocations per stage).
    // The stage capture keeps it for its readers only (see the memo's hit above).
    if (wantRegions) stageCapture.compiled = compiled;
    return compiled;
}

void Driver::insertStageMemo(std::uint64_t hash, const std::shared_ptr<const ShaderRecompiler::SourceHandle>& handle, std::uint32_t pushOffset, const std::vector<std::uint32_t>& key, std::uint64_t mappings, const std::shared_ptr<const ShaderRecompiler::RecompileResult>& result, const std::shared_ptr<const ShaderRecompiler::RecompileResult>& memoResult, std::span<const std::uint32_t> userData, std::span<const ShaderRecompiler::MemoryRegion> recent, std::uint64_t codeAddress) {
    auto& counters = stageMemoCounters();
    if (memoResult != nullptr) {
        // APS5_VERIFY_STAGE_MEMO: the memo's result against the capture's.
        countStageMemo(counters.verified);
        const auto difference = stageMemoDifference(*memoResult, *result);
        if (!difference.empty()) {
            countStageMemo(counters.differed);
            static std::atomic<int> reports{0};
            if (reports.fetch_add(1) < 32) std::fprintf(stderr, "[stage-memo] verify: program 0x%llx: %s\n", static_cast<unsigned long long>(codeAddress), difference.c_str());
        }
    }
    const auto patches = ShaderRecompiler::UserDataPatchesFor(*handle, *result);
    if (patches == nullptr) {
        countStageMemo(counters.refusedPatches);
        return;
    }
    // Every copy the analysis names must hold in this result (an image the materializer cleared as
    // invalid, say, holds zeros where the user words are not): the key would let the user words move
    // where the result does not follow them.
    for (const auto& patch : *patches) {
        if (patch.userWord >= userData.size()) {
            countStageMemo(counters.refusedMismatch);
            return;
        }
        const auto value = userData[patch.userWord];
        std::uint32_t held = 0;
        if (patch.binding == ShaderRecompiler::UserDataPatch::PushConstants) {
            if ((static_cast<std::size_t>(patch.word) + 1u) * sizeof(held) > result->pushConstants.size()) {
                countStageMemo(counters.refusedMismatch);
                return;
            }
            std::memcpy(&held, result->pushConstants.data() + static_cast<std::size_t>(patch.word) * sizeof(held), sizeof(held));
        } else {
            if (patch.binding >= result->bindings.size() || patch.word >= result->bindings[patch.binding].guestDescriptor.size()) {
                countStageMemo(counters.refusedMismatch);
                return;
            }
            held = result->bindings[patch.binding].guestDescriptor[patch.word];
        }
        if (held != value) {
            countStageMemo(counters.refusedMismatch);
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
        countStageMemo(counters.refusedUnwatched);
        return;
    }
    entry->generation = generation;
    {
        // A store landing between the capture's read and the collect above is stamped no later than
        // the generation: the words must still be the guest's now.
        const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
        if (!recent.empty() && !captureStable(recent)) {
            countStageMemo(counters.refusedUnstable);
            return;
        }
    }
    countStageMemo(counters.inserts);
    std::lock_guard lock(stageMemoMutex);
    auto [slot, inserted] = stageMemo.try_emplace(hash);
    if (inserted) {
        stageMemoOrder.push_back(hash);
        slot->order = std::prev(stageMemoOrder.end());
    } else {
        stageMemoOrder.splice(stageMemoOrder.end(), stageMemoOrder, slot->order);
    }
    // The served result belongs to the entry replaced; the misses in a row go on counting.
    slot->entry = std::move(entry);
    slot->source = handle->source.get();
    slot->pushOffset = pushOffset;
    slot->served = {};
    slot->generation = generation;
    // Eviction moves entries in the map: `slot` is not used past this point.
    while (stageMemo.size() > StageMemoCapacity() && !stageMemoOrder.empty()) {
        stageMemo.erase(stageMemoOrder.front());
        stageMemoOrder.pop_front();
        countStageMemo(counters.evictions);
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
            // A result the stage memo served is patched in place by the memo's later hits once nothing
            // else holds it. Held by the variant until evicted (a parked cache keeps its variants), it
            // made those hits copy the result instead: about half of them on the S2 menu page. The
            // variant keeps a copy of its own.
            variant->compiled = stageCapture.memo != nullptr && VariantResultCopies() ? std::make_shared<const ShaderRecompiler::RecompileResult>(*stageCapture.compiled) : stageCapture.compiled;
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
