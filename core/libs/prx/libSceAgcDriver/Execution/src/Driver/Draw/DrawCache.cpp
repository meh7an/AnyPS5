#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace AgcDriver::DriverDetail {

bool DrawRecipeRecord::Matches(const std::vector<std::shared_ptr<DispatchVariant>>& variants) const {
    if (stages.size() != variants.size()) return false;
    for (std::size_t i = 0; i < stages.size(); ++i) {
        if (stages[i].owner_before(variants[i]) || variants[i].owner_before(stages[i])) return false;
    }
    return true;
}

bool DrawRecipeRecord::Expired() const {
    return std::any_of(stages.begin(), stages.end(), [](const std::weak_ptr<const DispatchVariant>& stage) { return stage.expired(); });
}

bool Driver::drawEntries() {
    static const bool entries = std::getenv("APS5_NO_DRAW_SRT_ENTRIES") == nullptr && !stampValidate();
    return entries;
}

namespace {

bool drawCacheAlways() {
    static const bool always = std::getenv("APS5_DRAW_CACHE_ALWAYS") != nullptr;
    return always;
}

bool drawCacheParkBackoff() {
    static const bool backoff = [] {
        const char* text = std::getenv("APS5_DRAW_CACHE_PARK_BACKOFF");
        return text == nullptr || std::strcmp(text, "0") != 0;
    }();
    return backoff;
}

}

bool Driver::drawCacheActive() {
    if (drawCacheAlways()) return true;
    // Only a queue's worker draws: a plain load and store count the draw without a locked add
    // (a count another worker overwrote would only move the next probe by a draw).
    const auto draws = drawCacheDraws.load(std::memory_order_relaxed);
    drawCacheDraws.store(draws + 1, std::memory_order_relaxed);
    return draws >= drawCacheParkedUntil.load(std::memory_order_relaxed) || flipsCounted.load(std::memory_order_relaxed) >= drawCacheParkedUntilFlip.load(std::memory_order_relaxed);
}

void Driver::noteDrawCacheLookup(bool hit) {
    if (drawCacheAlways()) return;
    if (hit) drawCacheWindowHits.fetch_add(1, std::memory_order_relaxed);
    if (drawCacheWindowLookups.fetch_add(1, std::memory_order_relaxed) + 1 < DrawCacheProbeLookups) return;
    // The lookup that fills the window closes it.
    const auto hits = drawCacheWindowHits.exchange(0, std::memory_order_relaxed);
    drawCacheWindowLookups.store(0, std::memory_order_relaxed);
    if (static_cast<std::uint64_t>(hits) * 32u >= DrawCacheProbeLookups) {
        drawCacheFailedWindows.store(0, std::memory_order_relaxed);
        return;
    }
    const bool backoff = drawCacheParkBackoff();
    const auto doublings = backoff ? std::min(drawCacheFailedWindows.fetch_add(1, std::memory_order_relaxed), DrawCacheParkDoublings) : 0u;
    const auto parkDraws = DrawCacheParkDraws << doublings;
    // The flip limit first: until the draw limit moves, the cache still reads as active.
    drawCacheParkedUntilFlip.store(backoff ? flipsCounted.load(std::memory_order_relaxed) + DrawCacheParkMaxFlips : std::numeric_limits<std::uint64_t>::max(), std::memory_order_relaxed);
    drawCacheParkedUntil.store(drawCacheDraws.load(std::memory_order_relaxed) + parkDraws, std::memory_order_relaxed);
    const auto parks = drawCacheParks.fetch_add(1, std::memory_order_relaxed) + 1;
    if (parks > 8 && parks % 50 != 0) return;
    if (backoff) std::fprintf(stderr, "[draw-cache] %u of %u lookups hit: the cache is parked for %llu draws or %llu flips (%llu parks so far)\n", hits, DrawCacheProbeLookups, static_cast<unsigned long long>(parkDraws), static_cast<unsigned long long>(DrawCacheParkMaxFlips), static_cast<unsigned long long>(parks));
    else std::fprintf(stderr, "[draw-cache] %u of %u lookups hit: the cache is parked for %llu draws (%llu parks so far)\n", hits, DrawCacheProbeLookups, static_cast<unsigned long long>(parkDraws), static_cast<unsigned long long>(parks));
}

bool Driver::verifyDrawEntries() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_ENTRIES") != nullptr;
    return verify;
}

bool Driver::registerKeyEnabled() {
    static const bool registerKey = std::getenv("APS5_NO_DRAW_KEY") == nullptr;
    return registerKey;
}

bool Driver::verifyDrawRecipe() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_RECIPE") != nullptr;
    return verify;
}

std::size_t Driver::drawCacheEntries() {
    static const std::size_t entries = [] {
        const char* text = std::getenv("APS5_DRAW_CACHE_ENTRIES");
        const auto parsed = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        return parsed != 0 ? static_cast<std::size_t>(parsed) : std::size_t{4096};
    }();
    return entries;
}

void Driver::accountDrawVariant(const DispatchVariant& variant, bool added) {
    if (added) {
        ++drawCacheVariants;
        drawCacheVariantBytes += variantBytes(variant);
    } else {
        --drawCacheVariants;
        drawCacheVariantBytes -= variantBytes(variant);
    }
}

void Driver::insertDrawEntry(std::uint64_t key, std::vector<std::shared_ptr<DispatchVariant>>& fresh, std::shared_ptr<const DrawDecode> decode) {
    std::lock_guard cacheLock(drawCacheMutex);
    auto& counters = drawEntryCounters;
    ++counters.inserts;
    const auto found = drawCache.find(key);
    auto replacement = std::make_shared<DrawEntry>();
    replacement->stages.resize(fresh.size());
    replacement->decode = std::move(decode);
    if (found != drawCache.end()) {
        if (found->second->stages.size() == fresh.size()) replacement->stages = found->second->stages;
        if (found->second->decode != nullptr) replacement->decode = found->second->decode;
        replacement->recipes.store(found->second->recipes.load());
    }
    for (std::size_t i = 0; i < fresh.size(); ++i) {
        if (fresh[i] == nullptr) continue;
        auto& variants = replacement->stages[i];
        const auto present = std::find_if(variants.begin(), variants.end(), [&](const std::shared_ptr<DispatchVariant>& kept) { return kept->pushOffset == fresh[i]->pushOffset && kept->runs == fresh[i]->runs && kept->words == fresh[i]->words; });
        if (present != variants.end()) {
            ++counters.present;
            fresh[i] = *present;
            continue;
        }
        accountDrawVariant(*fresh[i], true);
        variants.insert(variants.begin(), fresh[i]);
        ++counters.variantsInserted;
        while (variants.size() > dispatchVariants()) {
            accountDrawVariant(*variants.back(), false);
            variants.pop_back();
            ++counters.variantsEvicted;
        }
    }
    replacement->touched = drawCacheHits;
    if (found == drawCache.end()) {
        drawOrder.push_front(key);
        replacement->order = drawOrder.begin();
        drawCache.emplace(key, std::move(replacement));
    } else {
        replacement->order = found->second->order;
        drawOrder.splice(drawOrder.begin(), drawOrder, replacement->order);
        found->second = std::move(replacement);
    }
    while (drawCache.size() > drawCacheEntries()) {
        const auto last = drawCache.find(drawOrder.back());
        for (const auto& variants : last->second->stages) {
            for (const auto& variant : variants) accountDrawVariant(*variant, false);
        }
        drawOrder.erase(last->second->order);
        drawCache.erase(last);
        ++drawCacheEvictions;
    }
}

std::shared_ptr<const DrawRecipe> Driver::findDrawRecipe(std::uint64_t key, const std::vector<std::shared_ptr<DispatchVariant>>& stages) {
    std::shared_ptr<const std::vector<DrawRecipeRecord>> records;
    {
        std::lock_guard cacheLock(drawCacheMutex);
        const auto found = drawCache.find(key);
        if (found == drawCache.end()) return nullptr;
        records = found->second->recipes.load();
    }
    if (records == nullptr) return nullptr;
    for (const auto& record : *records) {
        if (record.Matches(stages)) return record.recipe;
    }
    return nullptr;
}

void Driver::attachDrawRecipe(std::uint64_t key, const std::vector<std::shared_ptr<DispatchVariant>>& stages, std::shared_ptr<const DrawRecipe> recipe) {
    std::shared_ptr<DrawEntry> entry;
    {
        std::lock_guard cacheLock(drawCacheMutex);
        const auto found = drawCache.find(key);
        if (found == drawCache.end()) return;
        entry = found->second;
    }
    const auto old = entry->recipes.load();
    auto records = std::make_shared<std::vector<DrawRecipeRecord>>();
    records->push_back({std::vector<std::weak_ptr<const DispatchVariant>>(stages.begin(), stages.end()), std::move(recipe)});
    if (old != nullptr) {
        for (const auto& record : *old) {
            if (record.Matches(stages) || record.Expired() || records->size() >= dispatchVariants()) continue;
            records->push_back(record);
        }
    }
    entry->recipes.store(std::move(records));
    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, VulkanDevice::RecipeKind::Draw);
}

}
