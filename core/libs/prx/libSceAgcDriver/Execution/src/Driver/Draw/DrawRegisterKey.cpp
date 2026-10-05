#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace AgcDriver::DriverDetail {

namespace {

// The shader-bank user data words of the pixel, geometry and hull programs (32 each).
constexpr std::array<std::uint32_t, 3> UserDataBases{0x00cu, 0x08cu, 0x10cu};
constexpr const char* UserDataStages[3] = {"ps", "gs", "hs"};

bool userDataRegister(Graphics::RegisterBank bank, std::uint32_t offset) {
    if (bank != Graphics::RegisterBank::Shader) return false;
    for (const auto base : UserDataBases) {
        if (offset >= base && offset < base + 32u) return true;
    }
    return false;
}

struct StageMemory {
    std::vector<std::uint32_t> userData;
    std::vector<std::pair<std::uint64_t, std::vector<std::uint32_t>>> regions;
};

struct RelocationTrace {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::vector<StageMemory>> last;
    std::uint64_t draws = 0, repeats = 0, stages = 0, layoutDiffer = 0, contentSame = 0, contentSameInPlace = 0, regions = 0, words = 0, movedRegions = 0, differingMoved = 0, differingStatic = 0;
    std::array<std::uint64_t, 5> buckets{};
    std::vector<std::string> examples;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

struct DrawKeyTrace {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::array<std::uint32_t, 96>> last;
    std::uint64_t lookups = 0, hits = 0, repeats = 0, fresh = 0, sameWords = 0;
    std::map<std::uint32_t, std::uint64_t> differing;
    std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> example;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

}

std::uint64_t Driver::structuralDrawKey(const QueueState& queue, const ShaderRegistry& registry, std::uint64_t deviceSerial) {
    std::uint64_t key = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    mix(deviceSerial);
    for (const auto& range : Graphics::DrawKeyRegisters) {
        const auto& bank = range.bank == Graphics::RegisterBank::Context ? queue.context : range.bank == Graphics::RegisterBank::Shader ? queue.shader : queue.userConfig;
        const auto end = range.first + range.count;
        for (auto it = bank.lower_bound(range.first); it != bank.end() && it->first < end; ++it) {
            if (userDataRegister(range.bank, it->first)) continue;
            mix(it->first);
            mix(it->second);
        }
    }
    for (const auto base : {0x008u, 0x088u, 0x0c8u, 0x108u, 0x148u}) {
        const auto low = queue.shader.find(base);
        const auto high = queue.shader.find(base + 1);
        if (low == queue.shader.end() || high == queue.shader.end()) {
            mix(0);
            continue;
        }
        const auto address = (static_cast<std::uint64_t>(low->second) << 8u) | (static_cast<std::uint64_t>(high->second & 0xffu) << 40u);
        auto it = registry.upper_bound(address);
        if (it == registry.begin()) {
            mix(1);
            continue;
        }
        --it;
        mix(reinterpret_cast<std::uintptr_t>(it->second.get()));
        mix(address - it->second->codeAddress);
    }
    return key;
}

void Driver::traceDrawRelocation(std::uint64_t structuralKey, const std::vector<DrawProgram>& programs, const std::vector<StageCapture>& captures) {
    std::vector<StageMemory> current(captures.size());
    for (std::size_t stage = 0; stage < captures.size(); ++stage) {
        if (stage < programs.size()) current[stage].userData = programs[stage].userData;
        for (const auto& region : captures[stage].regions) {
            std::vector<std::uint32_t> words(region.bytes.size() / sizeof(std::uint32_t));
            std::memcpy(words.data(), region.bytes.data(), words.size() * sizeof(std::uint32_t));
            current[stage].regions.emplace_back(region.guestAddress, std::move(words));
        }
    }
    static RelocationTrace trace;
    std::lock_guard lock(trace.mutex);
    ++trace.draws;
    auto [it, inserted] = trace.last.try_emplace(structuralKey, current);
    if (!inserted) {
        ++trace.repeats;
        const auto& previous = it->second;
        for (std::size_t stage = 0; stage < std::min(previous.size(), current.size()); ++stage) {
            const auto& a = previous[stage];
            const auto& b = current[stage];
            ++trace.stages;
            bool sameLayout = a.regions.size() == b.regions.size();
            for (std::size_t r = 0; sameLayout && r < a.regions.size(); ++r) sameLayout = a.regions[r].second.size() == b.regions[r].second.size();
            if (!sameLayout) {
                ++trace.layoutDiffer;
                continue;
            }
            std::size_t differing = 0;
            bool moved = false;
            for (std::size_t r = 0; r < a.regions.size(); ++r) {
                const bool regionMoved = a.regions[r].first != b.regions[r].first;
                moved = moved || regionMoved;
                trace.regions += 1;
                trace.words += b.regions[r].second.size();
                if (regionMoved) ++trace.movedRegions;
                for (std::size_t w = 0; w < b.regions[r].second.size(); ++w) {
                    if (a.regions[r].second[w] == b.regions[r].second[w]) continue;
                    ++differing;
                    ++(regionMoved ? trace.differingMoved : trace.differingStatic);
                    if (trace.examples.size() < 24) {
                        char text[200];
                        std::snprintf(text, sizeof(text), "stage %zu region %zu/%zu (%zu words, %s 0x%llx->0x%llx) word %zu: 0x%08x -> 0x%08x", stage, r, a.regions.size(), b.regions[r].second.size(), regionMoved ? "moved" : "fixed", static_cast<unsigned long long>(a.regions[r].first), static_cast<unsigned long long>(b.regions[r].first), w, a.regions[r].second[w], b.regions[r].second[w]);
                        trace.examples.emplace_back(text);
                    }
                }
            }
            if (differing == 0) {
                ++trace.contentSame;
                if (!moved) ++trace.contentSameInPlace;
            } else {
                ++trace.buckets[differing == 1 ? 0 : differing <= 4 ? 1 : differing <= 16 ? 2 : differing <= 64 ? 3 : 4];
            }
        }
        it->second = std::move(current);
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - trace.lastReport < std::chrono::seconds(10)) return;
    trace.lastReport = now;
    std::fprintf(stderr, "[drawreloc] %llu draws, %llu repeating a structural key (%zu keys); %llu stages compared: layout differs %llu, same words %llu (%llu with nothing moved); differing words per stage 1 / 2-4 / 5-16 / 17-64 / more: %llu / %llu / %llu / %llu / %llu; %llu regions (%llu moved), %llu words; differing words in moved regions %llu, in fixed ones %llu\n", static_cast<unsigned long long>(trace.draws), static_cast<unsigned long long>(trace.repeats), trace.last.size(), static_cast<unsigned long long>(trace.stages), static_cast<unsigned long long>(trace.layoutDiffer), static_cast<unsigned long long>(trace.contentSame), static_cast<unsigned long long>(trace.contentSameInPlace), static_cast<unsigned long long>(trace.buckets[0]), static_cast<unsigned long long>(trace.buckets[1]), static_cast<unsigned long long>(trace.buckets[2]), static_cast<unsigned long long>(trace.buckets[3]), static_cast<unsigned long long>(trace.buckets[4]), static_cast<unsigned long long>(trace.regions), static_cast<unsigned long long>(trace.movedRegions), static_cast<unsigned long long>(trace.words), static_cast<unsigned long long>(trace.differingMoved), static_cast<unsigned long long>(trace.differingStatic));
    for (const auto& example : trace.examples) std::fprintf(stderr, "[drawreloc]   %s\n", example.c_str());
    trace.examples.clear();
    trace.draws = trace.repeats = trace.stages = trace.layoutDiffer = trace.contentSame = trace.contentSameInPlace = trace.regions = trace.words = trace.movedRegions = trace.differingMoved = trace.differingStatic = 0;
    trace.buckets.fill(0);
}

void Driver::traceDrawKey(const QueueState& queue, const ShaderRegistry& registry, std::uint64_t deviceSerial, bool hit) {
    // The draw key without the user data words; the words themselves beside it.
    const auto key = structuralDrawKey(queue, registry, deviceSerial);
    std::array<std::uint32_t, 96> words{};
    for (std::size_t stage = 0; stage < UserDataBases.size(); ++stage) {
        for (std::uint32_t i = 0; i < 32; ++i) {
            const auto found = queue.shader.find(UserDataBases[stage] + i);
            words[stage * 32 + i] = found == queue.shader.end() ? 0xdeadbeefu : found->second;
        }
    }
    static DrawKeyTrace trace;
    std::lock_guard lock(trace.mutex);
    ++trace.lookups;
    if (hit) ++trace.hits;
    auto [it, inserted] = trace.last.try_emplace(key, words);
    if (inserted) {
        ++trace.fresh;
    } else if (!hit) {
        ++trace.repeats;
        bool any = false;
        for (std::uint32_t i = 0; i < 96; ++i) {
            if (it->second[i] == words[i]) continue;
            any = true;
            ++trace.differing[i];
            trace.example[i] = {it->second[i], words[i]};
        }
        if (!any) ++trace.sameWords;
        it->second = words;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - trace.lastReport < std::chrono::seconds(10)) return;
    trace.lastReport = now;
    std::fprintf(stderr, "[drawkey] %llu lookups (cumulative): %llu hits; misses whose key without user data was seen before %llu (%llu with the same user data too), first seen %llu; %zu keys without user data\n", static_cast<unsigned long long>(trace.lookups), static_cast<unsigned long long>(trace.hits), static_cast<unsigned long long>(trace.repeats), static_cast<unsigned long long>(trace.sameWords), static_cast<unsigned long long>(trace.fresh), trace.last.size());
    std::vector<std::pair<std::uint64_t, std::uint32_t>> ranked;
    for (const auto& [position, count] : trace.differing) ranked.push_back({count, position});
    std::sort(ranked.begin(), ranked.end(), std::greater<>());
    for (std::size_t i = 0; i < ranked.size() && i < 12; ++i) {
        const auto position = ranked[i].second;
        const auto [before, after] = trace.example[position];
        std::fprintf(stderr, "[drawkey]   %s user word %u differs in %llu repeats (e.g. 0x%08x -> 0x%08x)\n", UserDataStages[position / 32], position % 32, static_cast<unsigned long long>(ranked[i].first), before, after);
    }
}

std::uint64_t Driver::drawRegisterKey(const QueueState& queue, const ShaderRegistry& registry, std::uint64_t deviceSerial) {
    static const bool allUserWords = std::getenv("APS5_DRAW_KEY_ALL_USER_WORDS") != nullptr;
    std::uint64_t key = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    const auto userEnd = [&](std::uint32_t base) {
        if (allUserWords) return base + 32u;
        const auto resources = queue.shader.find(base - 1);
        if (resources == queue.shader.end()) return base;
        const auto count = ((resources->second >> 1u) & 0x1fu) | (((resources->second >> 27u) & 1u) << 5u);
        return base + std::min(count, 32u);
    };
    const std::array<std::pair<std::uint32_t, std::uint32_t>, 3> users{{{0x00cu, userEnd(0x00cu)}, {0x08cu, userEnd(0x08cu)}, {0x10cu, userEnd(0x10cu)}}};
    const auto unread = [&](std::uint32_t offset) {
        return std::any_of(users.begin(), users.end(), [&](const auto& user) { return offset >= user.second && offset < user.first + 32u; });
    };
    mix(deviceSerial);
    for (const auto& range : Graphics::DrawKeyRegisters) {
        const auto& bank = range.bank == Graphics::RegisterBank::Context ? queue.context : range.bank == Graphics::RegisterBank::Shader ? queue.shader : queue.userConfig;
        mix((static_cast<std::uint64_t>(range.bank) << 32u) | range.first);
        const auto end = range.first + range.count;
        const bool shader = range.bank == Graphics::RegisterBank::Shader;
        for (auto it = bank.lower_bound(range.first); it != bank.end() && it->first < end; ++it) {
            if (shader && unread(it->first)) continue;
            mix(it->first);
            mix(it->second);
        }
    }
    for (const auto base : {0x008u, 0x088u, 0x0c8u, 0x108u, 0x148u}) {
        const auto low = queue.shader.find(base);
        const auto high = queue.shader.find(base + 1);
        if (low == queue.shader.end() || high == queue.shader.end()) {
            mix(0);
            continue;
        }
        const auto address = (static_cast<std::uint64_t>(low->second) << 8u) | (static_cast<std::uint64_t>(high->second & 0xffu) << 40u);
        auto it = registry.upper_bound(address);
        if (it == registry.begin()) {
            mix(1);
            continue;
        }
        --it;
        mix(reinterpret_cast<std::uintptr_t>(it->second.get()));
        mix(address - it->second->codeAddress);
    }
    return key;
}

bool Driver::sameVertexInfo(const ShaderRecompiler::ShaderVertexStageInfo& a, const ShaderRecompiler::ShaderVertexStageInfo& b) {
    if (a.resourcesNum != b.resourcesNum || a.fetchAttribReg != b.fetchAttribReg || a.fetchBufferReg != b.fetchBufferReg || a.fetchEmbedded != b.fetchEmbedded) return false;
    for (std::uint32_t i = 0; i < a.resourcesNum && i < a.resources.size(); ++i) {
        if (a.resources[i].fields != b.resources[i].fields) return false;
        const auto& x = a.resourcesDst[i];
        const auto& y = b.resourcesDst[i];
        if (x.registerStart != y.registerStart || x.registersNum != y.registersNum || x.attrId != y.attrId || x.fetchIndex != y.fetchIndex) return false;
    }
    return true;
}

bool Driver::sameDecode(const DrawDecode& a, const DrawDecode& b) {
    const auto& s = a.state;
    const auto& t = b.state;
    const auto sameColor = [](const Graphics::ColorTarget& x, const Graphics::ColorTarget& y) {
        return x.address == y.address && x.extent.width == y.extent.width && x.extent.height == y.extent.height && x.format == y.format && x.bytes == y.bytes && x.componentMapping == y.componentMapping && x.tileMode == y.tileMode && x.elementBytes == y.elementBytes && x.dccAddress == y.dccAddress && x.dccAlphaOnMsb == y.dccAlphaOnMsb && x.slot == y.slot && x.exportIndex == y.exportIndex;
    };
    const auto sameBlend = [](const VkPipelineColorBlendAttachmentState& x, const VkPipelineColorBlendAttachmentState& y) {
        return x.blendEnable == y.blendEnable && x.srcColorBlendFactor == y.srcColorBlendFactor && x.dstColorBlendFactor == y.dstColorBlendFactor && x.colorBlendOp == y.colorBlendOp && x.srcAlphaBlendFactor == y.srcAlphaBlendFactor && x.dstAlphaBlendFactor == y.dstAlphaBlendFactor && x.alphaBlendOp == y.alphaBlendOp && x.colorWriteMask == y.colorWriteMask;
    };
    const auto sameMesh = [](const std::optional<ShaderRecompiler::MeshConfiguration>& x, const std::optional<ShaderRecompiler::MeshConfiguration>& y) {
        if (x.has_value() != y.has_value()) return false;
        if (!x) return true;
        return x->inputPrimitive == y->inputPrimitive && x->primitivesPerGroup == y->primitivesPerGroup && x->verticesPerGroup == y->verticesPerGroup && x->maxVertices == y->maxVertices && x->maxPrimitives == y->maxPrimitives && x->threadsPerGroup == y->threadsPerGroup && x->ldsSizeDwords == y->ldsSizeDwords && x->provokingVertex == y->provokingVertex && x->esgsItemSize == y->esgsItemSize;
    };
    const auto sameTess = [](const std::optional<ShaderRecompiler::TessellationConfiguration>& x, const std::optional<ShaderRecompiler::TessellationConfiguration>& y) {
        if (x.has_value() != y.has_value()) return false;
        if (!x) return true;
        return x->inputControlPoints == y->inputControlPoints && x->outputControlPoints == y->outputControlPoints && x->domain == y->domain && x->partitioning == y->partitioning && x->outputTopology == y->outputTopology;
    };
    if (s.stages.path != t.stages.path || s.stages.registerValue != t.stages.registerValue || s.stages.vertexWaveSize != t.stages.vertexWaveSize || s.stages.fragmentWaveSize != t.stages.fragmentWaveSize || !sameMesh(s.stages.mesh, t.stages.mesh) || !sameTess(s.stages.tessellation, t.stages.tessellation)) return false;
    if (!sameColor(s.color, t.color) || s.colors.size() != t.colors.size() || s.blends.size() != t.blends.size()) return false;
    for (std::size_t i = 0; i < s.colors.size(); ++i) {
        if (!sameColor(s.colors[i], t.colors[i])) return false;
    }
    for (std::size_t i = 0; i < s.blends.size(); ++i) {
        if (!sameBlend(s.blends[i], t.blends[i])) return false;
    }
    if (s.hasColorTarget != t.hasColorTarget || s.rectList != t.rectList || s.renderExtent.width != t.renderExtent.width || s.renderExtent.height != t.renderExtent.height || s.topology != t.topology || s.negativeOneToOne != t.negativeOneToOne || s.depthClamp != t.depthClamp || s.cullMode != t.cullMode || s.frontFace != t.frontFace || !sameBlend(s.blend, t.blend) || s.blendConstants != t.blendConstants) return false;
    if (std::memcmp(&s.viewport, &t.viewport, sizeof(VkViewport)) != 0 || std::memcmp(&s.scissor, &t.scissor, sizeof(VkRect2D)) != 0) return false;
    const auto& p = a.pixel;
    const auto& q = b.pixel;
    if (p.interpolatorCount != q.interpolatorCount || p.interpolatorSettings != q.interpolatorSettings || p.wave32 != q.wave32 || p.inputAddr != q.inputAddr || p.hasPerspectiveCenterVgpr != q.hasPerspectiveCenterVgpr || p.perspectiveCentroid != q.perspectiveCentroid || p.posX != q.posX || p.posY != q.posY || p.posZ != q.posZ || p.posW != q.posW || p.frontFace != q.frontFace || p.ancillary != q.ancillary || p.sampleShading != q.sampleShading || p.noPerspective != q.noPerspective || p.linearCentroid != q.linearCentroid || p.pixelKillEnable != q.pixelKillEnable || p.depthExportEnable != q.depthExportEnable || p.sampleMaskExportEnable != q.sampleMaskExportEnable || p.earlyZ != q.earlyZ || p.executeOnNoop != q.executeOnNoop || p.conservativeZExport != q.conservativeZExport || p.orderedPixelShader != q.orderedPixelShader || p.targetOutputMode != q.targetOutputMode || p.targetExportMapping != q.targetExportMapping) return false;
    if (a.roles != b.roles || a.programs.size() != b.programs.size()) return false;
    for (std::size_t i = 0; i < a.programs.size(); ++i) {
        const auto& x = a.programs[i];
        const auto& y = b.programs[i];
        if (x.binary.stage != y.binary.stage || x.binary.codeAddress != y.binary.codeAddress || x.userDataBase != y.userDataBase || x.firstUserSgpr != y.firstUserSgpr || x.userData != y.userData || x.snapshot != y.snapshot || x.codeOffset != y.codeOffset) return false;
    }
    return true;
}

}
