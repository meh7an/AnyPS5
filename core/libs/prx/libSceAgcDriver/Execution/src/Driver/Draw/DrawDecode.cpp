#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <list>
#include <unordered_map>

namespace AgcDriver::DriverDetail {

namespace {

std::uint32_t ReadGraphicsRegister(const Registers& registers, std::uint32_t offset) {
    const auto found = registers.find(offset);
    return found == registers.end() ? 0u : found->second;
}

void readMergedPointer(const QueueState& queue, DrawProgram& program) {
    const auto pointerBase = program.mergedPointer;
    if (!program.mergedPointerRequired && !queue.shader.contains(pointerBase) && !queue.shader.contains(pointerBase + 1)) return;
    const auto low = ReadGraphicsRegister(queue.shader, pointerBase);
    const auto high = ReadGraphicsRegister(queue.shader, pointerBase + 1);
    const auto address = static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32u);
    require(address != 0 || !program.mergedPointerRequired, "merged shader user-data address is null");
    if (address == 0) return;
    GuestMemory::CheckRange(reinterpret_cast<const void*>(address), 8, 4);
    program.userData[0] = low;
    program.userData[1] = high;
}

void rereadUserData(const QueueState& queue, DrawProgram& program) {
    const std::size_t front = program.mergedPointer != 0 ? 8 : 0;
    for (std::size_t i = front; i < program.userData.size(); ++i) program.userData[i] = readUserData(queue.shader, program.userDataBase + static_cast<std::uint32_t>(i - front));
    if (front == 0) return;
    program.userData[0] = 0;
    program.userData[1] = 0;
    readMergedPointer(queue, program);
}

struct DecodeCacheEntry {
    std::shared_ptr<const ShaderRegistry> registry;
    std::uint64_t forgetSerial = 1;
    std::vector<std::shared_ptr<DrawDecode>> copies;
    std::size_t replace = 0;
    std::list<std::uint64_t>::iterator order;
};

struct DecodeCache {
    std::unordered_map<std::uint64_t, DecodeCacheEntry> entries;
    std::list<std::uint64_t> order;
    std::uint64_t hits = 0, misses = 0, copied = 0, verified = 0, mismatches = 0;
};

struct DecodeCacheTag;

constexpr std::size_t DecodeCacheEntries = 4096;
constexpr std::size_t DecodeCacheCopies = 4;

bool DecodeCacheEnabled() {
    static const bool enabled = std::getenv("APS5_NO_DECODE_CACHE") == nullptr;
    return enabled;
}

bool VerifyDecodeCache() {
    static const bool verify = std::getenv("APS5_VERIFY_DECODE_CACHE") != nullptr;
    return verify;
}

}

void DecodeGraphicsPrograms(DrawDecode& decoded, const QueueState& queue, const ShaderRegistry& registry, bool staticAbi, bool includeFragment) {
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;
    const auto programAddress = [&](std::uint32_t base) {
        Graphics::NoteRegisterRead(Graphics::RegisterBank::Shader, base);
        Graphics::NoteRegisterRead(Graphics::RegisterBank::Shader, base + 1);
        const auto high = ReadGraphicsRegister(queue.shader, base + 1);
        require((high & ~0xffu) == 0, "reserved graphics program address bits are set");
        return (static_cast<std::uint64_t>(ReadGraphicsRegister(queue.shader, base)) << 8u) | (static_cast<std::uint64_t>(high) << 40u);
    };
    const bool pixelSkipped = Graphics::PixelProgramSkipped(queue);
    const auto prepare = [&](std::uint64_t address, std::uint8_t type, Stage stage, std::uint32_t rsrc2, std::uint32_t userDataBase) {
        const bool nullPixel = stage == Stage::Fragment && (address == 0 || pixelSkipped);
        if (nullPixel) address = NullPixelProgramAddress();
        const auto program = ProgramSnapshot(registry, address, {type});
        const auto& snapshot = *program;
        require((address - snapshot.codeAddress) % sizeof(std::uint32_t) == 0, "graphics entry point is not dword aligned");
        require(snapshot.header.empty() || snapshot.type == type, "graphics program refers to an incompatible shader binary type");
        if (!nullPixel && snapshot.header.empty()) NoteRawGraphicsDraw();
        if (!nullPixel) Graphics::NoteRegisterRead(Graphics::RegisterBank::Shader, rsrc2);
        const auto resources = nullPixel ? 0u : ReadGraphicsRegister(queue.shader, rsrc2);
        const auto userCount = ((resources >> 1u) & 0x1fu) | (((resources >> 27u) & 1u) << 5u);
        require(userCount <= 32, "graphics user SGPR count exceeds the register bank");
        const auto codeOffset = static_cast<std::size_t>((address - snapshot.codeAddress) / sizeof(std::uint32_t));
        DrawProgram result{
            {stage, address, std::span(snapshot.code).subspan(codeOffset), snapshot.headerAddress, snapshot.header},
            userDataBase,
            8,
            {},
            {{{snapshot.codeAddress, std::as_bytes(std::span(snapshot.code))}, {snapshot.headerAddress, snapshot.header}}},
            program,
            codeOffset
        };
        result.userData.reserve(userCount + 8u);
        for (std::uint32_t i = 0; i < userCount; ++i) {
            Graphics::NoteRegisterRead(Graphics::RegisterBank::Shader, userDataBase + i);
            result.userData.push_back((staticAbi ? ReadGraphicsRegister(queue.shader, userDataBase + i) : readUserData(queue.shader, userDataBase + i)));
        }
        return result;
    };
    {
        auto* product = &decoded;
        auto& programs = product->programs;
        auto& roles = product->roles;
        programs.reserve(5);
        roles.reserve(5);
        const auto append = [&](std::uint32_t base, std::uint8_t type, Stage stage, std::uint32_t resources, std::uint32_t users, Role role) {
            programs.push_back(prepare(programAddress(base), type, stage, resources, users));
            roles.push_back(role);
        };
        const auto initializeMerged = [&](DrawProgram& program, std::uint32_t pointerBase, bool pointerRequired) {
            program.firstUserSgpr = 0;
            program.userData.insert(program.userData.begin(), 8, 0);
            program.mergedPointer = pointerBase;
            program.mergedPointerRequired = pointerRequired;
            Graphics::NoteRegisterRead(Graphics::RegisterBank::Shader, pointerBase);
            Graphics::NoteRegisterRead(Graphics::RegisterBank::Shader, pointerBase + 1);
            if (staticAbi) return;
            readMergedPointer(queue, program);
        };
        const auto& graphics = product->state;
        if (graphics.stages.path == Graphics::ShaderPath::Tessellation) {
            if (!staticAbi || programAddress(0x148) != 0) append(0x148, 5, Stage::Local, 0x10b, 0x10c, Role::Local);
            append(0x108, 7, Stage::TessellationControl, 0x10b, 0x10c, Role::Hull);
            initializeMerged(programs.back(), 0x102, true);
            append(0x0c8, 2, Stage::TessellationEvaluation, 0x08b, 0x08c, Role::Domain);
        } else if (graphics.stages.path == Graphics::ShaderPath::Geometry) {
            const auto frontAddress = programAddress(0xc8);
            const auto front = ProgramSnapshot(registry, frontAddress, {2, 4});
            const auto type = front->header.empty() ? std::uint8_t{2} : front->type;
            require(type == 2 || type == 4, "invalid geometry front binary type");
            append(0xc8, type, Stage::Mesh, 0x8b, 0x8c, Role::Main);
            initializeMerged(programs.back(), 0x82, type == 4);
            if (type == 4) append(0x88, 6, Stage::Mesh, 0x8b, 0x8c, Role::GeometryBack);
        } else {
            append(0xc8, 2, Stage::Vertex, 0x8b, 0x8c, Role::Main);
        }
        if (includeFragment) {
            if (pixelSkipped) {
                const auto rejection = Graphics::NullPixelProgramRejection(queue);
                require(rejection.empty(), rejection.c_str());
            }
            append(0x008, 1, Stage::Fragment, 0x00b, 0x00c, Role::Fragment);
            programs.back().firstUserSgpr = 0;
        }
    }
}

std::shared_ptr<DrawDecode> Driver::decodeDraw(const QueueState& queue, const Submission& submission) {
    auto product = std::make_shared<DrawDecode>();
    product->state = Graphics::DecodeState(queue, VertexWorkgroupRequired(queue, *submission.shaders));
    DecodeGraphicsPrograms(*product, queue, *submission.shaders, false, true);
    product->pixel = Graphics::DecodePixelStageInfo(queue.context, Graphics::ExportMappings(product->state), Graphics::PixelProgramSkipped(queue));
    product->pixel.targetExportPacking = Graphics::ExportPackings(product->state);
    product->pixel.dualSourceBlend = product->state.dualSourceBlend;
    product->pixel.alphaToCoverage = product->state.alphaToCoverage;
    if (product->pixel.ancillary) product->pixel.rasterPrimitive = Graphics::RasterPrimitiveOf(product->state);
    if (product->pixel.ancillary) product->pixel.layerExported = product->state.layerExports != 0;
    return product;
}

void Driver::resolveDrawDecode(const QueueState& queue, const Submission& submission, std::shared_ptr<const DrawDecode>& decode, bool registerKey, std::uint64_t drawKey, bool profile) {
    if (decode == nullptr && !verifyDrawRecipe() && DecodeCacheEnabled()) {
        auto& cache = ThreadScratch<DecodeCache, DecodeCacheTag>();
        const auto decodeKey = drawDecodeKey(queue, *submission.shaders);
        const auto report = [&] {
            if (!profile && !VerifyDecodeCache()) return;
            if ((cache.hits + cache.misses) % 500000 != 0) return;
            std::fprintf(stderr, "[decode-cache] %llu hits (%llu copied first), %llu misses, %zu entries; verified %llu hits: %llu decode mismatches\n", static_cast<unsigned long long>(cache.hits), static_cast<unsigned long long>(cache.copied), static_cast<unsigned long long>(cache.misses), cache.entries.size(), static_cast<unsigned long long>(cache.verified), static_cast<unsigned long long>(cache.mismatches));
        };
        const auto serial = GuestMemory::ForgetSerial();
        const bool settled = (serial & 1u) == 0;
        const auto found = cache.entries.find(decodeKey);
        if (found != cache.entries.end() && settled && found->second.forgetSerial == serial && found->second.registry == submission.shaders && DecodeProgramsCurrent(*found->second.copies.front(), *submission.shaders)) {
            auto& entry = found->second;
            cache.order.splice(cache.order.end(), cache.order, entry.order);
            std::shared_ptr<DrawDecode> served;
            for (const auto& copy : entry.copies) {
                if (copy.use_count() != 1) continue;
                std::atomic_thread_fence(std::memory_order_acquire);
                served = copy;
                break;
            }
            if (served == nullptr) {
                served = std::make_shared<DrawDecode>(*entry.copies.front());
                ++cache.copied;
                if (entry.copies.size() < DecodeCacheCopies) {
                    entry.copies.push_back(served);
                } else {
                    entry.copies[entry.replace] = served;
                    entry.replace = (entry.replace + 1) % entry.copies.size();
                }
            }
            for (auto& program : served->programs) rereadUserData(queue, program);
            ++cache.hits;
            if (VerifyDecodeCache()) {
                ++cache.verified;
                const bool same = sameDecode(*decodeDraw(queue, submission), *served);
                if (!same && ++cache.mismatches <= 20) std::fprintf(stderr, "[decode-cache] verify: a hit at key 0x%llx decodes differently\n", static_cast<unsigned long long>(decodeKey));
            }
            report();
            decode = std::move(served);
            return;
        }
        auto fresh = decodeDraw(queue, submission);
        ++cache.misses;
        if (found != cache.entries.end()) {
            found->second.registry = submission.shaders;
            found->second.forgetSerial = settled ? serial : 1;
            found->second.copies.assign(1, fresh);
            found->second.replace = 0;
            cache.order.splice(cache.order.end(), cache.order, found->second.order);
        } else {
            while (cache.entries.size() >= DecodeCacheEntries && !cache.order.empty()) {
                cache.entries.erase(cache.order.front());
                cache.order.pop_front();
            }
            auto& entry = cache.entries[decodeKey];
            entry.registry = submission.shaders;
            entry.forgetSerial = settled ? serial : 1;
            entry.copies.push_back(fresh);
            cache.order.push_back(decodeKey);
            entry.order = std::prev(cache.order.end());
        }
        report();
        decode = std::move(fresh);
        return;
    }
    if (decode == nullptr || verifyDrawRecipe()) {
        std::vector<Graphics::RegisterRead> readLog;
        const Graphics::RegisterReadLogScope logScope(verifyDrawRecipe() && registerKey ? &readLog : nullptr);
        auto fresh = decodeDraw(queue, submission);
        if (verifyDrawRecipe() && registerKey) {
            std::uint64_t facadeMismatches = 0;
            for (const auto read : readLog) {
                if (Graphics::DrawKeyCovers(read)) continue;
                ++facadeMismatches;
                static std::atomic<std::uint64_t> reports{0};
                if (reports.fetch_add(1) < 20) std::fprintf(stderr, "[draw-cache] verify: the decoders read %s register 0x%x, which DrawKeyRegisters lacks\n", Graphics::RegisterBankName(read.bank), read.offset);
            }
            std::uint64_t decodeMismatches = 0;
            if (decode != nullptr && !sameDecode(*decode, *fresh)) {
                decodeMismatches = 1;
                static std::atomic<std::uint64_t> reports{0};
                if (reports.fetch_add(1) < 20) std::fprintf(stderr, "[draw-cache] verify: the entry's decode differs from a fresh decode (key 0x%llx, target 0x%llx)\n", static_cast<unsigned long long>(drawKey), static_cast<unsigned long long>(fresh->state.color.address));
            }
            std::lock_guard cacheLock(drawCacheMutex);
            drawEntryCounters.facadeMismatches += facadeMismatches;
            if (decode != nullptr) ++drawEntryCounters.verifyDecodes;
            drawEntryCounters.verifyDecodeMismatches += decodeMismatches;
        }
        decode = std::move(fresh);
    } else if (profile) {
        std::lock_guard cacheLock(drawCacheMutex);
        ++drawEntryCounters.decodeSkipped;
    }
}

}
