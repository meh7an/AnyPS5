#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <list>
#include <unordered_map>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {

// A merged stage's user pointer words into the front of its userData (left zero when unset).
void readMergedPointer(const QueueState& queue, DrawProgram& program) {
    const auto pointerBase = program.mergedPointer;
    if (!program.mergedPointerRequired && !queue.shader.contains(pointerBase) && !queue.shader.contains(pointerBase + 1)) return;
    const auto low = readRegister(queue.shader, pointerBase);
    const auto high = readRegister(queue.shader, pointerBase + 1);
    const auto address = static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32u);
    require(address != 0 || !program.mergedPointerRequired, "merged shader user-data address is null");
    if (address == 0) return;
    GuestMemory::CheckRange(reinterpret_cast<const void*>(address), 8, 4);
    program.userData[0] = low;
    program.userData[1] = high;
}

// A decoded program's user words read again, as decodeDraw read them: everything else in the
// decode depends on registers QueueState::decodeGeneration covers.
void rereadUserData(const QueueState& queue, DrawProgram& program) {
    const std::size_t front = program.mergedPointer != 0 ? 8 : 0;
    // One pass over the bank; a word never written takes the per-word reads, which name it.
    if (front > program.userData.size() || !queue.shader.ReadRange(program.userDataBase, std::span(program.userData).subspan(front))) {
        for (std::size_t i = front; i < program.userData.size(); ++i) program.userData[i] = readRegister(queue.shader, program.userDataBase + static_cast<std::uint32_t>(i - front));
    }
    if (front == 0) return;
    program.userData[0] = 0;
    program.userData[1] = 0;
    readMergedPointer(queue, program);
}

// The thread's last decode, reused for a draw with the same decode generation, shader registry
// and guest mappings (an even ForgetSerial: the decode's CheckRange answers hold), its user words
// read again. Only while nothing else holds it: a draw cache entry may keep a decode.
// APS5_NO_DECODE_MEMO=1 decodes every draw; APS5_VERIFY_DECODE_MEMO=1 also decodes every hit and
// compares the decodes and the registers they came from.
struct DecodeMemo {
    std::uint64_t generation = 0;
    std::shared_ptr<const ShaderRegistry> registry;
    std::uint64_t forgetSerial = 1;
    std::shared_ptr<DrawDecode> decode;
    Registers context, shader, userConfig;
    std::uint64_t hits = 0, misses = 0, verified = 0, registerMismatches = 0, decodeMismatches = 0;
};

struct DecodeMemoTag;

bool DecodeMemoEnabled() {
    static const bool enabled = std::getenv("APS5_NO_DECODE_MEMO") == nullptr;
    return enabled;
}

bool VerifyDecodeMemo() {
    static const bool verify = std::getenv("APS5_VERIFY_DECODE_MEMO") != nullptr;
    return verify;
}

// The decode cache: the thread's decodes by the digest of the registers a decode depends on (every
// context and user-config register, every shader register but the ones a draw reads again: the
// decode generation's set), reused for a draw with the same shader registry and guest mappings,
// its user words read again. The decode memo only serves a draw whose registers did not change
// since the last one, and one changes before nearly every draw (0.3% hits on the S3K menu); a
// digest finds the same registers again however they changed in between. A decode the draws in
// flight hold is copied before its user words change, and the entry keeps the copy for later
// draws. APS5_NO_DECODE_CACHE=1 takes the decode memo instead; APS5_VERIFY_DECODE_CACHE=1 also
// decodes every hit and compares.
struct DecodeCacheEntry {
    std::shared_ptr<const ShaderRegistry> registry;
    std::uint64_t forgetSerial = 1;
    std::vector<std::shared_ptr<DrawDecode>> copies;
    // The copy a full list's next copy replaces, in turn: the list keeps the newest copies, which
    // the ring's draws give back soon, over ones a draw cache entry keeps until it goes.
    std::size_t replace = 0;
    std::list<std::uint64_t>::iterator order;
};

struct DecodeCache {
    std::unordered_map<std::uint64_t, DecodeCacheEntry> entries;
    std::list<std::uint64_t> order;
    std::uint64_t hits = 0, misses = 0, copied = 0, verified = 0, mismatches = 0;
};

struct DecodeCacheTag;

// The draw thread's ring holds up to 8 draws (DrawThread::Slots): copies for those and the one
// being prepared serve a run of draws with the same registers.
constexpr std::size_t DecodeCacheEntries = 4096;
constexpr std::size_t DecodeCacheCopies = 10;

bool DecodeCacheEnabled() {
    static const bool enabled = std::getenv("APS5_NO_DECODE_CACHE") == nullptr;
    return enabled;
}

bool VerifyDecodeCache() {
    static const bool verify = std::getenv("APS5_VERIFY_DECODE_CACHE") != nullptr;
    return verify;
}

// The shader registers a draw reads again: the user words of the pixel, vertex/geometry and hull
// stages and the merged stages' user pointers (PerDrawShaderRegister, QueueState.hpp).
constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 5> PerDrawShaderRegisters{{{0x0cu, 32u}, {0x8cu, 32u}, {0x10cu, 32u}, {0x82u, 2u}, {0x102u, 2u}}};

std::uint64_t decodeDigest(const QueueState& queue) {
    // A queue's shader bank keeps the per-draw registers out of its digest already
    // (Registers::ExcludePerDrawWords); a bank that kept them has them taken out here.
    auto shader = queue.shader.Digest();
    if (!queue.shader.PerDrawWordsExcluded()) {
        for (const auto& [first, count] : PerDrawShaderRegisters) {
            for (auto offset = first; offset < first + count; ++offset) {
                if (queue.shader.contains(offset)) shader ^= Registers::EntryDigest(offset, queue.shader.at(offset));
            }
        }
    }
    // Odd multipliers keep an entry of one bank from cancelling the same entry of another.
    return queue.context.Digest() ^ (shader * 0x9e3779b97f4a7c15ull) ^ (queue.userConfig.Digest() * 0xc2b2ae3d27d4eb4full);
}

// The banks equal outside the shader registers a draw reads again (decodeGeneration's exclusions).
bool sameDecodeRegisters(const Registers& a, const Registers& b, bool shader) {
    const auto skip = [&](std::uint32_t offset) { return shader && PerDrawShaderRegister(offset); };
    auto x = a.begin();
    auto y = b.begin();
    while (true) {
        while (x != a.end() && skip((*x).first)) ++x;
        while (y != b.end() && skip((*y).first)) ++y;
        if (x == a.end() || y == b.end()) return x == a.end() && y == b.end();
        if ((*x).first != (*y).first || (*x).second != (*y).second) return false;
        ++x;
        ++y;
    }
}

}

std::shared_ptr<DrawDecode> Driver::decodeDraw(const QueueState& queue, const Submission& submission) {
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;
    const auto programAddress = [&](std::uint32_t base) {
        Graphics::NoteRegisterRead(Graphics::RegisterBank::Shader, base);
        Graphics::NoteRegisterRead(Graphics::RegisterBank::Shader, base + 1);
        const auto high = readRegister(queue.shader, base + 1);
        require((high & ~0xffu) == 0, "reserved graphics program address bits are set");
        return (static_cast<std::uint64_t>(readRegister(queue.shader, base)) << 8u) | (static_cast<std::uint64_t>(high) << 40u);
    };
    const bool pixelSkipped = Graphics::PixelProgramSkipped(queue);
    const auto prepare = [&](std::uint64_t address, std::uint8_t type, Stage stage, std::uint32_t rsrc2, std::uint32_t userDataBase) {
        const bool nullPixel = stage == Stage::Fragment && (address == 0 || pixelSkipped);
        if (nullPixel) address = NullPixelProgramAddress();
        const auto* registered = RegisteredShaderAt(submission.shaders, address);
        require(registered != nullptr, "graphics program does not belong to a registered shader");
        const auto& snapshot = **registered;
        require(address - snapshot.codeAddress < snapshot.code.size() * sizeof(std::uint32_t), "graphics program is outside registered shader code");
        require(snapshot.type == type, "graphics program refers to an incompatible shader binary type");
        Graphics::NoteRegisterRead(Graphics::RegisterBank::Shader, rsrc2);
        const auto resources = nullPixel && !queue.shader.contains(rsrc2) ? 0u : readRegister(queue.shader, rsrc2);
        const auto userCount = ((resources >> 1u) & 0x1fu) | (((resources >> 27u) & 1u) << 5u);
        require(userCount <= 32, "graphics user SGPR count exceeds the register bank");
        const auto codeOffset = static_cast<std::size_t>((address - snapshot.codeAddress) / sizeof(std::uint32_t));
        DrawProgram result{
            {stage, address, std::span(snapshot.code).subspan(codeOffset), snapshot.headerAddress, snapshot.header},
            userDataBase,
            8,
            {},
            {{{snapshot.codeAddress, std::as_bytes(std::span(snapshot.code))}, {snapshot.headerAddress, snapshot.header}}},
            *registered,
            codeOffset
        };
        // One allocation: the words, and the 8 a merged stage puts in front (initializeMerged).
        result.userData.reserve(userCount + 8u);
        for (std::uint32_t i = 0; i < userCount; ++i) {
            Graphics::NoteRegisterRead(Graphics::RegisterBank::Shader, userDataBase + i);
            result.userData.push_back(readUserData(queue.shader, userDataBase + i));
        }
        return result;
    };
    {
        auto product = std::make_shared<DrawDecode>();
        product->state = Graphics::DecodeState(queue);
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
            readMergedPointer(queue, program);
        };
        const auto& graphics = product->state;
        if (graphics.stages.path == Graphics::ShaderPath::Tessellation) {
            append(0x148, 5, Stage::Local, 0x10b, 0x10c, Role::Local);
            append(0x108, 7, Stage::TessellationControl, 0x10b, 0x10c, Role::Hull);
            initializeMerged(programs.back(), 0x102, true);
            append(0x0c8, 2, Stage::TessellationEvaluation, 0x08b, 0x08c, Role::Domain);
        } else if (graphics.stages.path == Graphics::ShaderPath::Geometry) {
            const auto frontAddress = programAddress(0xc8);
            const auto* front = RegisteredShaderAt(submission.shaders, frontAddress);
            require(front != nullptr, "geometry front program is not registered");
            const auto type = (*front)->type;
            require(type == 2 || type == 4, "invalid geometry front binary type");
            append(0xc8, type, Stage::Mesh, 0x8b, 0x8c, Role::Main);
            initializeMerged(programs.back(), 0x82, type == 4);
            if (type == 4) append(0x88, 6, Stage::Mesh, 0x8b, 0x8c, Role::GeometryBack);
        } else {
            append(0xc8, 2, Stage::Vertex, 0x8b, 0x8c, Role::Main);
        }
        const bool nullPixel = pixelSkipped;
        if (nullPixel) {
            const auto rejection = Graphics::NullPixelProgramRejection(queue);
            require(rejection.empty(), rejection.c_str());
        }
        if (nullPixel) {
            programs.push_back(prepare(0, 1, Stage::Fragment, 0x00b, 0x00c));
            roles.push_back(Role::Fragment);
        } else {
            append(0x008, 1, Stage::Fragment, 0x00b, 0x00c, Role::Fragment);
        }
        programs.back().firstUserSgpr = 0;
        product->pixel = Graphics::DecodePixelStageInfo(queue.context, Graphics::ExportMappings(graphics), nullPixel);
        return product;
    }
}

void Driver::resolveDrawDecode(const QueueState& queue, const Submission& submission, std::shared_ptr<const DrawDecode>& decode, bool registerKey, std::uint64_t drawKey, bool profile) {
    if (decode == nullptr && !verifyDrawRecipe() && DecodeCacheEnabled()) {
        auto& cache = ThreadScratch<DecodeCache, DecodeCacheTag>();
        const auto report = [&] {
            if (!profile && !VerifyDecodeCache()) return;
            if ((cache.hits + cache.misses) % 500000 != 0) return;
            std::fprintf(stderr, "[decode-cache] %llu hits (%llu copied first), %llu misses, %zu entries; verified %llu hits: %llu decode mismatches\n", static_cast<unsigned long long>(cache.hits), static_cast<unsigned long long>(cache.copied), static_cast<unsigned long long>(cache.misses), cache.entries.size(), static_cast<unsigned long long>(cache.verified), static_cast<unsigned long long>(cache.mismatches));
        };
        const auto serial = GuestMemory::ForgetSerial();
        const bool settled = (serial & 1u) == 0;
        const auto key = decodeDigest(queue);
        const auto found = cache.entries.find(key);
        if (found != cache.entries.end() && settled && found->second.forgetSerial == serial && found->second.registry == submission.shaders) {
            auto& entry = found->second;
            cache.order.splice(cache.order.end(), cache.order, entry.order);
            std::shared_ptr<DrawDecode> served;
            // Held by nothing but the entry: no draw uses it any more. use_count reads relaxed:
            // the fence orders the rewrite after the last holder's release.
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
                if (!same && ++cache.mismatches <= 20) std::fprintf(stderr, "[decode-cache] verify: a hit at digest 0x%llx decodes differently\n", static_cast<unsigned long long>(key));
            }
            report();
            decode = std::move(served);
            return;
        }
        auto fresh = decodeDraw(queue, submission);
        ++cache.misses;
        if (found != cache.entries.end()) {
            // The registry or the mappings changed under these registers: the decode is taken again.
            found->second.registry = submission.shaders;
            found->second.forgetSerial = settled ? serial : 1;
            found->second.copies.assign(1, fresh);
            cache.order.splice(cache.order.end(), cache.order, found->second.order);
        } else {
            while (cache.entries.size() >= DecodeCacheEntries && !cache.order.empty()) {
                cache.entries.erase(cache.order.front());
                cache.order.pop_front();
            }
            auto& entry = cache.entries[key];
            entry.registry = submission.shaders;
            entry.forgetSerial = settled ? serial : 1;
            entry.copies.push_back(fresh);
            cache.order.push_back(key);
            entry.order = std::prev(cache.order.end());
        }
        report();
        decode = std::move(fresh);
        return;
    }
    if (decode == nullptr && !verifyDrawRecipe() && DecodeMemoEnabled() && queue.decodeGeneration != 0) {
        auto& memo = ThreadScratch<DecodeMemo, DecodeMemoTag>();
        const auto report = [&] {
            if (!profile && !VerifyDecodeMemo()) return;
            if ((memo.hits + memo.misses) % 500000 != 0) return;
            std::fprintf(stderr, "[decode-memo] %llu hits, %llu misses; verified %llu hits: %llu register mismatches, %llu decode mismatches\n", static_cast<unsigned long long>(memo.hits), static_cast<unsigned long long>(memo.misses), static_cast<unsigned long long>(memo.verified), static_cast<unsigned long long>(memo.registerMismatches), static_cast<unsigned long long>(memo.decodeMismatches));
        };
        const auto serial = GuestMemory::ForgetSerial();
        const bool settled = (serial & 1u) == 0;
        if (settled && memo.decode != nullptr && memo.generation == queue.decodeGeneration && memo.forgetSerial == serial && memo.registry == submission.shaders && memo.decode.use_count() == 1) {
            for (auto& program : memo.decode->programs) rereadUserData(queue, program);
            ++memo.hits;
            if (VerifyDecodeMemo()) {
                ++memo.verified;
                const bool registersSame = sameDecodeRegisters(memo.context, queue.context, false) && sameDecodeRegisters(memo.shader, queue.shader, true) && sameDecodeRegisters(memo.userConfig, queue.userConfig, false);
                const bool decodeSame = sameDecode(*decodeDraw(queue, submission), *memo.decode);
                if (!registersSame) ++memo.registerMismatches;
                if (!decodeSame) ++memo.decodeMismatches;
                if ((!registersSame || !decodeSame) && memo.registerMismatches + memo.decodeMismatches <= 20) std::fprintf(stderr, "[decode-memo] verify: a hit at generation %llu %s\n", static_cast<unsigned long long>(memo.generation), !registersSame ? "came from other registers" : "decodes differently");
            }
            report();
            decode = memo.decode;
            return;
        }
        auto fresh = decodeDraw(queue, submission);
        ++memo.misses;
        memo.generation = queue.decodeGeneration;
        memo.registry = submission.shaders;
        memo.forgetSerial = settled ? serial : 1;
        memo.decode = fresh;
        if (VerifyDecodeMemo()) {
            memo.context = queue.context;
            memo.shader = queue.shader;
            memo.userConfig = queue.userConfig;
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
