#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/exceptions/Unwind.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <mutex>
#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>
#include "Sse4aEmulation.hpp"

namespace {

// The reporter runs on a faulting thread that may itself hold the CRT stream lock (a fault inside
// printf), so it writes straight to the stderr handle.
void Report(const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    int length = std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    if (length <= 0) return;
    if (length > static_cast<int>(sizeof(buffer)) - 1) length = sizeof(buffer) - 1;
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), buffer, static_cast<DWORD>(length), &written, nullptr);
}

void ReportAllThreads();

void DescribeAddress(std::uint64_t address, char* buffer, std::size_t size) {
    HMODULE module = nullptr;
    char path[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module) && module) {
        GetModuleFileNameA(module, path, sizeof(path));
        const char* name = path;
        for (const char* cursor = path; *cursor; ++cursor) {
            if (*cursor == '\\' || *cursor == '/') name = cursor + 1;
        }
        std::snprintf(buffer, size, "0x%016llx %s+0x%llx", static_cast<unsigned long long>(address), name, static_cast<unsigned long long>(address - reinterpret_cast<std::uint64_t>(module)));
        return;
    }
    std::snprintf(buffer, size, "0x%016llx", static_cast<unsigned long long>(address));
}

bool IsExecutable(std::uint64_t address) {
    MEMORY_BASIC_INFORMATION memory{};
    if (VirtualQuery(reinterpret_cast<const void*>(address), &memory, sizeof(memory)) != sizeof(memory)) return false;
    if (memory.State != MEM_COMMIT) return false;
    const DWORD protection = memory.Protect & 0xff;
    return protection == PAGE_EXECUTE || protection == PAGE_EXECUTE_READ || protection == PAGE_EXECUTE_READWRITE || protection == PAGE_EXECUTE_WRITECOPY;
}

bool IsReadable(std::uint64_t address) {
    MEMORY_BASIC_INFORMATION memory{};
    if (VirtualQuery(reinterpret_cast<const void*>(address), &memory, sizeof(memory)) != sizeof(memory)) return false;
    return memory.State == MEM_COMMIT && (memory.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
}

bool IsFatal(DWORD code) {
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
    case EXCEPTION_ILLEGAL_INSTRUCTION:
    case EXCEPTION_PRIV_INSTRUCTION:
    case EXCEPTION_STACK_OVERFLOW:
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
    case EXCEPTION_IN_PAGE_ERROR:
        return true;
    default:
        return false;
    }
}

// Debug aid: APS5_WATCH_WRITE=<guest ELF vaddr> reports every write to that 8-byte location by
// making its page read-only and single-stepping each trapped write.
std::uintptr_t g_watchAddress = 0;
std::uintptr_t g_watchPage = 0;
thread_local bool t_watchStepping = false;
thread_local bool t_watchHit = false;
thread_local unsigned long long t_watchBefore = 0;
thread_local std::uintptr_t t_watchFault = 0;
thread_local char t_watchWhere[MAX_PATH + 64];

void ProtectWatchPage(DWORD protection) {
    DWORD old;
    VirtualProtect(reinterpret_cast<void*>(g_watchPage), 0x1000, protection, &old);
}

bool HandleWatch(EXCEPTION_POINTERS* info) {
    if (g_watchPage == 0) return false;
    const auto* record = info->ExceptionRecord;
    auto* context = info->ContextRecord;
    if (record->ExceptionCode == EXCEPTION_SINGLE_STEP && t_watchStepping) {
        t_watchStepping = false;
        ProtectWatchPage(PAGE_READONLY);
        const auto current = *reinterpret_cast<const unsigned long long*>(g_watchAddress);
        if (t_watchHit || current != t_watchBefore) {
            if (!t_watchHit) Report("[watch] thread %lu changed the value with a write starting at 0x%llx from %s\n", GetCurrentThreadId(), static_cast<unsigned long long>(t_watchFault), t_watchWhere);
            Report("[watch]   new value 0x%016llx at %lld ms\n", current, static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()));
            t_watchHit = false;
        }
        return true;
    }
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || record->ExceptionInformation[0] != 1) return false;
    const auto address = static_cast<std::uintptr_t>(record->ExceptionInformation[1]);
    if (address < g_watchPage || address >= g_watchPage + 0x1000) return false;
    // Block writes (memset, memcpy) fault on their first byte, so compare the value after the step
    // as well as matching the fault address.
    t_watchBefore = *reinterpret_cast<const unsigned long long*>(g_watchAddress);
    t_watchFault = address;
    DescribeAddress(context->Rip, t_watchWhere, sizeof(t_watchWhere));
    if (address + 32 > g_watchAddress && address < g_watchAddress + 8) {
        char line[MAX_PATH + 64];
        DescribeAddress(context->Rip, line, sizeof(line));
        Report("[watch] thread %lu writes 0x%llx at %s rcx=%llx rdx=%llx r8=%llx\n", GetCurrentThreadId(), static_cast<unsigned long long>(address), line, context->Rcx, context->Rdx, context->R8);
        for (std::uint64_t slot = context->Rsp; slot < context->Rsp + 0x80 && IsReadable(slot); slot += 8) {
            const auto value = *reinterpret_cast<const std::uint64_t*>(slot);
            if (!IsExecutable(value)) continue;
            DescribeAddress(value, line, sizeof(line));
            Report("[watch]     [rsp+0x%llx] %s\n", static_cast<unsigned long long>(slot - context->Rsp), line);
        }
        std::uint64_t frame = context->Rbp;
        for (int depth = 0; depth < 6 && frame != 0 && (frame & 7) == 0 && IsReadable(frame) && IsReadable(frame + 8); ++depth) {
            const auto returnAddress = reinterpret_cast<const std::uint64_t*>(frame)[1];
            if (!IsExecutable(returnAddress)) break;
            DescribeAddress(returnAddress, line, sizeof(line));
            Report("[watch]     #%d %s\n", depth, line);
            const auto next = reinterpret_cast<const std::uint64_t*>(frame)[0];
            if (next <= frame) break;
            frame = next;
        }
        t_watchHit = true;
    }
    ProtectWatchPage(PAGE_READWRITE);
    context->EFlags |= 0x100;
    t_watchStepping = true;
    return true;
}

void InstallWatch() {
    // APS5_WATCH_ADDR=<absolute hex address> watches guest data that is mapped later: the page is
    // protected once it has been committed.
    if (const char* absolute = std::getenv("APS5_WATCH_ADDR")) {
        g_watchAddress = std::strtoull(absolute, nullptr, 16);
        const auto page = g_watchAddress & ~static_cast<std::uintptr_t>(0xfff);
        CreateThread(nullptr, 0, [](void* parameter) -> DWORD {
            const auto page = reinterpret_cast<std::uintptr_t>(parameter);
            for (;;) {
                MEMORY_BASIC_INFORMATION memory{};
                if (VirtualQuery(reinterpret_cast<const void*>(page), &memory, sizeof(memory)) == sizeof(memory) && memory.State == MEM_COMMIT) break;
                Sleep(1);
            }
            g_watchPage = page;
            ProtectWatchPage(PAGE_READONLY);
            Report("[watch] watching 0x%llx (page 0x%llx)\n", static_cast<unsigned long long>(g_watchAddress), static_cast<unsigned long long>(page));
            return 0;
        }, reinterpret_cast<void*>(page), 0, nullptr);
        return;
    }
    const char* value = std::getenv("APS5_WATCH_WRITE");
    if (!value) return;
    const auto elfAddress = std::strtoull(value, nullptr, 0);
    g_watchAddress = reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr)) + 0x10000 + elfAddress;
    g_watchPage = g_watchAddress & ~static_cast<std::uintptr_t>(0xfff);
    ProtectWatchPage(PAGE_READONLY);
    Report("[watch] watching 0x%llx (page 0x%llx)\n", static_cast<unsigned long long>(g_watchAddress), static_cast<unsigned long long>(g_watchPage));
}

// SSE4a (EXTRQ / INSERTQ) is AMD-only: the PS5's Zen 2 has it and titles use it, so on an Intel host the
// instruction raises STATUS_ILLEGAL_INSTRUCTION and is emulated on the faulting thread's CONTEXT.
// APS5_NO_SSE4A_EMULATION=1 disables this (the fault is then reported as fatal as before) and
// APS5_TRACE_SSE4A=1 prints each emulated instruction once per rip.
bool g_sse4aEmulation = true;
bool g_sse4aTrace = false;
std::atomic<unsigned long long> g_sse4aEmulated{0};

bool IsEnvironmentSet(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' && !(value[0] == '0' && value[1] == '\0');
}

// Copies the bytes at `address` that are mapped; the read stays within readable pages.
std::size_t ReadCode(std::uint64_t address, std::uint8_t* buffer, std::size_t size) {
    std::size_t count = 0;
    while (count < size) {
        const auto cursor = address + count;
        if ((cursor & 0xfff) == 0 || count == 0) {
            if (!IsReadable(cursor)) break;
        }
        buffer[count++] = *reinterpret_cast<const std::uint8_t*>(cursor);
    }
    return count;
}

void TraceSse4a(std::uint64_t rip, const sse4a::Instruction& instruction, const sse4a::Field& field, unsigned long long count) {
    static std::mutex mutex;
    static std::uint64_t seen[128];
    static std::size_t seenCount = 0;
    {
        std::lock_guard lock(mutex);
        for (std::size_t i = 0; i < seenCount; ++i) {
            if (seen[i] == rip) return;
        }
        if (seenCount < sizeof(seen) / sizeof(seen[0])) seen[seenCount++] = rip;
    }
    char line[MAX_PATH + 64];
    DescribeAddress(rip, line, sizeof(line));
    const char* mnemonic = instruction.op == sse4a::Op::Extrq ? "extrq" : "insertq";
    char operands[64];
    if (instruction.op == sse4a::Op::Extrq && !instruction.registerForm) {
        std::snprintf(operands, sizeof(operands), "xmm%u, %u, %u", instruction.destination, instruction.length, instruction.index);
    } else if (!instruction.registerForm) {
        std::snprintf(operands, sizeof(operands), "xmm%u, xmm%u, %u, %u", instruction.destination, instruction.source, instruction.length, instruction.index);
    } else {
        std::snprintf(operands, sizeof(operands), "xmm%u, xmm%u", instruction.destination, instruction.source);
    }
    Report("[sse4a] #%llu %s %s (length %u, index %u, %u bytes) at %s on thread %lu\n", count, mnemonic, operands, field.length, field.index, static_cast<unsigned>(instruction.size), line, GetCurrentThreadId());
}

bool HandleSse4a(EXCEPTION_POINTERS* info) {
    if (!g_sse4aEmulation) return false;
    const auto* record = info->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ILLEGAL_INSTRUCTION) return false;
    auto* context = info->ContextRecord;
    const auto rip = context->Rip;
    std::uint8_t bytes[sse4a::kMaxInstructionSize];
    const std::size_t available = ReadCode(rip, bytes, sizeof(bytes));
    sse4a::Instruction instruction;
    sse4a::Field field;
    if (!sse4a::Emulate(bytes, available, *context, &instruction, &field)) return false;
    const auto count = ++g_sse4aEmulated;
    if (g_sse4aTrace) {
        TraceSse4a(rip, instruction, field, count);
        if (count % 100000 == 0) Report("[sse4a] %llu instructions emulated so far (tick %llu ms)\n", static_cast<unsigned long long>(count), static_cast<unsigned long long>(GetTickCount64()));
    }
    return true;
}

LONG WINAPI ReportCrash(EXCEPTION_POINTERS* info) {
    static std::atomic<bool> reported{false};
    const auto* fault = info->ExceptionRecord;
    // A read or write of a guarded guest page (GuestArenaGuard): the guard's owner makes the bytes
    // current, the page opens and the access runs again.
    if (fault->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && fault->NumberParameters >= 2 && fault->ExceptionInformation[0] <= 1 && GuestArena::GuestArenaHandleGuard_nid_postfix(fault->ExceptionInformation[1], fault->ExceptionInformation[0] == 1, info->ContextRecord->Rip)) return EXCEPTION_CONTINUE_EXECUTION;
    if (fault->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && fault->NumberParameters >= 2 && fault->ExceptionInformation[0] == 1 && GuestArena::GuestArenaHandleWrite_nid_postfix(fault->ExceptionInformation[1])) return EXCEPTION_CONTINUE_EXECUTION;
    if (HandleWatch(info)) return EXCEPTION_CONTINUE_EXECUTION;
    if (HandleSse4a(info)) return EXCEPTION_CONTINUE_EXECUTION;
    const auto* record = info->ExceptionRecord;
    if (!IsFatal(record->ExceptionCode)) return EXCEPTION_CONTINUE_SEARCH;
    if (reported.exchange(true)) {
        // Another thread is already reporting; let it finish before this fault ends the process.
        Sleep(INFINITE);
    }
    const auto* context = info->ContextRecord;
    char line[MAX_PATH + 64];
    char threadName[128] = "";
    PWSTR description = nullptr;
    if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &description)) && description) {
        WideCharToMultiByte(CP_UTF8, 0, description, -1, threadName, sizeof(threadName), nullptr, nullptr);
        LocalFree(description);
    }
    Report("\nFATAL: unhandled exception 0x%08lx on thread %lu '%s'\n", record->ExceptionCode, GetCurrentThreadId(), threadName);
    DescribeAddress(context->Rip, line, sizeof(line));
    Report("  rip %s\n", line);
    if (record->ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION) {
        std::uint8_t bytes[8];
        const std::size_t available = ReadCode(context->Rip, bytes, sizeof(bytes));
        char hex[3 * sizeof(bytes) + 1] = "";
        for (std::size_t i = 0; i < available; ++i) std::snprintf(hex + 3 * i, sizeof(hex) - 3 * i, "%02x ", bytes[i]);
        Report("  bytes %s(sse4a emulation %s, %llu emulated so far)\n", hex, g_sse4aEmulation ? "on" : "off", g_sse4aEmulated.load());
    }
    if (record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record->NumberParameters >= 2) {
        const char* kind = record->ExceptionInformation[0] == 0 ? "read" : record->ExceptionInformation[0] == 1 ? "write" : "execute";
        Report("  %s of 0x%016llx\n", kind, static_cast<unsigned long long>(record->ExceptionInformation[1]));
    }
    Report("  rax %016llx rbx %016llx rcx %016llx rdx %016llx\n", context->Rax, context->Rbx, context->Rcx, context->Rdx);
    Report("  rsi %016llx rdi %016llx rbp %016llx rsp %016llx\n", context->Rsi, context->Rdi, context->Rbp, context->Rsp);
    Report("  r8  %016llx r9  %016llx r10 %016llx r11 %016llx\n", context->R8, context->R9, context->R10, context->R11);
    Report("  r12 %016llx r13 %016llx r14 %016llx r15 %016llx\n", context->R12, context->R13, context->R14, context->R15);
    // The words each register points at, to recognise which structure a bad pointer came from.
    {
        const std::pair<const char*, std::uint64_t> pointers[] = {{"rax", context->Rax}, {"rbx", context->Rbx}, {"rcx", context->Rcx}, {"rdx", context->Rdx}, {"rsi", context->Rsi}, {"rdi", context->Rdi}, {"r12", context->R12}, {"r13", context->R13}, {"r14", context->R14}, {"r15", context->R15}};
        for (const auto& [name, value] : pointers) {
            if (value < 0x10000 || (value & 7) != 0 || !IsReadable(value) || !IsReadable(value + 0x38)) continue;
            const auto* words = reinterpret_cast<const std::uint64_t*>(value);
            Report("  [%s] %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx\n", name, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
        }
    }
    Report("  frame pointer chain:\n");
    std::uint64_t frame = context->Rbp;
    for (int depth = 0; depth < 32 && frame != 0 && (frame & 7) == 0 && IsReadable(frame) && IsReadable(frame + 8); ++depth) {
        const auto returnAddress = reinterpret_cast<const std::uint64_t*>(frame)[1];
        if (!IsExecutable(returnAddress)) break;
        DescribeAddress(returnAddress, line, sizeof(line));
        Report("    #%d %s\n", depth, line);
        const auto next = reinterpret_cast<const std::uint64_t*>(frame)[0];
        if (next <= frame) break;
        frame = next;
    }
    Report("  stack return address candidates:\n");
    int printed = 0;
    for (std::uint64_t slot = context->Rsp; printed < 16 && slot < context->Rsp + 0x2000; slot += 8) {
        if (!IsReadable(slot)) break;
        const auto value = *reinterpret_cast<const std::uint64_t*>(slot);
        if (!IsExecutable(value)) continue;
        DescribeAddress(value, line, sizeof(line));
        Report("    [rsp+0x%llx] %s\n", static_cast<unsigned long long>(slot - context->Rsp), line);
        ++printed;
    }
    if (context->Rip == 0 || !IsExecutable(context->Rip)) {
        // A jump into nothing usually follows a return through a clobbered frame; the frames that just
        // returned are still below rsp.
        Report("  recently popped return address candidates:\n");
        for (std::uint64_t slot = context->Rsp - 8; slot >= context->Rsp - 0x800; slot -= 8) {
            if (!IsReadable(slot)) break;
            const auto value = *reinterpret_cast<const std::uint64_t*>(slot);
            if (!IsExecutable(value)) continue;
            DescribeAddress(value, line, sizeof(line));
            Report("    [rsp-0x%llx] %s\n", static_cast<unsigned long long>(context->Rsp - slot), line);
        }
        // Registers often still point into the stack the code ran on before the bad return.
        const std::pair<const char*, std::uint64_t> registers[] = {{"rdi", context->Rdi}, {"rsi", context->Rsi}, {"rbx", context->Rbx}, {"r12", context->R12}, {"r13", context->R13}, {"r14", context->R14}, {"r15", context->R15}};
        for (const auto& [name, value] : registers) {
            if (value < 0x10000 || (value & 7) != 0 || !IsReadable(value)) continue;
            Report("  code addresses above %s (0x%llx):\n", name, static_cast<unsigned long long>(value));
            int found = 0;
            for (std::uint64_t slot = value; found < 24 && slot < value + 0x1000; slot += 8) {
                if (!IsReadable(slot)) break;
                const auto candidate = *reinterpret_cast<const std::uint64_t*>(slot);
                if (!IsExecutable(candidate)) continue;
                DescribeAddress(candidate, line, sizeof(line));
                Report("    [+0x%llx] %s\n", static_cast<unsigned long long>(slot - value), line);
                ++found;
            }
        }
    }
    std::fflush(stderr);
    ReportAllThreads();
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

std::size_t CaptureFrames(const CONTEXT* context, std::uint64_t* frames, std::size_t capacity) {
    std::size_t count = 0;
    std::uint64_t frame = context->Rbp;
    while (count < capacity && frame != 0 && (frame & 7) == 0 && IsReadable(frame) && IsReadable(frame + 8)) {
        const auto returnAddress = reinterpret_cast<const std::uint64_t*>(frame)[1];
        if (!IsExecutable(returnAddress)) break;
        frames[count++] = returnAddress;
        const auto next = reinterpret_cast<const std::uint64_t*>(frame)[0];
        if (next <= frame) break;
        frame = next;
    }
    return count;
}

void ReportThreadContext(const CONTEXT* context, DWORD threadId, const char* name, const std::uint64_t* frames, std::size_t frameCount) {
    char line[MAX_PATH + 64];
    Report("  thread %lu '%s': rip ", static_cast<unsigned long>(threadId), name ? name : "");
    DescribeAddress(context->Rip, line, sizeof(line));
    Report("%s rsp 0x%016llx rbp 0x%016llx\n", line, static_cast<unsigned long long>(context->Rsp), static_cast<unsigned long long>(context->Rbp));
    for (std::size_t depth = 0; depth < frameCount; ++depth) {
        DescribeAddress(frames[depth], line, sizeof(line));
        Report("    #%llu %s\n", static_cast<unsigned long long>(depth), line);
    }
}

void ReportAllThreads() {
    const DWORD current = GetCurrentThreadId();
    const DWORD pid = GetCurrentProcessId();
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    struct CapturedThread {
        DWORD id;
        char name[128];
        CONTEXT context;
        std::uint64_t frames[12];
        std::size_t frameCount;
    };
    static CapturedThread captured[128];
    std::size_t capturedCount = 0;
    std::size_t skippedCount = 0;
    if (Thread32First(snapshot, &entry)) {
        do {
            if (entry.th32OwnerProcessID != pid || entry.th32ThreadID == current) continue;
            if (capturedCount >= sizeof(captured) / sizeof(captured[0])) {
                ++skippedCount;
                continue;
            }
            HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
            if (!thread) continue;
            if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
                CloseHandle(thread);
                continue;
            }
            CONTEXT context{};
            context.ContextFlags = CONTEXT_FULL;
            const BOOL haveContext = GetThreadContext(thread, &context);
            std::uint64_t frames[12] = {};
            const std::size_t frameCount = haveContext ? CaptureFrames(&context, frames, sizeof(frames) / sizeof(frames[0])) : 0;
            ResumeThread(thread);
            char name[128] = "";
            PWSTR description = nullptr;
            if (SUCCEEDED(GetThreadDescription(thread, &description)) && description) {
                WideCharToMultiByte(CP_UTF8, 0, description, -1, name, sizeof(name), nullptr, nullptr);
                LocalFree(description);
            }
            CloseHandle(thread);
            if (!haveContext) continue;
            captured[capturedCount].id = entry.th32ThreadID;
            std::memcpy(captured[capturedCount].name, name, sizeof(captured[capturedCount].name));
            captured[capturedCount].context = context;
            std::memcpy(captured[capturedCount].frames, frames, sizeof(captured[capturedCount].frames));
            captured[capturedCount].frameCount = frameCount;
            ++capturedCount;
        } while (Thread32Next(snapshot, &entry));
    }
    CloseHandle(snapshot);
    Report("  all threads:\n");
    for (std::size_t i = 0; i < capturedCount; ++i) {
        ReportThreadContext(&captured[i].context, captured[i].id, captured[i].name, captured[i].frames, captured[i].frameCount);
    }
    if (skippedCount != 0) Report("  %llu further thread(s) left out\n", static_cast<unsigned long long>(skippedCount));
}

unsigned long long EnvironmentSeconds(const char* name, unsigned long long fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] < '0' || text[0] > '9') return fallback;
    char* end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    return end != nullptr && *end == 0 ? value : fallback;
}

// A module's DWARF unwind records as code ranges: those of its .eh_frame (renamed .ehfram), or of a
// relinked guest's .ehmeta search table.
struct UnwindRecord {
    std::uint64_t begin;
    std::uint64_t end;
    const LibcUnwind::Byte* fde;
};

void IndexUnwindRecords(std::uint64_t base, const IMAGE_SECTION_HEADER& section, std::vector<UnwindRecord>& records) {
    using LibcUnwind::Byte;
    using LibcUnwind::Word;
    const auto add = [&](const Byte* fde) {
        Word start = 0;
        Word length = 0;
        if (LibcUnwind::FdeRange(fde, start, length) && length != 0) records.push_back({start, start + length, fde});
    };
    const auto* p = reinterpret_cast<const Byte*>(base + section.VirtualAddress);
    if (std::memcmp(section.Name, ".ehfram", 8) == 0) {
        const auto* end = p + section.Misc.VirtualSize;
        while (end - p >= 8) {
            const auto* record = p;
            const auto length = LibcUnwind::Read<std::uint32_t>(p);
            if (length == 0) continue;
            if (length == 0xffffffff || length < 4 || static_cast<std::uint64_t>(end - p) < length) return;
            if (LibcUnwind::Read<std::uint32_t>(p) != 0) add(record);
            p = record + 4 + length;
        }
    } else if (std::memcmp(section.Name, ".ehmeta", 8) == 0) {
        const auto* header = reinterpret_cast<const Byte*>(base + LibcUnwind::Read<std::uint32_t>(p));
        if (header[0] != 1 || header[3] == 255) return;
        p = header + 4;
        LibcUnwind::Encoded(p, header[1], Word(header));
        const Word count = LibcUnwind::Encoded(p, header[2]);
        for (Word i = 0; i < count; ++i) {
            LibcUnwind::Encoded(p, header[3], Word(header));
            add(reinterpret_cast<const Byte*>(LibcUnwind::Encoded(p, header[3], Word(header))));
        }
    }
}

// A leaf is in an epilogue when only pops and stack releases lie between it and its return (or the
// jump of a tail call); replaying them moves the context to the caller. Synchronous DWARF records
// (-fno-asynchronous-unwind-tables) are exact only at calls, so they need not describe an epilogue.
bool ReplayEpilogue(CONTEXT& context, std::uint64_t low, std::uint64_t high) {
    static constexpr DWORD64 CONTEXT::* Registers[16] = {
        &CONTEXT::Rax, &CONTEXT::Rcx, &CONTEXT::Rdx, &CONTEXT::Rbx, &CONTEXT::Rsp, &CONTEXT::Rbp, &CONTEXT::Rsi, &CONTEXT::Rdi,
        &CONTEXT::R8, &CONTEXT::R9, &CONTEXT::R10, &CONTEXT::R11, &CONTEXT::R12, &CONTEXT::R13, &CONTEXT::R14, &CONTEXT::R15};
    CONTEXT replay = context;
    const auto* code = reinterpret_cast<const std::uint8_t*>(context.Rip);
    bool released = false;
    for (int step = 0; step < 16; ++step) {
        if (replay.Rsp < low || replay.Rsp + 8 > high) return false;
        const auto top = *reinterpret_cast<const DWORD64*>(replay.Rsp);
        const bool extended = code[0] == 0x41;
        const auto opcode = code[extended ? 1 : 0];
        if (opcode >= 0x58 && opcode <= 0x5f && (extended || opcode != 0x5c)) {
            replay.*Registers[opcode - 0x58 + (extended ? 8 : 0)] = top;
            replay.Rsp += 8;
            code += extended ? 2 : 1;
            released = true;
            continue;
        }
        if (code[0] == 0x48 && code[1] == 0x83 && code[2] == 0xc4) {
            replay.Rsp += static_cast<std::int8_t>(code[3]);
            code += 4;
            released = true;
            continue;
        }
        if (code[0] == 0x48 && code[1] == 0x81 && code[2] == 0xc4) {
            std::int32_t amount;
            std::memcpy(&amount, code + 3, sizeof(amount));
            replay.Rsp += amount;
            code += 7;
            released = true;
            continue;
        }
        const bool ret = code[0] == 0xc3 || (code[0] == 0xf3 && code[1] == 0xc3);
        const bool tail = released && (code[0] == 0xe9 || code[0] == 0xeb || (code[0] == 0xff && code[1] == 0x25) || (code[0] == 0x48 && code[1] == 0xff && code[2] == 0x25));
        if (!ret && !tail) return false;
        replay.Rip = top;
        replay.Rsp += 8;
        context = replay;
        return true;
    }
    return false;
}

// One frame through its DWARF record; saved registers are read only inside [low, high).
bool StepDwarf(CONTEXT& context, const LibcUnwind::Byte* fde, bool leaf, std::uint64_t low, std::uint64_t high) {
    static constexpr DWORD64 CONTEXT::* Registers[17] = {
        &CONTEXT::Rax, &CONTEXT::Rdx, &CONTEXT::Rcx, &CONTEXT::Rbx, &CONTEXT::Rsi, &CONTEXT::Rdi, &CONTEXT::Rbp, &CONTEXT::Rsp,
        &CONTEXT::R8, &CONTEXT::R9, &CONTEXT::R10, &CONTEXT::R11, &CONTEXT::R12, &CONTEXT::R13, &CONTEXT::R14, &CONTEXT::R15, &CONTEXT::Rip};
    _Unwind_Context frame;
    for (unsigned i = 0; i < 17; ++i) frame.registers[i] = context.*Registers[i];
    frame.signalFrame = leaf;
    if (!LibcUnwind::StepWithin(frame, fde, low, high)) return false;
    for (unsigned i = 0; i < 17; ++i) context.*Registers[i] = frame.registers[i];
    return true;
}

// APS5_SAMPLE_SECONDS: after APS5_SAMPLE_DELAY seconds (default 60), or once the file
// APS5_SAMPLE_TRIGGER names exists, every thread's instruction pointer is sampled for that many
// seconds; the busy threads and the hottest module offsets are reported. A thread is suspended only
// for the context read: nothing is allocated while it is held.
DWORD WINAPI SampleProfiler(LPVOID param) {
    const auto seconds = static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(param));
    if (const char* trigger = std::getenv("APS5_SAMPLE_TRIGGER"); trigger != nullptr && trigger[0] != 0) {
        while (GetFileAttributesA(trigger) == INVALID_FILE_ATTRIBUTES) Sleep(100);
    } else {
        Sleep(static_cast<DWORD>(EnvironmentSeconds("APS5_SAMPLE_DELAY", 60) * 1000ull));
    }
    struct Sampled {
        DWORD id;
        HANDLE handle;
        std::string name;
        unsigned long long busy = 0;
        unsigned long long total = 0;
    };
    const DWORD self = GetCurrentThreadId();
    const DWORD pid = GetCurrentProcessId();
    std::vector<Sampled> threads;
    std::unordered_map<std::uint64_t, unsigned long long> hits;
    // Every sample is charged to its stack, unwound over a copy taken while the thread was
    // suspended. A leaf just after a syscall instruction is a kernel call; one of the wait services
    // is a blocked thread, any other is work.
    struct Range {
        std::uint64_t begin;
        std::uint64_t end;
        bool own;
    };
    std::vector<Range> code;
    std::vector<UnwindRecord> unwindRecords;
    {
        HANDLE modules = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, pid);
        if (modules != INVALID_HANDLE_VALUE) {
            MODULEENTRY32 module{};
            module.dwSize = sizeof(module);
            if (Module32First(modules, &module)) {
                do {
                    const std::string name = module.szModule;
                    const bool own = name.size() > 4 && (name.compare(name.size() - 4, 4, ".prx") == 0 || name.compare(name.size() - 4, 4, ".exe") == 0);
                    const auto base = reinterpret_cast<std::uint64_t>(module.modBaseAddr);
                    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
                    if (dos->e_magic != IMAGE_DOS_SIGNATURE) continue;
                    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + static_cast<std::uint64_t>(dos->e_lfanew));
                    if (nt->Signature != IMAGE_NT_SIGNATURE) continue;
                    const auto* section = IMAGE_FIRST_SECTION(nt);
                    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section) {
                        if (own) IndexUnwindRecords(base, *section, unwindRecords);
                        if ((section->Characteristics & IMAGE_SCN_MEM_EXECUTE) == 0 || (section->Characteristics & IMAGE_SCN_MEM_READ) == 0) continue;
                        const auto size = std::max<std::uint64_t>(section->Misc.VirtualSize, section->SizeOfRawData);
                        code.push_back({base + section->VirtualAddress, base + section->VirtualAddress + size, own});
                    }
                } while (Module32Next(modules, &module));
            }
            CloseHandle(modules);
        }
    }
    std::sort(code.begin(), code.end(), [](const Range& a, const Range& b) { return a.begin < b.begin; });
    const auto codeAt = [&](std::uint64_t address) -> const Range* {
        auto it = std::upper_bound(code.begin(), code.end(), address, [](std::uint64_t value, const Range& range) { return value < range.begin; });
        if (it == code.begin()) return nullptr;
        --it;
        return address < it->end ? &*it : nullptr;
    };
    std::sort(unwindRecords.begin(), unwindRecords.end(), [](const UnwindRecord& a, const UnwindRecord& b) { return a.begin < b.begin; });
    const auto recordAt = [&](std::uint64_t address) -> const LibcUnwind::Byte* {
        auto it = std::upper_bound(unwindRecords.begin(), unwindRecords.end(), address, [](std::uint64_t value, const UnwindRecord& record) { return value < record.begin; });
        if (it == unwindRecords.begin()) return nullptr;
        --it;
        return address < it->end ? it->fde : nullptr;
    };
    // Module+offset, or the nearest export for a system module (ntdll's Nt services).
    std::unordered_map<std::uint64_t, std::string> names;
    const auto nameOf = [&](std::uint64_t address) -> const std::string& {
        auto [it, inserted] = names.try_emplace(address);
        if (!inserted) return it->second;
        HMODULE module = nullptr;
        char path[MAX_PATH] = "?";
        char text[MAX_PATH + 160];
        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module) || module == nullptr) {
            std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(address));
            it->second = text;
            return it->second;
        }
        GetModuleFileNameA(module, path, sizeof(path));
        const char* name = path;
        for (const char* cursor = path; *cursor; ++cursor) {
            if (*cursor == '\\' || *cursor == '/') name = cursor + 1;
        }
        const auto base = reinterpret_cast<std::uint64_t>(module);
        const auto* range = codeAt(address);
        const char* best = nullptr;
        std::uint64_t bestAddress = 0;
        if (range == nullptr || !range->own) {
            const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + static_cast<std::uint64_t>(reinterpret_cast<const IMAGE_DOS_HEADER*>(base)->e_lfanew));
            const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
            if (directory.VirtualAddress != 0 && directory.Size != 0) {
                const auto* exports = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + directory.VirtualAddress);
                const auto* functions = reinterpret_cast<const DWORD*>(base + exports->AddressOfFunctions);
                const auto* exportNames = reinterpret_cast<const DWORD*>(base + exports->AddressOfNames);
                const auto* ordinals = reinterpret_cast<const WORD*>(base + exports->AddressOfNameOrdinals);
                for (DWORD i = 0; i < exports->NumberOfNames; ++i) {
                    const auto function = base + functions[ordinals[i]];
                    if (function <= address && function > bestAddress) {
                        bestAddress = function;
                        best = reinterpret_cast<const char*>(base + exportNames[i]);
                    }
                }
            }
        }
        if (best != nullptr && address - bestAddress < 0x4000) std::snprintf(text, sizeof(text), "%s!%s+0x%llx", name, best, static_cast<unsigned long long>(address - bestAddress));
        else std::snprintf(text, sizeof(text), "%s+0x%llx", name, static_cast<unsigned long long>(address - base));
        it->second = text;
        return it->second;
    };
    struct Kinds {
        unsigned long long run = 0;
        unsigned long long kernel = 0;
        unsigned long long wait = 0;
    };
    std::unordered_map<DWORD, Kinds> kinds;
    std::unordered_map<std::string, unsigned long long> stacks;
    // The stack copy, with as much zeroed margin past it: the unwinder may read a frame's saved
    // registers beyond the copied part.
    constexpr std::size_t StackCopyBytes = 128 * 1024;
    constexpr std::size_t MaxFrames = 24;
    auto* stackCopy = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, StackCopyBytes * 2, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (stackCopy == nullptr) return 0;
    const auto refresh = [&] {
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snapshot == INVALID_HANDLE_VALUE) return;
        THREADENTRY32 entry{};
        entry.dwSize = sizeof(entry);
        if (Thread32First(snapshot, &entry)) {
            do {
                if (entry.th32OwnerProcessID != pid || entry.th32ThreadID == self) continue;
                if (std::any_of(threads.begin(), threads.end(), [&](const Sampled& known) { return known.id == entry.th32ThreadID; })) continue;
                HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
                if (!thread) continue;
                std::string name;
                PWSTR description = nullptr;
                if (SUCCEEDED(GetThreadDescription(thread, &description)) && description) {
                    char text[128] = "";
                    WideCharToMultiByte(CP_UTF8, 0, description, -1, text, sizeof(text), nullptr, nullptr);
                    name = text;
                    LocalFree(description);
                }
                threads.push_back({entry.th32ThreadID, thread, name});
            } while (Thread32Next(snapshot, &entry));
        }
        CloseHandle(snapshot);
    };
    const auto end = GetTickCount64() + seconds * 1000ull;
    auto nextRefresh = 0ull;
    unsigned long long rounds = 0;
    while (GetTickCount64() < end) {
        if (GetTickCount64() >= nextRefresh) {
            refresh();
            nextRefresh = GetTickCount64() + 1000ull;
        }
        for (auto& thread : threads) {
            if (SuspendThread(thread.handle) == static_cast<DWORD>(-1)) continue;
            CONTEXT context{};
            context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            const BOOL read = GetThreadContext(thread.handle, &context);
            std::size_t copied = 0;
            if (read) {
                MEMORY_BASIC_INFORMATION region{};
                if (VirtualQuery(reinterpret_cast<LPCVOID>(context.Rsp), &region, sizeof(region)) != 0 && region.State == MEM_COMMIT) {
                    const auto end = reinterpret_cast<std::uint64_t>(region.BaseAddress) + region.RegionSize;
                    copied = static_cast<std::size_t>(std::min<std::uint64_t>(StackCopyBytes, end - context.Rsp));
                    std::memcpy(stackCopy, reinterpret_cast<const void*>(context.Rsp), copied);
                }
            }
            ResumeThread(thread.handle);
            if (!read) continue;
            // The frames, unwound over the copy (registers that point into the thread's stack are
            // moved into the copy first; a frame leaving the copy ends the walk): through a module's
            // .pdata, an epilogue replay at the leaf, or the module's DWARF records. Our modules'
            // compiled C++ has no .pdata, and no DWARF record where the compiler proved a function
            // cannot throw: the walk ends there. Other code without unwind data is a leaf whose
            // return address is at RSP.
            std::uint64_t frames[MaxFrames];
            std::size_t frameCount = 0;
            {
                CONTEXT unwind = context;
                const auto originalLow = context.Rsp;
                const auto originalHigh = context.Rsp + copied;
                const auto copyLow = reinterpret_cast<std::uint64_t>(stackCopy);
                const auto copyHigh = copyLow + copied;
                const auto rebase = [&] {
                    for (DWORD64* value : {&unwind.Rsp, &unwind.Rbp, &unwind.Rbx, &unwind.Rsi, &unwind.Rdi, &unwind.R12, &unwind.R13, &unwind.R14, &unwind.R15}) {
                        if (*value >= originalLow && *value < originalHigh) *value = *value - originalLow + copyLow;
                    }
                };
                rebase();
                const auto step = [&](const Range& range, bool leaf) {
                    DWORD64 imageBase = 0;
                    if (auto* function = RtlLookupFunctionEntry(unwind.Rip, &imageBase, nullptr); function != nullptr) {
                        PVOID handlerData = nullptr;
                        DWORD64 establisher = 0;
                        RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, unwind.Rip, function, &unwind, &handlerData, &establisher, nullptr);
                        return true;
                    }
                    if (leaf && ReplayEpilogue(unwind, copyLow, copyHigh)) return true;
                    if (const auto* fde = recordAt(leaf ? unwind.Rip : unwind.Rip - 1); fde != nullptr) return StepDwarf(unwind, fde, leaf, copyLow, copyHigh);
                    if (range.own) return false;
                    unwind.Rip = *reinterpret_cast<const DWORD64*>(unwind.Rsp);
                    unwind.Rsp += 8;
                    return true;
                };
                while (frameCount < MaxFrames) {
                    frames[frameCount++] = unwind.Rip;
                    const auto* range = codeAt(unwind.Rip);
                    if (copied == 0 || unwind.Rsp < copyLow || unwind.Rsp + 8 > copyHigh || range == nullptr || !step(*range, frameCount == 1)) break;
                    rebase();
                    if (unwind.Rip == 0 || unwind.Rsp < copyLow || unwind.Rsp > copyHigh) break;
                }
            }
            ++thread.total;
            const auto* leafCode = codeAt(context.Rip);
            const auto* leafBytes = reinterpret_cast<const std::uint8_t*>(context.Rip);
            const bool kernel = leafCode != nullptr && context.Rip >= leafCode->begin + 2 && leafBytes[-2] == 0x0f && leafBytes[-1] == 0x05;
            const auto& leaf = nameOf(context.Rip);
            const bool waiting = kernel && (leaf.find("Wait") != std::string::npos || leaf.find("Delay") != std::string::npos || leaf.find("RemoveIoCompletion") != std::string::npos);
            auto& kind = kinds[thread.id];
            if (waiting) ++kind.wait;
            else if (kernel) ++kind.kernel;
            else ++kind.run;
            if (!waiting) {
                ++thread.busy;
                ++hits[context.Rip];
            }
            std::string key = std::to_string(thread.id) + (waiting ? " wait " : kernel ? " kernel " : " run ") + leaf;
            for (std::size_t frame = 1; frame < frameCount; ++frame) {
                key += " <- ";
                // A return address names its call: one byte back lies inside the calling instruction.
                key += nameOf(frames[frame] - 1);
            }
            ++stacks[key];
        }
        ++rounds;
        Sleep(1);
    }
    // APS5_SAMPLE_OUT names a file for the report (other threads' stderr lines cannot interleave
    // with it there); stderr otherwise.
    HANDLE out = INVALID_HANDLE_VALUE;
    if (const char* path = std::getenv("APS5_SAMPLE_OUT"); path != nullptr && path[0] != 0) out = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    const auto emit = [&](const char* format, ...) {
        char buffer[2048];
        va_list args;
        va_start(args, format);
        int length = std::vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);
        if (length <= 0) return;
        if (length > static_cast<int>(sizeof(buffer)) - 1) length = sizeof(buffer) - 1;
        DWORD written = 0;
        WriteFile(out != INVALID_HANDLE_VALUE ? out : GetStdHandle(STD_ERROR_HANDLE), buffer, static_cast<DWORD>(length), &written, nullptr);
    };
    std::sort(threads.begin(), threads.end(), [](const Sampled& a, const Sampled& b) { return a.busy > b.busy; });
    emit("[sample] %llu rounds over %llu s; threads by samples running / in other kernel calls / blocked in a wait (of all):\n", rounds, seconds);
    std::unordered_map<DWORD, bool> busyThread;
    for (std::size_t i = 0; i < threads.size() && i < 24 && threads[i].busy != 0; ++i) {
        const auto& kind = kinds[threads[i].id];
        emit("[sample]   thread %lu '%s' run %llu kernel %llu wait %llu / %llu\n", static_cast<unsigned long>(threads[i].id), threads[i].name.c_str(), kind.run, kind.kernel, kind.wait, threads[i].total);
        busyThread[threads[i].id] = threads[i].busy * 50 >= threads[i].total;
    }
    std::unordered_map<std::string, unsigned long long> byOffset;
    unsigned long long busy = 0;
    for (const auto& [rip, count] : hits) {
        char line[MAX_PATH + 64];
        DescribeAddress(rip, line, sizeof(line));
        byOffset[line] += count;
        busy += count;
    }
    std::vector<std::pair<std::string, unsigned long long>> ranked(byOffset.begin(), byOffset.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    emit("[sample] hottest instructions (%llu busy samples):\n", busy);
    for (std::size_t i = 0; i < ranked.size() && i < 80; ++i) emit("[sample]   %6llu %5.2f%% %s\n", ranked[i].second, busy != 0 ? 100.0 * static_cast<double>(ranked[i].second) / static_cast<double>(busy) : 0.0, ranked[i].first.c_str());
    // Every stack of a thread busy at least 2% of its samples (its waits too: they show what it
    // blocks on), then the other threads' most common waits.
    std::vector<std::pair<std::string, unsigned long long>> rankedStacks(stacks.begin(), stacks.end());
    std::sort(rankedStacks.begin(), rankedStacks.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    const auto threadOf = [](const std::string& key) { return static_cast<DWORD>(std::strtoul(key.c_str(), nullptr, 10)); };
    emit("[sample] stacks of the busy threads (thread, kind, leaf <- return addresses: system frames up to the first emulator or game frame, then emulator and game frames):\n");
    std::size_t emitted = 0;
    for (const auto& [key, count] : rankedStacks) {
        if (emitted >= 3000) break;
        if (!busyThread[threadOf(key)] || count < 2) continue;
        emit("[sample-stack] %6llu %s\n", count, key.c_str());
        ++emitted;
    }
    emit("[sample] other threads' stacks:\n");
    emitted = 0;
    for (const auto& [key, count] : rankedStacks) {
        if (emitted >= 150) break;
        if (busyThread[threadOf(key)]) continue;
        emit("[sample-stack] %6llu %s\n", count, key.c_str());
        ++emitted;
    }
    if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
    for (const auto& thread : threads) CloseHandle(thread.handle);
    return 0;
}

DWORD WINAPI HangWatchdog(LPVOID param) {
    const auto seconds = static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(param));
    Sleep(static_cast<DWORD>(seconds * 1000ull));
    Report("\nFATAL: timed dump after %llu s (APS5_HANG_DUMP_SECS); aborting\n", seconds);
    std::fflush(stderr);
    std::abort();
    return 0;
}

// abort() and the UCRT's invalid-parameter path end the process with a silent fast fail
// (0xc0000409); both are reported with the calling thread's stack first.
void ReportBacktrace(const char* what, bool reportThreads) {
    void* frames[48];
    const auto count = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
    Report("FATAL: %s on thread %lu\n", what, static_cast<unsigned long>(GetCurrentThreadId()));
    char line[256];
    for (USHORT i = 0; i < count; ++i) {
        DescribeAddress(reinterpret_cast<std::uint64_t>(frames[i]), line, sizeof(line));
        Report("    #%u %s\n", static_cast<unsigned>(i), line);
    }
    if (reportThreads) ReportAllThreads();
    std::fflush(stderr);
}

void AbortSignalHandler(int) {
    ReportBacktrace("abort() (SIGABRT)", true);
}

void InvalidParameterHandler(const wchar_t*, const wchar_t*, const wchar_t*, unsigned, std::uintptr_t) {
    ReportBacktrace("UCRT invalid parameter", true);
}

// Static destruction runs on the thread that ends the process (ExitProcess -> DLL_PROCESS_DETACH),
// so its stack names the exit's caller when no libc exit/abort path was taken.
struct ExitReporter {
    ~ExitReporter() {
        if (std::getenv("APS5_TRACE_EXIT") != nullptr) ReportBacktrace("process exit (static destruction)", false);
    }
} g_exitReporter;

const bool g_crashReportInstalled = [] {
    g_sse4aEmulation = !IsEnvironmentSet("APS5_NO_SSE4A_EMULATION");
    g_sse4aTrace = IsEnvironmentSet("APS5_TRACE_SSE4A");
    AddVectoredExceptionHandler(1, ReportCrash);
    std::signal(SIGABRT, AbortSignalHandler);
    _set_invalid_parameter_handler(InvalidParameterHandler);
    InstallWatch();
    if (const char* hang = std::getenv("APS5_HANG_DUMP_SECS")) {
        unsigned long long seconds = 0;
        if (hang[0] >= '0' && hang[0] <= '9') {
            char* end = nullptr;
            const auto value = std::strtoull(hang, &end, 10);
            if (end && *end == 0) seconds = value;
        }
        if (seconds > 0) CreateThread(nullptr, 0, HangWatchdog, reinterpret_cast<LPVOID>(static_cast<std::uintptr_t>(seconds)), 0, nullptr);
    }
    if (const auto seconds = EnvironmentSeconds("APS5_SAMPLE_SECONDS", 0); seconds > 0) CreateThread(nullptr, 0, SampleProfiler, reinterpret_cast<LPVOID>(static_cast<std::uintptr_t>(seconds)), 0, nullptr);
    return true;
}();

}
#endif
