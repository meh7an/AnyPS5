#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#include <windows.h>
#endif
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
namespace GuestMemory = AgcDriver::GuestMemory;

#ifdef _WIN32
constexpr std::size_t Unit = 65536;
constexpr std::size_t BlockBytes = 8 * Unit;

struct Touch {
    std::uintptr_t address;
    bool write;
};
std::vector<Touch> touches;
bool openUnit = false;

// The guard handler: records the touch and, when asked, opens the touch's whole unit (the arena
// opens the touched page alone otherwise).
void recordTouch(std::uintptr_t address, bool write, std::uintptr_t) {
    touches.push_back({address, write});
    if (openUnit) GuestArena::GuestArenaUnguard_nid_postfix(reinterpret_cast<void*>(address & ~(Unit - 1)), Unit);
}

bool guarded(std::uintptr_t address, std::uintptr_t* end = nullptr, std::uint32_t* protection = nullptr) {
    std::uintptr_t runEnd = 0;
    std::uint32_t logical = 0;
    const bool result = GuestArena::GuestArenaGuardRun_nid_postfix(address, address + BlockBytes, &runEnd, &logical);
    if (end != nullptr) *end = runEnd;
    if (protection != nullptr) *protection = logical;
    return result;
}

// The faulting accesses, fenced: the handler runs inside them on this thread, so the compiler must
// not move the test's own stores (openUnit) or loads (touches) across them.
[[gnu::noinline]] std::uint8_t readByte(std::uintptr_t address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const std::uint8_t value = *reinterpret_cast<volatile const std::uint8_t*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return value;
}

[[gnu::noinline]] void writeByte(std::uintptr_t address, std::uint8_t value) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    *reinterpret_cast<volatile std::uint8_t*>(address) = value;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

// The guest arena's guards (GuestArenaGuard): a guarded unit reads and writes as before through the
// fault path, which calls the handler first; the driver's page classification counts a guarded
// page at its logical protection; a protection change keeps the guard with the new logical
// protection; a host read (GuestArenaOpenForHostRead) reaches the handler and opens first; a reset
// drops it.
int PageGuardTests() {
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, Unit);
    GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
    const auto base = reinterpret_cast<std::uintptr_t>(block);
    std::memset(block, 0x5a, BlockBytes);
    GuestArena::GuestArenaSetGuardHandler_nid_postfix(&recordTouch);

    // Refused: an unaligned range, a range outside the arena.
    Require(!GuestArena::GuestArenaGuard_nid_postfix(reinterpret_cast<void*>(base + 4096), Unit), "page guards: an unaligned range was guarded");
    std::vector<std::uint8_t> outside(2 * Unit);
    Require(!GuestArena::GuestArenaGuard_nid_postfix(outside.data(), Unit), "page guards: memory outside the arena was guarded");

    // Units 2 and 3 guarded: one run at the logical protection, units 0 and 1 outside it.
    Require(GuestArena::GuestArenaGuard_nid_postfix(reinterpret_cast<void*>(base + 2 * Unit), 2 * Unit), "page guards: committed memory was refused");
    std::uintptr_t end = 0;
    std::uint32_t protection = 0;
    Require(guarded(base + 2 * Unit, &end, &protection) && end == base + 4 * Unit && protection == PAGE_READWRITE, "page guards: the guarded run is wrong");
    Require(!guarded(base, &end) && end == base + 2 * Unit, "page guards: the run before the guard is wrong");
    // The driver's classification sees the logical protection (its first query of this block).
    Require(GuestMemory::DescribeCommitted(base, BlockBytes, true).whole, "page guards: guarded pages do not count as committed and writable");

    // A read in unit 2: the handler sees it, the byte is the guest's, the touched page opens and
    // the rest of the unit stays guarded.
    Require(readByte(base + 2 * Unit + 100) == 0x5a, "page guards: a guarded byte read wrong");
    Require(touches.size() == 1 && touches[0].address == base + 2 * Unit + 100 && !touches[0].write, "page guards: the read did not reach the handler");
    Require(!guarded(base + 2 * Unit) && guarded(base + 3 * Unit - 16384), "page guards: the read opened the wrong pages");
    // A write in unit 3, the handler opening the whole unit: the store lands.
    openUnit = true;
    writeByte(base + 3 * Unit + 20000, 0x77);
    Require(touches.size() == 2 && touches[1].address == base + 3 * Unit + 20000 && touches[1].write, "page guards: the write did not reach the handler");
    Require(readByte(base + 3 * Unit + 20000) == 0x77, "page guards: the guarded write did not land");
    Require(!guarded(base + 3 * Unit) && !guarded(base + 4 * Unit - 16384), "page guards: the write's unit did not open");
    Require(touches.size() == 2, "page guards: an open page reached the handler");

    // A protection change over a guarded unit: still guarded, at the new logical protection; the
    // read opens it read-only.
    Require(GuestArena::GuestArenaGuard_nid_postfix(reinterpret_cast<void*>(base + 5 * Unit), Unit), "page guards: unit 5 was refused");
    GuestArena::GuestArenaCommit_nid_postfix(reinterpret_cast<void*>(base + 5 * Unit), Unit, PAGE_READONLY, Unit);
    Require(guarded(base + 5 * Unit, &end, &protection) && protection == PAGE_READONLY, "page guards: a protection change lifted the guard");
    Require(readByte(base + 5 * Unit + 8) == 0x5a && touches.size() == 3, "page guards: the read-only unit read wrong");
    MEMORY_BASIC_INFORMATION memory{};
    VirtualQuery(reinterpret_cast<const void*>(base + 5 * Unit), &memory, sizeof(memory));
    Require(memory.Protect == PAGE_READONLY, "page guards: the opened unit is not read-only");
    GuestArena::GuestArenaCommit_nid_postfix(reinterpret_cast<void*>(base + 5 * Unit), Unit, PAGE_READWRITE, Unit);

    // A host read over units 2 and 3 (WriteFile from them, say): announced to the handler (no
    // instruction) before it reads, once per unit while the handler opens whole units (the pages it
    // opened with the first are not handed to it again), and open after.
    Require(GuestArena::GuestArenaGuard_nid_postfix(reinterpret_cast<void*>(base + 2 * Unit), 2 * Unit), "page guards: units 2 and 3 were refused again");
    touches.clear();
    GuestArena::GuestArenaOpenForHostRead_nid_postfix(reinterpret_cast<const void*>(base + 2 * Unit + 100), 2 * Unit - 200);
    Require(touches.size() == 2 && !touches[0].write && touches[0].address == base + 2 * Unit + 100 && !touches[1].write && touches[1].address == base + 3 * Unit, "page guards: the host read was not announced once per unit");
    Require(!guarded(base + 2 * Unit) && !guarded(base + 4 * Unit - 16384) && readByte(base + 3 * Unit + 30000) == 0x5a && touches.size() == 2, "page guards: the host read left a page guarded");
    // Memory outside the arena has no guards: nothing is announced.
    GuestArena::GuestArenaOpenForHostRead_nid_postfix(outside.data(), outside.size());
    Require(touches.size() == 2, "page guards: a host read outside the arena reached the handler");

    // Unguard opens; a reset drops what is left.
    Require(GuestArena::GuestArenaGuard_nid_postfix(reinterpret_cast<void*>(base + 6 * Unit), 2 * Unit), "page guards: units 6 and 7 were refused");
    GuestArena::GuestArenaUnguard_nid_postfix(reinterpret_cast<void*>(base + 6 * Unit), Unit);
    Require(!guarded(base + 6 * Unit) && guarded(base + 7 * Unit), "page guards: unguard opened the wrong unit");
    GuestArena::GuestArenaSetGuardHandler_nid_postfix(nullptr);
    GuestArena::GuestArenaReset_nid_postfix(block, BlockBytes);
    Require(!guarded(base + 7 * Unit), "page guards: a reset kept a guard");
    GuestArena::GuestArenaRelease_nid_postfix(block, BlockBytes);
    std::puts("page guard tests passed");
    return 0;
}

// Shared views (the direct memory a title maps): a guarded physical page is inaccessible on every
// alias, a new alias included, and opens on all of them; write tracking arms it without lifting the
// guard, and it opens armed; a host write announced by HostWrite reaches the handler and opens it
// first; a protection change made outside the arena is undone by GuestArenaReguard.
int SharedViewGuardTests() {
    constexpr std::size_t Bytes = 2 * Unit;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(Bytes), nullptr);
    Require(section != nullptr, "page guards: cannot create a section");
    void* first = GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Unit);
    void* second = GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Unit);
    GuestArena::GuestArenaMap_nid_postfix(first, Bytes, section, 0, PAGE_READWRITE);
    GuestArena::GuestArenaMap_nid_postfix(second, Bytes, section, 0, PAGE_READWRITE);
    const auto a = reinterpret_cast<std::uintptr_t>(first);
    const auto b = reinterpret_cast<std::uintptr_t>(second);
    std::memset(first, 0x3c, Bytes);
    Require(readByte(b + 5) == 0x3c, "page guards: the aliases do not share memory");
    touches.clear();
    openUnit = true;
    GuestArena::GuestArenaSetGuardHandler_nid_postfix(&recordTouch);

    // Guarding unit 0 through the first alias guards the second's.
    Require(GuestArena::GuestArenaGuard_nid_postfix(first, Unit), "page guards: a view was refused");
    Require(guarded(a) && guarded(b) && !guarded(a + Unit) && !guarded(b + Unit), "page guards: the aliases' guard states are wrong");
    MEMORY_BASIC_INFORMATION memory{};
    VirtualQuery(second, &memory, sizeof(memory));
    Require(memory.Protect == PAGE_NOACCESS, "page guards: the second alias is accessible");
    Require(GuestMemory::DescribeCommitted(b, Unit, true).whole, "page guards: a guarded view does not count as committed and writable");
    // A read through the second alias reaches the handler and opens both.
    Require(readByte(b + 100) == 0x3c && touches.size() == 1 && touches[0].address == b + 100 && !touches[0].write, "page guards: the alias read was not handled");
    Require(!guarded(a) && !guarded(b), "page guards: the alias read did not open both aliases");

    // A collect arms the views; a guarded page stays inaccessible and opens armed (read-only), so a
    // write after the guard's fault reopens it through the write tracking.
    Require(GuestArena::GuestArenaGuard_nid_postfix(first, Unit), "page guards: the view was refused again");
    std::array<void*, 64> pages{};
    std::size_t count = pages.size();
    GuestArena::GuestArenaCollectWrites_nid_postfix(a, Unit, pages.data(), &count, true);
    VirtualQuery(first, &memory, sizeof(memory));
    Require(memory.Protect == PAGE_NOACCESS, "page guards: arming lifted the guard");
    writeByte(a + 200, 0x44);
    Require(touches.size() == 2 && touches[1].write && readByte(b + 200) == 0x44, "page guards: the guarded write did not land through the arming");

    // A host write over a guarded page: announced to the handler (no instruction), then open.
    Require(GuestArena::GuestArenaGuard_nid_postfix(first, Unit), "page guards: the view was refused a third time");
    {
        const GuestArena::HostWrite host(reinterpret_cast<void*>(b + 300), 16);
        Require(host.Open(), "page guards: the host write was refused");
        std::memset(reinterpret_cast<void*>(b + 300), 0x55, 16);
    }
    Require(touches.size() == 3 && touches[2].write && touches[2].address == b + 300 && readByte(a + 300) == 0x55, "page guards: the host write was not announced to the handler");

    // A new alias of a guarded physical page starts guarded.
    Require(GuestArena::GuestArenaGuard_nid_postfix(first, Unit), "page guards: the view was refused a fourth time");
    void* third = GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Unit);
    GuestArena::GuestArenaMap_nid_postfix(third, Bytes, section, 0, PAGE_READWRITE);
    VirtualQuery(third, &memory, sizeof(memory));
    Require(guarded(reinterpret_cast<std::uintptr_t>(third)) && memory.Protect == PAGE_NOACCESS, "page guards: a new alias of a guarded page is accessible");

    // A protection change made outside the arena (libkernel's mprotect) lifts the guard of the page
    // it changes; GuestArenaReguard puts it back.
    DWORD previous = 0;
    VirtualProtect(second, 16384, PAGE_READWRITE, &previous);
    GuestArena::GuestArenaReguard_nid_postfix(second, 16384, PAGE_READWRITE);
    VirtualQuery(second, &memory, sizeof(memory));
    Require(memory.Protect == PAGE_NOACCESS, "page guards: reguard left the page open");

    GuestArena::GuestArenaUnguard_nid_postfix(first, Unit);
    GuestArena::GuestArenaSetGuardHandler_nid_postfix(nullptr);
    for (void* view : {first, second, third}) {
        GuestArena::GuestArenaReset_nid_postfix(view, Bytes);
        GuestArena::GuestArenaRelease_nid_postfix(view, Bytes);
    }
    CloseHandle(section);
    std::puts("shared view guard tests passed");
    return 0;
}
#endif

}

int main() {
#ifdef _WIN32
    try {
        if (!GuestArena::GuestArenaAvailable_nid_postfix()) {
            std::puts("skipped, no guest arena");
            return VulkanTestSkipped;
        }
        const int result = PageGuardTests();
        return result != 0 ? result : SharedViewGuardTests();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#else
    std::puts("skipped, page guards are Windows only");
    return VulkanTestSkipped;
#endif
}
