#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTARENA_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTARENA_HPP

#include <cstddef>
#include <cstdint>

// Guest virtual memory is placed inside one reserved arena below the PS5 application map limit
// absolute address and breaks on host addresses outside that range.
namespace GuestArena {

#ifndef _WIN32
using SharedBackingResolver = bool (*)(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset);
#endif

extern "C" {

bool GuestArenaAvailable_nid_postfix();
bool GuestArenaContains_nid_postfix(const void* pointer, std::size_t bytes);
void* GuestArenaAllocate_nid_postfix(std::size_t bytes, std::size_t alignment);
void* GuestArenaAllocateAtOrAbove_nid_postfix(std::uintptr_t hint, std::size_t bytes, std::size_t alignment);
void GuestArenaMarkUsed_nid_postfix(const void* pointer, std::size_t bytes);
void GuestArenaRelease_nid_postfix(const void* pointer, std::size_t bytes);
// The reserved range, and whether it was reserved with page write watching (Windows MEM_WRITE_WATCH).
void GuestArenaRange_nid_postfix(std::uintptr_t* base, std::size_t* bytes);
bool GuestArenaWriteWatched_nid_postfix();
#ifdef _WIN32
void GuestArenaSetProtection_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint32_t protection);
bool GuestArenaHandleWrite_nid_postfix(std::uintptr_t address);
void GuestArenaPinWritable_nid_postfix(const void* pointer, std::size_t bytes);
void GuestArenaUnpinWritable_nid_postfix(const void* pointer, std::size_t bytes);
bool GuestArenaProtection_nid_postfix(std::uintptr_t address, std::uint32_t* protection);
bool GuestArenaCollectWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
bool GuestArenaHostRegionOverlaps_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestArenaCommit_nid_postfix(void* pointer, std::size_t bytes, std::uint32_t protection, std::size_t granule);
void GuestArenaReset_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaMap_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection);
void* GuestArenaMapAlias_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestArenaUnmapAlias_nid_postfix(void* alias);
// Guards: guest pages made inaccessible to the host CPU while the guest's bytes there are not current
// (a device copy holds newer ones, see the AGC driver's buffer shadows): committed private memory,
// or shared views, whose physical page is guarded on every alias (a new alias included). A guarded
// page keeps a logical protection (and a view its write arming): GuestArenaProtection reports it, a
// protection change keeps the guard, a reset drops it. A host access to a guarded page calls the
// handler (the address, whether it writes, the faulting instruction; 0 for a host write announced
// by HostWrite) before the page opens, and the access then runs again. GuestArenaGuard takes 16 KiB
// aligned ranges and guards nothing (false) when a page is neither committed private memory nor a
// view with a readable protection. GuestArenaGuardRun tells whether `address`'s page is guarded
// and where the run of pages sharing its guard state and logical protection ends (at most `limit`).
typedef void (*GuestArenaGuardHandler)(std::uintptr_t address, bool write, std::uintptr_t instruction);
void GuestArenaSetGuardHandler_nid_postfix(GuestArenaGuardHandler handler);
bool GuestArenaGuard_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaUnguard_nid_postfix(void* pointer, std::size_t bytes);
bool GuestArenaGuardRun_nid_postfix(std::uintptr_t address, std::uintptr_t limit, std::uintptr_t* end, std::uint32_t* protection);
bool GuestArenaHandleGuard_nid_postfix(std::uintptr_t address, bool write, std::uintptr_t instruction);
// After a protection change made outside the arena over a range (libkernel's mprotect): its guards
// hold again, a private page's at `protection` as its logical one.
void GuestArenaReguard_nid_postfix(void* pointer, std::size_t bytes, std::uint32_t protection);
#else
void GuestArenaSetSharedBacking_nid_postfix(SharedBackingResolver resolver);
bool GuestArenaSharedBacking_nid_postfix(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset);
#endif
bool GuestArenaBeginHostWrite_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaEndHostWrite_nid_postfix(void* pointer, std::size_t bytes);

}

class HostWrite {
public:
    HostWrite(void* pointer, std::size_t bytes) : pointer(pointer), bytes(bytes), open(GuestArenaBeginHostWrite_nid_postfix(pointer, bytes)) {}
    ~HostWrite() {
        if (open) GuestArenaEndHostWrite_nid_postfix(pointer, bytes);
    }
    HostWrite(const HostWrite&) = delete;
    HostWrite& operator=(const HostWrite&) = delete;
    bool Open() const { return open; }

private:
    void* pointer;
    std::size_t bytes;
    bool open;
};

}

#endif
