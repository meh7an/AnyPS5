#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP

#ifdef _WIN32
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <map>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <system_error>

namespace GuestArena {

class WindowsMappings {
public:
    static WindowsMappings& Get() {
        static WindowsMappings mappings;
        return mappings;
    }

    void* Reserve(void* address, std::size_t bytes) {
        return allocate(GetCurrentProcess(), address, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
    }

    void Commit(void* address, std::size_t bytes, DWORD protection, std::size_t granule, bool watched) {
        std::lock_guard lock(mutex);
        const auto end = reinterpret_cast<std::uintptr_t>(address) + bytes;
        for (auto cursor = reinterpret_cast<std::uintptr_t>(address); cursor < end;) {
            const auto memory = query(cursor);
            const auto stop = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
            if (memory.State == MEM_RESERVE) {
                const auto limit = std::min(end, cursor + granule);
                auto placeholderEnd = stop;
                while (placeholderEnd < limit) {
                    const auto next = query(placeholderEnd);
                    if (next.State != MEM_RESERVE) break;
                    placeholderEnd = reinterpret_cast<std::uintptr_t>(next.BaseAddress) + next.RegionSize;
                }
                const auto size = std::min(limit, placeholderEnd) - cursor;
                reset(cursor, size);
                const DWORD flags = MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | (watched ? MEM_WRITE_WATCH : 0);
                if (!allocate(GetCurrentProcess(), reinterpret_cast<void*>(cursor), size, flags, protection, nullptr, 0)) fail("replace guest placeholder with private memory");
                cursor += size;
            } else {
                if (memory.State != MEM_COMMIT) throw std::runtime_error("guest memory is not committed");
                const auto mapped = views.find(cursor & ~(pageBytes - 1));
                if (mapped != views.end()) {
                    mapped->second.protection = protection;
                    mapped->second.armed = false;
                    invalidate(*mapped->second.page);
                }
                DWORD previous;
                if (!VirtualProtect(reinterpret_cast<void*>(cursor), stop - cursor, protection, &previous)) fail("protect guest memory");
                // A guarded page takes the new protection as its logical one and stays guarded.
                reguard(cursor, stop, protection);
                cursor = stop;
            }
        }
    }

    void Reset(void* address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        reset(reinterpret_cast<std::uintptr_t>(address), bytes);
    }

    void Map(void* address, std::size_t bytes, HANDLE section, std::uint64_t offset, DWORD protection) {
        std::lock_guard lock(mutex);
        auto cursor = reinterpret_cast<std::uintptr_t>(address);
        reset(cursor, bytes);
        HANDLE duplicate = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), section, GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS)) fail("keep shared guest section");
        const auto owned = std::make_shared<Section>(duplicate);
        for (std::size_t done = 0; done < bytes; done += pageBytes) {
            split(cursor + done, pageBytes);
            void* page = reinterpret_cast<void*>(cursor + done);
            if (!map(section, GetCurrentProcess(), page, offset + done, pageBytes, MEM_REPLACE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, nullptr, 0)) fail("map shared guest page");
            DWORD previous;
            if (!VirtualProtect(page, pageBytes, protection, &previous)) fail("protect shared guest page");
            const auto key = std::make_pair(reinterpret_cast<std::uintptr_t>(section), offset + done);
            auto shared = physical[key].lock();
            if (!shared) {
                shared = std::make_shared<SharedPage>();
                physical[key] = shared;
            }
            const auto base = cursor + done;
            shared->aliases.push_back(base);
            views.emplace(base, View{shared, protection, 0, false, owned, offset + done, 0});
            invalidate(*shared);
            // A new alias of a guarded physical page is guarded with the others.
            if (shared->guarded && !VirtualProtect(page, pageBytes, PAGE_NOACCESS, &previous)) fail("guard a new view of guarded guest memory");
        }
    }

    void SetProtection(std::uintptr_t address, std::size_t bytes, DWORD protection) {
        std::lock_guard lock(mutex);
        for (auto it = views.lower_bound(address); it != views.end() && it->first < address + bytes; ++it) {
            it->second.protection = protection;
            it->second.armed = false;
            invalidate(*it->second.page);
        }
    }

    void Pin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            auto& page = *it->second.page;
            ++page.pins;
            for (const auto alias : page.aliases) {
                auto& view = views.at(alias);
                if (!view.armed) continue;
                DWORD previous;
                if (!VirtualProtect(reinterpret_cast<void*>(alias), pageBytes, view.protection, &previous)) fail("pin shared guest page writable");
                view.armed = false;
            }
            invalidate(page);
        }
    }

    void Unpin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            auto& page = *it->second.page;
            if (page.pins != 0) --page.pins;
            invalidate(page);
        }
    }

    bool HandleWrite(std::uintptr_t address) {
        std::lock_guard lock(mutex);
        const auto base = address & ~(pageBytes - 1);
        const auto found = views.find(base);
        // A guarded page's faults are HandleGuard's.
        if (found == views.end() || !writable(found->second.protection) || found->second.page->guarded) return false;
        auto& view = found->second;
        invalidate(*view.page);
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(base), pageBytes, view.protection, &previous)) fail("resume shared memory write");
        view.armed = false;
        return true;
    }

    bool BeginHostWrite(std::uintptr_t address, std::size_t bytes) {
        // A host write (file I/O into guest memory, say) cannot take a guard's fault: guarded pages
        // become current and open first.
        openForHostWrite(address, bytes);
        std::lock_guard lock(mutex);
        const auto first = views.lower_bound(address & ~(pageBytes - 1));
        const auto end = address + bytes;
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            if (!writable(it->second.protection)) return false;
        }
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            auto& view = it->second;
            ++view.hostWrites;
            invalidate(*view.page);
            if (!view.armed) continue;
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(it->first), pageBytes, view.protection, &previous)) fail("open shared memory to a host write");
            view.armed = false;
        }
        return true;
    }

    void EndHostWrite(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto end = address + bytes;
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
            --it->second.hostWrites;
            invalidate(*it->second.page);
        }
    }

    void* MapAlias(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto refuse = [&](const char* reason) {
            char text[192];
            std::snprintf(text, sizeof(text), "read-write alias of shared guest memory 0x%llx+0x%llx: %s", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), reason);
            return std::runtime_error(text);
        };
        if (address % pageBytes != 0 || bytes % pageBytes != 0 || bytes == 0) throw refuse("the range is not made of whole shared pages");
        auto view = views.find(address);
        if (view == views.end()) throw refuse("the range does not start at a shared view");
        const auto section = view->second.section;
        const auto offset = view->second.offset;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        for (std::size_t done = 0; done < bytes; done += pageBytes, ++view) {
            if (view == views.end() || view->first != address + done || view->second.offset != offset + done) throw refuse("the range is not one contiguous run of views of a section");
            if (view->second.section != section && !sameSection(view->second.section->handle, section->handle)) throw refuse("the range spans several sections");
        }
        const auto lead = offset % system.dwAllocationGranularity;
        void* alias = map(section->handle, GetCurrentProcess(), nullptr, offset - lead, lead + bytes, 0, PAGE_READWRITE, nullptr, 0);
        if (alias == nullptr) {
            char text[160];
            std::snprintf(text, sizeof(text), "MapViewOfFile3 of a read-write alias of shared guest memory 0x%llx+0x%llx", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes));
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), text);
        }
        return static_cast<char*>(alias) + lead;
    }

    void UnmapAlias(void* alias) {
        if (alias == nullptr) return;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        const auto base = reinterpret_cast<std::uintptr_t>(alias) & ~(static_cast<std::uintptr_t>(system.dwAllocationGranularity) - 1);
        if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(base), 0)) fail("unmap shared guest alias");
    }

    bool Protection(std::uintptr_t address, std::uint32_t* protection) {
        std::lock_guard lock(mutex);
        if (const auto guarded = guards.find(address & ~(pageBytes - 1)); guarded != guards.end()) {
            *protection = guarded->second;
            return true;
        }
        const auto found = views.find(address & ~(pageBytes - 1));
        if (found == views.end()) return false;
        *protection = found->second.protection;
        return true;
    }

    // Guards (GuestArena.hpp): pages made PAGE_NOACCESS while the guest's bytes there are not
    // current. A private page keeps its logical protection in `guards`; a shared view's physical
    // page is guarded on every alias, its views keeping their logical protection and write arming.
    // False, guarding nothing, when a page of the range is neither committed private memory nor a
    // view with a readable protection.
    bool Guard(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        if (address % pageBytes != 0 || bytes % pageBytes != 0 || bytes == 0) return false;
        const auto end = address + bytes;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> runs;
        std::vector<DWORD> protections;
        std::vector<SharedPage*> shared;
        for (auto cursor = address; cursor < end;) {
            if (const auto view = views.find(cursor); view != views.end()) {
                if (!readable(view->second.protection)) return false;
                shared.push_back(view->second.page.get());
                cursor += pageBytes;
                continue;
            }
            auto next = guards.lower_bound(cursor);
            if (next != guards.end() && next->first == cursor) {
                cursor += pageBytes;
                continue;
            }
            const auto memory = query(cursor);
            if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE || !readable(memory.Protect)) return false;
            auto stop = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
            if (next != guards.end() && next->first < stop) stop = next->first;
            runs.emplace_back(cursor, stop);
            protections.push_back(memory.Protect);
            cursor = stop;
        }
        for (std::size_t i = 0; i < runs.size(); ++i) {
            const auto [begin, stop] = runs[i];
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(begin), stop - begin, PAGE_NOACCESS, &previous)) fail("guard guest memory");
            for (auto page = begin; page < stop; page += pageBytes) guards.emplace(page, protections[i]);
        }
        for (auto* page : shared) guardShared(*page);
        return true;
    }

    void Unguard(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        unguard(address, address + bytes);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) openShared(*it->second.page);
    }

    // After a protection change made outside the arena over [address, address + bytes) (libkernel's
    // mprotect): its guards hold again, private ones at `protection` as their logical one.
    void Reguard(std::uintptr_t address, std::size_t bytes, DWORD protection) {
        std::lock_guard lock(mutex);
        reguard(address, address + bytes, protection);
    }

    // Whether the page holding `address` is guarded, and the end of the run of pages from it that
    // share its guard state and logical protection (at most `limit`).
    bool GuardRun(std::uintptr_t address, std::uintptr_t limit, std::uintptr_t* end, std::uint32_t* protection) {
        std::lock_guard lock(mutex);
        const auto page = address & ~(pageBytes - 1);
        if (const auto view = views.find(page); view != views.end()) {
            *end = std::min(page + pageBytes, limit);
            *protection = view->second.protection;
            return view->second.page->guarded;
        }
        auto it = guards.lower_bound(page);
        if (it == guards.end() || it->first != page) {
            *end = it != guards.end() ? std::min(it->first, limit) : limit;
            return false;
        }
        const auto logical = it->second;
        auto runEnd = page + pageBytes;
        for (++it; it != guards.end() && it->first == runEnd && runEnd < limit && it->second == logical; ++it) runEnd += pageBytes;
        *end = std::min(runEnd, limit);
        *protection = logical;
        return true;
    }

    // A host access fault at `address`: whether it was a guard's. The handler, called outside the
    // lock, may make the bytes current and open the page; a page it leaves guarded is opened here.
    // A fault on a page another thread opened meanwhile is a guard's too (the access runs again).
    bool HandleGuard(std::uintptr_t address, bool write, std::uintptr_t instruction) {
        const auto page = address & ~(pageBytes - 1);
        {
            std::lock_guard lock(mutex);
            if (!guardedPage(page)) {
                // An access the page allows now: a guard opened since the fault. A write to an armed
                // view is HandleWrite's.
                MEMORY_BASIC_INFORMATION memory{};
                if (VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) != sizeof(memory) || memory.State != MEM_COMMIT) return false;
                if (memory.Type != MEM_PRIVATE && !views.contains(page)) return false;
                return write ? writable(memory.Protect & 0xffu) : readable(memory.Protect);
            }
        }
        if (const auto handler = guardHandler.load(std::memory_order_acquire)) handler(address, write, instruction);
        std::lock_guard lock(mutex);
        openPage(page);
        return true;
    }

    // Before a host write into [address, address + bytes) (BeginHostWrite): the handler makes each
    // guarded page's bytes current, and the page opens.
    void openForHostWrite(std::uintptr_t address, std::size_t bytes) {
        std::vector<std::uintptr_t> guarded;
        {
            std::lock_guard lock(mutex);
            for (auto page = address & ~(pageBytes - 1); page < address + bytes; page += pageBytes) {
                if (guardedPage(page)) guarded.push_back(page);
            }
        }
        const auto handler = guardHandler.load(std::memory_order_acquire);
        for (const auto page : guarded) {
            if (handler != nullptr) handler(std::max(page, address), true, 0);
            std::lock_guard lock(mutex);
            openPage(page);
        }
    }

    void SetGuardHandler(void (*handler)(std::uintptr_t, bool, std::uintptr_t)) {
        guardHandler.store(handler, std::memory_order_release);
    }

    bool Collect(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear) {
        std::lock_guard lock(mutex);
        const auto capacity = *count;
        *count = 0;
        const auto end = address + bytes;
        for (auto cursor = address; cursor < end;) {
            const auto nextClean = cleanRanges.upper_bound(cursor);
            if (nextClean != cleanRanges.begin()) {
                const auto clean = std::prev(nextClean);
                if (cursor < clean->second) {
                    cursor = std::min(end, clean->second);
                    continue;
                }
            }
            const auto base = cursor & ~(pageBytes - 1);
            const auto found = views.find(base);
            if (found != views.end()) {
                auto& view = found->second;
                const auto stop = std::min(end, base + pageBytes);
                if (view.protection == PAGE_NOACCESS) return false;
                const bool pinned = view.page->pins != 0;
                if (pinned || view.seen != view.page->generation) {
                    const auto needed = (stop - cursor + 4095) / 4096;
                    if (needed > capacity - *count) {
                        for (auto at = cursor; *count < capacity; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                        return true;
                    }
                    for (auto at = cursor; at < stop; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                }
                if (clear && !pinned) {
                    for (const auto alias : view.page->aliases) {
                        auto& other = views.at(alias);
                        if (!writable(other.protection) || other.armed || other.hostWrites != 0) continue;
                        other.armed = true;
                        // A guarded page stays inaccessible; it opens armed.
                        if (other.page->guarded) continue;
                        DWORD previous;
                        if (!VirtualProtect(reinterpret_cast<void*>(alias), pageBytes, actualProtection(other), &previous)) fail("arm shared memory write tracking");
                    }
                    view.seen = view.page->generation;
                    rememberClean(base, base + pageBytes);
                }
                cursor = stop;
            } else {
                const auto memory = query(cursor);
                if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE) return false;
                const auto stop = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                ULONG_PTR available = capacity - *count;
                if (available == 0) return true;
                DWORD granularity = 0;
                if (GetWriteWatch(clear ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(cursor), stop - cursor, pages + *count, &available, &granularity) != 0) fail("collect private guest writes");
                *count += available;
                if (*count == capacity) return true;
                cursor = stop;
            }
        }
        return true;
    }

private:
    static constexpr std::size_t pageBytes = 0x4000;
    struct SharedPage {
        std::uint64_t generation = 1;
        std::vector<std::uintptr_t> aliases;
        // Guarded (Guard): every alias is PAGE_NOACCESS, whatever its arming.
        bool guarded = false;
        std::uint32_t pins = 0;
    };
    struct Section {
        HANDLE handle;
        explicit Section(HANDLE handle) : handle(handle) {}
        Section(const Section&) = delete;
        Section& operator=(const Section&) = delete;
        ~Section() { CloseHandle(handle); }
    };
    struct View {
        std::shared_ptr<SharedPage> page;
        DWORD protection;
        std::uint64_t seen;
        bool armed;
        std::shared_ptr<Section> section;
        std::uint64_t offset;
        std::uint32_t hostWrites;
    };
    void forgetClean(std::uintptr_t start, std::uintptr_t end) {
        auto it = cleanRanges.lower_bound(start);
        if (it != cleanRanges.begin() && std::prev(it)->second > start) --it;
        while (it != cleanRanges.end() && it->first < end) {
            const auto first = it->first;
            const auto last = it->second;
            it = cleanRanges.erase(it);
            if (first < start) cleanRanges.emplace(first, start);
            if (last > end) it = cleanRanges.emplace(end, last).first;
        }
    }

    void rememberClean(std::uintptr_t start, std::uintptr_t end) {
        auto it = cleanRanges.lower_bound(start);
        if (it != cleanRanges.begin() && std::prev(it)->second >= start) --it;
        while (it != cleanRanges.end() && it->first <= end) {
            start = std::min(start, it->first);
            end = std::max(end, it->second);
            it = cleanRanges.erase(it);
        }
        cleanRanges.emplace(start, end);
    }

    void invalidate(SharedPage& page) {
        ++page.generation;
        for (const auto alias : page.aliases) forgetClean(alias, alias + pageBytes);
    }

    // A view's protection as set on its page: none while its physical page is guarded, read-only
    // while write tracking is armed, its logical protection otherwise.
    static DWORD actualProtection(const View& view) {
        if (view.page->guarded) return PAGE_NOACCESS;
        if (view.armed) return view.protection == PAGE_EXECUTE_READWRITE ? PAGE_EXECUTE_READ : PAGE_READONLY;
        return view.protection;
    }

    // A shared page guarded or opened on every alias.
    void guardShared(SharedPage& page) {
        if (page.guarded) return;
        page.guarded = true;
        for (const auto alias : page.aliases) {
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(alias), pageBytes, PAGE_NOACCESS, &previous)) fail("guard shared guest memory");
        }
    }

    void openShared(SharedPage& page) {
        if (!page.guarded) return;
        page.guarded = false;
        for (const auto alias : page.aliases) {
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(alias), pageBytes, actualProtection(views.at(alias)), &previous)) fail("open shared guest memory");
        }
    }

    bool guardedPage(std::uintptr_t page) const {
        if (const auto view = views.find(page); view != views.end()) return view->second.page->guarded;
        return guards.contains(page);
    }

    // Opens the page at `page`, private or shared.
    void openPage(std::uintptr_t page) {
        if (const auto view = views.find(page); view != views.end()) openShared(*view->second.page);
        else unguard(page, page + pageBytes);
    }

    // Opens the guarded private pages of [begin, end) at their logical protection, a run of equal
    // ones per call.
    void unguard(std::uintptr_t begin, std::uintptr_t end) {
        auto it = guards.lower_bound(begin & ~(pageBytes - 1));
        while (it != guards.end() && it->first < end) {
            const auto runBegin = it->first;
            const auto protection = it->second;
            auto runEnd = runBegin + pageBytes;
            it = guards.erase(it);
            while (it != guards.end() && it->first == runEnd && runEnd < end && it->second == protection) {
                runEnd += pageBytes;
                it = guards.erase(it);
            }
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(runBegin), runEnd - runBegin, protection, &previous)) fail("open guarded guest memory");
        }
    }

    // After a protection change over [begin, end): its guarded private pages take `protection` as
    // their logical one, and they and its views of guarded shared pages are guarded again (a view
    // keeps the logical protection the change gave it).
    void reguard(std::uintptr_t begin, std::uintptr_t end, DWORD protection) {
        for (auto it = guards.lower_bound(begin); it != guards.end() && it->first < end;) {
            const auto runBegin = it->first;
            auto runEnd = runBegin;
            for (; it != guards.end() && it->first == runEnd && runEnd < end; ++it, runEnd += pageBytes) it->second = protection;
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(runBegin), runEnd - runBegin, PAGE_NOACCESS, &previous)) fail("guard guest memory again");
        }
        for (auto it = views.lower_bound(begin & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
            if (!it->second.page->guarded) continue;
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(it->first), pageBytes, PAGE_NOACCESS, &previous)) fail("guard shared guest memory again");
        }
    }

    static bool readable(DWORD protection) {
        if ((protection & PAGE_GUARD) != 0) return false;
        switch (protection & 0xffu) {
            case PAGE_READONLY:
            case PAGE_READWRITE:
            case PAGE_WRITECOPY:
            case PAGE_EXECUTE_READ:
            case PAGE_EXECUTE_READWRITE:
            case PAGE_EXECUTE_WRITECOPY:
                return true;
            default:
                return false;
        }
    }

    bool sameSection(HANDLE first, HANDLE second) const {
        return compare != nullptr && compare(first, second);
    }

    static bool writable(DWORD protection) {
        return protection == PAGE_READWRITE || protection == PAGE_EXECUTE_READWRITE;
    }
    using AllocateFunction = PVOID (WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using MapFunction = PVOID (WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using UnmapFunction = BOOL (WINAPI*)(HANDLE, PVOID, ULONG);
    using CompareFunction = BOOL (WINAPI*)(HANDLE, HANDLE);

    WindowsMappings() {
        const auto module = GetModuleHandleW(L"KernelBase.dll");
        if (!module) fail("load Windows memory API");
        allocate = reinterpret_cast<AllocateFunction>(GetProcAddress(module, "VirtualAlloc2"));
        map = reinterpret_cast<MapFunction>(GetProcAddress(module, "MapViewOfFile3"));
        unmap = reinterpret_cast<UnmapFunction>(GetProcAddress(module, "UnmapViewOfFile2"));
        if (!allocate || !map || !unmap) throw std::runtime_error("Windows placeholder memory APIs are required");
        compare = reinterpret_cast<CompareFunction>(GetProcAddress(module, "CompareObjectHandles"));
    }

    [[noreturn]] static void fail(const char* operation) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
    }

    static MEMORY_BASIC_INFORMATION query(std::uintptr_t address) {
        MEMORY_BASIC_INFORMATION memory{};
        if (VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) != sizeof(memory)) fail("query guest memory");
        return memory;
    }

    static void split(std::uintptr_t address, std::size_t bytes) {
        auto memory = query(address);
        memory = query(reinterpret_cast<std::uintptr_t>(memory.AllocationBase));
        if (memory.State != MEM_RESERVE) throw std::runtime_error("guest mapping requires a placeholder");
        const auto base = reinterpret_cast<std::uintptr_t>(memory.BaseAddress);
        if (address != base) {
            if (!VirtualFree(reinterpret_cast<void*>(base), address - base, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder prefix");
            memory = query(address);
        }
        if (memory.RegionSize < bytes) throw std::runtime_error("guest placeholder is too small");
        if (memory.RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder suffix");
    }

    void reset(std::uintptr_t address, std::size_t bytes) {
        const auto end = address + bytes;
        forgetClean(address, end);
        // Released memory takes its guards with it.
        guards.erase(guards.lower_bound(address), guards.lower_bound(end));
        for (auto cursor = address; cursor < end;) {
            const auto memory = query(cursor);
            if (memory.State == MEM_RESERVE) {
                cursor = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                continue;
            }
            if (reinterpret_cast<std::uintptr_t>(memory.AllocationBase) != cursor) throw std::runtime_error("cannot release part of a host allocation");
            auto allocationEnd = cursor;
            do {
                const auto part = query(allocationEnd);
                if (part.AllocationBase != memory.AllocationBase) break;
                allocationEnd = reinterpret_cast<std::uintptr_t>(part.BaseAddress) + part.RegionSize;
            } while (allocationEnd < end);
            if (allocationEnd > end || query(allocationEnd).AllocationBase == memory.AllocationBase) throw std::runtime_error("guest release truncates a host allocation");
            if (memory.Type == MEM_MAPPED) {
                if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(cursor), MEM_PRESERVE_PLACEHOLDER)) fail("unmap shared guest page");
                const auto found = views.find(cursor);
                if (found != views.end()) {
                    std::erase(found->second.page->aliases, cursor);
                    views.erase(found);
                }
            } else if (memory.Type == MEM_PRIVATE) {
                if (!VirtualFree(reinterpret_cast<void*>(cursor), allocationEnd - cursor, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release private guest memory");
            } else {
                throw std::runtime_error("unsupported guest mapping type");
            }
            cursor = allocationEnd;
        }
        const auto last = query(reinterpret_cast<std::uintptr_t>(query(end - 1).AllocationBase));
        const auto lastBase = reinterpret_cast<std::uintptr_t>(last.BaseAddress);
        if (lastBase + last.RegionSize > end) split(lastBase, end - lastBase);
        const auto first = query(address);
        split(address, std::min(bytes, reinterpret_cast<std::uintptr_t>(first.BaseAddress) + first.RegionSize - address));
        if (query(address).RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) fail("coalesce guest placeholders");
    }

    std::map<std::uintptr_t, std::uintptr_t> cleanRanges;
    std::map<std::uintptr_t, View> views;
    // Guarded private pages and their logical protection (Guard), and the fault handler.
    std::map<std::uintptr_t, DWORD> guards;
    std::atomic<void (*)(std::uintptr_t, bool, std::uintptr_t)> guardHandler{nullptr};
    std::map<std::pair<std::uintptr_t, std::uint64_t>, std::weak_ptr<SharedPage>> physical;
    std::mutex mutex;
    AllocateFunction allocate = nullptr;
    MapFunction map = nullptr;
    UnmapFunction unmap = nullptr;
    CompareFunction compare = nullptr;
};

}
#endif

#endif
