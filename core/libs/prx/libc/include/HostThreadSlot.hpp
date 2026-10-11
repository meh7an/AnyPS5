#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_HOSTTHREADSLOT_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_HOSTTHREADSLOT_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#if defined(_WIN32) && defined(__x86_64__) && defined(__GNUC__)
#define HOST_THREAD_SLOT_TEB 1
// As <windows.h> declares them (without bringing its macros into every includer).
extern "C" __declspec(dllimport) unsigned long __stdcall TlsAlloc(void);
extern "C" __declspec(dllimport) int __stdcall TlsSetValue(unsigned long index, void* value);
#endif

// A per-thread value of up to 8 bytes whose bits start out zero on every thread, for the hot paths:
// on x64 Windows a TLS slot of the thread environment block (TlsAlloc), read and written with one or
// two instructions, where MinGW emulates thread_local (__emutls_get_address, then
// pthread_getspecific and a winpthreads spin lock on every access). Each Tag names its own slot, in
// the module that uses it; the TLS index is never freed. Elsewhere, a thread_local.
template<typename TValue, typename TTag>
class HostThreadSlot {
    static_assert(std::is_trivially_copyable_v<TValue> && sizeof(TValue) <= sizeof(std::uintptr_t));

public:
    static TValue Get() noexcept {
        const auto bits = load();
        TValue value;
        std::memcpy(&value, &bits, sizeof(TValue));
        return value;
    }

    static void Set(TValue value) noexcept {
        std::uintptr_t bits = 0;
        std::memcpy(&bits, &value, sizeof(TValue));
        store(bits);
    }

private:
#ifdef HOST_THREAD_SLOT_TEB
    // The TEB's TlsSlots (the first 64 indexes) and its pointer to TlsExpansionSlots (the others,
    // allocated for a thread by its first TlsSetValue there).
    static constexpr unsigned long DirectSlots = 64;
    static constexpr std::uintptr_t DirectSlotsOffset = 0x1480;
    static constexpr std::uintptr_t ExpansionSlotsOffset = 0x1780;

    static unsigned long index() noexcept {
        static const unsigned long allocated = [] {
            const auto value = TlsAlloc();
            if (value == 0xffffffffu) std::abort();
            return value;
        }();
        return allocated;
    }

    static std::uintptr_t readTeb(std::uintptr_t offset) noexcept {
        std::uintptr_t value;
        __asm__ __volatile__("movq %%gs:(%1), %0" : "=r"(value) : "r"(offset));
        return value;
    }

    static std::uintptr_t load() noexcept {
        const auto slot = index();
        if (slot < DirectSlots) return readTeb(DirectSlotsOffset + slot * sizeof(void*));
        const auto* expansion = reinterpret_cast<const std::uintptr_t*>(readTeb(ExpansionSlotsOffset));
        return expansion != nullptr ? expansion[slot - DirectSlots] : 0;
    }

    static void store(std::uintptr_t bits) noexcept {
        const auto slot = index();
        if (slot < DirectSlots) {
            __asm__ __volatile__("movq %0, %%gs:(%1)" : : "r"(bits), "r"(DirectSlotsOffset + slot * sizeof(void*)) : "memory");
        } else if (!TlsSetValue(slot, reinterpret_cast<void*>(bits))) {
            std::abort();
        }
    }
#else
    static std::uintptr_t& bitsOfThread() noexcept {
        static thread_local std::uintptr_t bits = 0;
        return bits;
    }

    static std::uintptr_t load() noexcept { return bitsOfThread(); }
    static void store(std::uintptr_t bits) noexcept { bitsOfThread() = bits; }
#endif
};

#endif
