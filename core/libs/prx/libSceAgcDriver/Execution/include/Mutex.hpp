#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_MUTEX_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_MUTEX_HPP

#include <atomic>
#include <cstdint>

namespace AgcDriver {

// Sleeps while `word` holds `value`; it may also return early, so a caller checks again.
inline void WaitWhile(const std::atomic<std::uint32_t>& word, std::uint32_t value) noexcept {
    word.wait(value, std::memory_order_relaxed);
}

// Wakes one, or every, thread sleeping on `word` in WaitWhile.
inline void WakeOne(std::atomic<std::uint32_t>& word) noexcept {
    word.notify_one();
}

inline void WakeAll(std::atomic<std::uint32_t>& word) noexcept {
    word.notify_all();
}

// The driver's mutex, in place of std::mutex, which goes through winpthreads on MinGW: its lock and
// unlock calls were a large share of the per-draw path. Taking or releasing an uncontended Mutex is
// one atomic instruction; a contended lock spins briefly, then sleeps on the lock word (std::atomic
// wait). Like std::mutex it is neither recursive nor fair. All of it is inline: other modules build
// the driver's headers in, and a module's exported C++ functions are renamed to NIDs.
// std::condition_variable waits with std::mutex only, so the mutexes it waits with stay std::mutex.
class Mutex {
public:
    constexpr Mutex() noexcept = default;
    Mutex(const Mutex&) = delete;
    Mutex& operator=(const Mutex&) = delete;

    void lock() noexcept {
        if (!try_lock()) lockContended();
    }

    bool try_lock() noexcept {
        std::uint32_t expected = Unlocked;
        return state.compare_exchange_strong(expected, Locked, std::memory_order_acquire, std::memory_order_relaxed);
    }

    void unlock() noexcept {
        if (state.exchange(Unlocked, std::memory_order_release) == Contended) WakeOne(state);
    }

private:
    static constexpr std::uint32_t Unlocked = 0;
    static constexpr std::uint32_t Locked = 1;
    // Locked, and a thread may sleep on the word: the unlock wakes one.
    static constexpr std::uint32_t Contended = 2;

    [[gnu::noinline]] void lockContended() noexcept {
        // Most holds are short: spin a little before sleeping.
        for (int spin = 0; spin < 100; ++spin) {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
            if (state.load(std::memory_order_relaxed) == Unlocked && try_lock()) return;
        }
        // From here on the word reads Contended while the mutex is held, so each unlock wakes a
        // sleeper. Swapping Contended in for Unlocked takes the mutex; it stays marked, as others may
        // sleep on it.
        while (state.exchange(Contended, std::memory_order_acquire) != Unlocked) WaitWhile(state, Contended);
    }

    std::atomic<std::uint32_t> state{Unlocked};
};

// A token that names the calling thread, unique among live threads: on Windows, the address of the
// thread environment block (one instruction, where a thread_local is emulated through a winpthreads
// lock); elsewhere a thread local's address.
inline const void* CurrentThreadToken() noexcept {
#if defined(_WIN32) && defined(__x86_64__) && defined(__GNUC__)
    const void* block;
    __asm__("movq %%gs:0x30, %0" : "=r"(block));
    return block;
#else
    thread_local char token;
    return &token;
#endif
}

}

#endif
