#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_THREADSCRATCH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_THREADSCRATCH_HPP

#include "prx/libc/include/HostThreadSlot.hpp"
#include "ThreadOwned.hpp"

namespace AgcDriver {

// The calling thread's object of type T for Tag: created on the thread's first use and destroyed
// when the thread exits. For per-draw scratch that its user clears and refills, so the capacity
// (and memory) of one draw is reused by the next instead of allocated anew. Reached through a thread
// slot (HostThreadSlot): thread_local is emulated on MinGW, a winpthreads lock per access. A user
// must not hold the object across a call that may use it again.
template<typename T, typename Tag>
T& ThreadScratch() {
    auto* object = HostThreadSlot<T*, Tag>::Get();
    if (object == nullptr) {
        ShaderRecompiler::ThreadOwned(object);
        HostThreadSlot<T*, Tag>::Set(object);
    }
    return *object;
}

// A container's ThreadScratch lent to one user for a scope: swapped out empty and swapped back
// cleared, so a nested user on the thread finds it empty instead of the outer user's contents.
template<typename T, typename Tag>
class ThreadScratchLease {
public:
    ThreadScratchLease() : slot(ThreadScratch<T, Tag>()) {
        value.swap(slot);
        value.clear();
    }

    ~ThreadScratchLease() {
        value.clear();
        value.swap(slot);
    }

    ThreadScratchLease(const ThreadScratchLease&) = delete;
    ThreadScratchLease& operator=(const ThreadScratchLease&) = delete;

    T value;

private:
    T& slot;
};

}

#endif
