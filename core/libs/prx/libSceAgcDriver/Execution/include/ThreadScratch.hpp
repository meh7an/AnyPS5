#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_THREADSCRATCH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_THREADSCRATCH_HPP

#include "prx/libc/include/HostThreadSlot.hpp"
#include "ThreadOwned.hpp"

namespace AgcDriver {

template<typename T, typename Tag>
T& ThreadScratch() {
    auto* object = HostThreadSlot<T*, Tag>::Get();
    if (object == nullptr) {
        ShaderRecompiler::ThreadOwned(object);
        HostThreadSlot<T*, Tag>::Set(object);
    }
    return *object;
}

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
