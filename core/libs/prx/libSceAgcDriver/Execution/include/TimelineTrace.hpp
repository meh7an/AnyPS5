#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_TIMELINETRACE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_TIMELINETRACE_HPP

#include "prx/libc/include/HostThreadSlot.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>
#include <x86intrin.h>

// A frame timeline of the queue worker and its draw thread, for finding the critical path of a frame
// (the sampler gives totals, not what each thread waited behind). APS5_TIMELINE=<frames> records that
// many frames of events once the file APS5_SAMPLE_TRIGGER names exists (the measuring script makes it
// when the page shows), then writes them to APS5_TIMELINE_OUT: each thread's events with their time
// stamp counter, and the counter's rate. Every event is a few instructions into a buffer of its own
// thread; nothing is recorded without the switch.
namespace AgcDriver::TimelineTrace {

enum class Event : std::uint8_t {
    // The worker: a packet outside the state fast path (argument: opcode), a wait for the draw
    // thread (argument: DrawThread::Wait), a submit between packets, a record handed over (argument:
    // 0 a draw without color targets, 1 a draw with them, 2 a ring record), a flip.
    PacketBegin, PacketEnd, WaitBegin, WaitEnd, SubmitBegin, SubmitEnd, Pushed, Flip,
    // The draw thread: a wait for a record, a record (argument as Pushed).
    IdleBegin, IdleEnd, RecordBegin, RecordEnd,
    // A submit's parts (Recorder::Submit: the queue's lock, vkQueueSubmit; SubmitRecorded's reap),
    // and the presenter's vkQueuePresentKHR under the queue's lock.
    QueueLockBegin, QueueLockEnd, QueueSubmitBegin, QueueSubmitEnd, ReapBegin, ReapEnd, PresentBegin, PresentEnd,
};

struct Entry {
    std::uint64_t tsc;
    std::uint32_t argument;
    std::uint8_t event;
    std::uint8_t pad[3];
};

struct Buffer {
    static constexpr std::size_t Capacity = std::size_t{1} << 22;
    std::unique_ptr<Entry[]> entries{new Entry[Capacity]};
    std::atomic<std::size_t> count{0};
    std::uint32_t thread = 0;
};

struct State {
    std::mutex mutex;
    std::vector<std::unique_ptr<Buffer>> buffers;
    // 0 before the trigger, 1 recording, 2 written.
    std::atomic<int> phase{0};
    std::uint32_t frames = 0;
    std::uint32_t flips = 0;
    std::uint64_t startTsc = 0;
    std::chrono::steady_clock::time_point startClock;
};

inline State& state() {
    static State instance;
    return instance;
}

inline std::uint32_t framesToRecord() {
    static const std::uint32_t frames = [] {
        const char* text = std::getenv("APS5_TIMELINE");
        return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 10)) : 0u;
    }();
    return frames;
}

inline bool Enabled() {
    return framesToRecord() != 0;
}

struct BufferTag;

inline Buffer* threadBuffer() {
    auto* buffer = HostThreadSlot<Buffer*, BufferTag>::Get();
    if (buffer != nullptr) return buffer;
    auto& shared = state();
    std::lock_guard lock(shared.mutex);
    shared.buffers.push_back(std::make_unique<Buffer>());
    buffer = shared.buffers.back().get();
    buffer->thread = static_cast<std::uint32_t>(shared.buffers.size() - 1);
    HostThreadSlot<Buffer*, BufferTag>::Set(buffer);
    return buffer;
}

inline void append(Event event, std::uint32_t argument) {
    auto* buffer = threadBuffer();
    const auto at = buffer->count.load(std::memory_order_relaxed);
    if (at >= Buffer::Capacity) return;
    buffer->entries[at] = {__rdtsc(), argument, static_cast<std::uint8_t>(event), {}};
    buffer->count.store(at + 1, std::memory_order_release);
}

// Writes every thread's events: a header (magic, the start's counter and clock, the end's), then per
// thread its ordinal, its count and its entries.
inline void write(State& shared) {
    const auto endTsc = __rdtsc();
    const auto endClock = std::chrono::steady_clock::now();
    const char* path = std::getenv("APS5_TIMELINE_OUT");
    std::FILE* file = std::fopen(path != nullptr ? path : "timeline.bin", "wb");
    if (file == nullptr) return;
    const auto nanoseconds = [](std::chrono::steady_clock::time_point point) { return static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(point.time_since_epoch()).count()); };
    const char magic[8] = {'A', 'P', 'S', '5', 'T', 'L', '1', 0};
    std::fwrite(magic, 1, sizeof(magic), file);
    const std::int64_t times[4] = {static_cast<std::int64_t>(shared.startTsc), nanoseconds(shared.startClock), static_cast<std::int64_t>(endTsc), nanoseconds(endClock)};
    std::fwrite(times, sizeof(times[0]), 4, file);
    std::lock_guard lock(shared.mutex);
    const auto threads = static_cast<std::uint32_t>(shared.buffers.size());
    std::fwrite(&threads, sizeof(threads), 1, file);
    for (const auto& buffer : shared.buffers) {
        const std::uint64_t count = buffer->count.load(std::memory_order_acquire);
        std::fwrite(&buffer->thread, sizeof(buffer->thread), 1, file);
        std::fwrite(&count, sizeof(count), 1, file);
        std::fwrite(buffer->entries.get(), sizeof(Entry), static_cast<std::size_t>(count), file);
    }
    std::fclose(file);
    std::fprintf(stderr, "[timeline] %u frames of %u threads written\n", shared.flips, threads);
}

inline void Mark(Event event, std::uint32_t argument = 0) {
    if (!Enabled()) return;
    auto& shared = state();
    if (event == Event::Flip) {
        const auto phase = shared.phase.load(std::memory_order_acquire);
        if (phase == 0) {
            const char* trigger = std::getenv("APS5_SAMPLE_TRIGGER");
            std::FILE* file = trigger != nullptr ? std::fopen(trigger, "rb") : nullptr;
            if (file == nullptr) return;
            std::fclose(file);
            shared.startTsc = __rdtsc();
            shared.startClock = std::chrono::steady_clock::now();
            shared.phase.store(1, std::memory_order_release);
        } else if (phase == 1 && ++shared.flips >= framesToRecord()) {
            shared.phase.store(2, std::memory_order_release);
            write(shared);
            return;
        }
    }
    if (shared.phase.load(std::memory_order_acquire) != 1) return;
    append(event, argument);
}

}

#endif
