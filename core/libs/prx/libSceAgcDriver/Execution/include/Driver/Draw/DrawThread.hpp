#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWTHREAD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWTHREAD_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace AgcDriver::DriverDetail {

struct PreparedDraw;

// A queue's draw back thread (docs/dev/FRONT_BACK_SPLIT.md, stages 2 and 3), queue 0's unless
// APS5_NO_DRAW_THREAD=1. The queue worker, the front, prepares each draw into Slot() and hands it
// over with Push; the back thread (Driver::runDrawThread) records the records in order, each under
// GuestMemory::GpuMutex, and empties them. A ring of Slots() records: up to Slots() - 1 are in
// flight while the front prepares the next, so a slow record no longer holds up the front at once.
// The front drains the back (waits until it recorded every record) before whatever must land after
// those draws: its own GpuMutex acquisitions (the GPU-lock front, GuestMemory::SetGpuLockFront),
// packets with effects outside the queue state, a draw that writes guest memory, and the end of a
// submission.
class DrawThread {
public:
    // Why the front waited for the back thread, for the [draw-thread] line (APS5_PROFILE_DRAW).
    enum class Wait : std::uint8_t { Push, Lock, Packet, Writer, End, Count };

    explicit DrawThread(std::uint32_t queue);
    ~DrawThread();
    DrawThread(const DrawThread&) = delete;
    DrawThread& operator=(const DrawThread&) = delete;

    // On unless APS5_NO_DRAW_THREAD=1 (the worker records its draws itself, as before) or
    // APS5_LOCKED_DRAW_PREPARE (a draw prepared under the GPU mutex cannot hand itself over while
    // holding it).
    static bool Enabled();
    // Stage 4 (APS5_RING_WRITES=1): end-of-pipe labels and small packet stores go through the ring
    // (Driver::ringWrite), so the worker no longer drains the thread before them.
    static bool RingWrites();

    // The front's side.
    // Whether the front may prepare a draw into Slot(): not while it prepares one there already (a
    // draw retried inside a draw takes the serial path), nor after the back thread left.
    bool FrontReady() const { return !preparing && !abandoned.load(std::memory_order_acquire); }
    // The slot the next draw is prepared into, free whenever the front runs: Push waits for the
    // record that used it last.
    PreparedDraw& Slot() { return Record(pushed.load(std::memory_order_relaxed)); }
    // Hands Slot() over once the slot after it is free (its last record is done); false when the
    // back thread left, and the caller records the draw itself.
    bool Push();
    // Waits until the back thread recorded every record handed to it. A drain before a packet names
    // its opcode, for the [draw-thread] line's packet breakdown.
    void Drain(Wait why, std::uint32_t opcode = 0);
    bool Busy() const { return recorded.load(std::memory_order_acquire) != pushed.load(std::memory_order_relaxed) && !abandoned.load(std::memory_order_acquire); }
    // The records handed over and the records done so far (wrapping counters).
    std::uint32_t PushedCount() const { return pushed.load(std::memory_order_relaxed); }
    std::uint32_t RecordedCount() const { return recorded.load(std::memory_order_acquire); }
    // Makes the calling thread the front: its GPU-lock front and its FrontDrawThread().
    void AttachFront();
    // Lets the back thread record what it was handed, stops it and detaches the front.
    void Stop();

    // The back thread's side.
    // Waits until record `taken` was handed over; false once the thread is to stop with nothing
    // left to record.
    bool AwaitRecord(std::uint32_t taken);
    // The slot record `taken` was prepared in.
    PreparedDraw& Record(std::uint32_t taken) { return *slots[taken % slots.size()]; }
    // The ring's slots: APS5_DRAW_RING (2 to 64, 8 by default; 2 is one record in flight).
    std::size_t Slots() const { return slots.size(); }
    // `count` records are done (and their slots empty).
    void Recorded(std::uint32_t count);
    // The back thread leaves early (an exception other than a draw's failure): every record handed
    // over counts as done, and the front records its draws itself from here on.
    void Abandon();

    const std::uint32_t queue;
    std::thread thread;
    // The worker's collect memo (GuestMemory::ShareCollectMemo), set before the thread starts.
    void* collectMemo = nullptr;
    // Set by the front while it prepares a draw into Slot().
    bool preparing = false;

private:
    void report();
    // Waits until `count` records are done (wrapping counters compared as differences); the wait
    // in milliseconds under APS5_PROFILE_DRAW, else 0.
    double awaitRecorded(std::uint32_t count, Wait why);

    std::vector<std::unique_ptr<PreparedDraw>> slots;
    alignas(64) std::atomic<std::uint32_t> pushed{0};
    // What the back thread sleeps on: rung by every Push and by Stop.
    std::atomic<std::uint32_t> doorbell{0};
    std::atomic<bool> backSleeping{false};
    std::atomic<bool> stopping{false};
    alignas(64) std::atomic<std::uint32_t> recorded{0};
    std::atomic<bool> frontSleeping{false};
    std::atomic<bool> abandoned{false};

    // The front's waits (APS5_PROFILE_DRAW), reported and reset every 10 s by the front.
    std::uint64_t pushes = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(Wait::Count)> waits{};
    std::array<double, static_cast<std::size_t>(Wait::Count)> waitedMs{};
    // The drains before packets by opcode.
    std::array<std::uint64_t, 256> packetWaits{};
    std::array<double, 256> packetWaitedMs{};
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

// The calling queue worker's draw back thread, or null.
DrawThread* FrontDrawThread();

}

#endif
