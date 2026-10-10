#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawThread.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include <cstdlib>

namespace AgcDriver::DriverDetail {

void Driver::flushBetweenPackets(std::uint32_t queue, std::uint32_t header, bool labelPacket) {
    static const bool packetFlush = !LabelBatchSubmit() && std::getenv("APS5_NO_LABEL_PACKET_FLUSH") == nullptr;
    static const bool ownLock = std::getenv("APS5_LABEL_OWN_LOCK") != nullptr;
    static std::atomic<std::uint64_t> packetFlushes{0}, deadlineFlushes{0}, capFlushes{0}, boundaryReaps{0}, overdueLocks{0}, presentDeferrals{0};
    auto& deferred = deferredLabels();
    const bool queued = !deferred.labels.empty();
    const bool queuedTable = queued && QueuedLabelTable();
    if (queuedTable) noteQueuedLabels(queue);
    const bool selfLocking = !ownLock && PacketLocksItself(header);
    bool submit = false, record = false, needsRecord = false, deferredDue = false, pendingDue = false;
    const auto pending = Graphics::Recorder::PendingLabelSince();

    // The worker's clock and last reap try (a thread slot: thread_local is emulated on MinGW). The
    // clock is read at every packet while the worker's own labels wait, whose deadline decides whether
    // it records them now, else at every 8th packet: the label deadline and the reap interval (a
    // millisecond and more) take a few microseconds of lag, and a read per packet was 1.8% of the
    // worker on the S2 page, where completions are pending all the time. APS5_CLOCK_EACH_PACKET=1 reads
    // it at every packet, as before.
    static const bool clockEachPacket = std::getenv("APS5_CLOCK_EACH_PACKET") != nullptr;
    struct PacketClock {
        std::chrono::steady_clock::time_point now{};
        std::chrono::steady_clock::time_point lastReapTry{};
        std::uint32_t packets = 0;
    };
    struct PacketClockTag {};
    auto& clock = ThreadScratch<PacketClock, PacketClockTag>();
    const bool completions = completionsPending();
    const bool reapDue = completions && (reapEachPacket() || queue == 0);
    if (queued || ((pending.has_value() || reapDue) && (clockEachPacket || (++clock.packets & 7u) == 0))) clock.now = std::chrono::steady_clock::now();
    const auto now = clock.now;

    bool recordAtLock = false;
    if (queued) {
        if (!labelPacket) {
            record = true;

            needsRecord = !queuedTable && NeedsRecordedLabels(header) && !selfLocking;
        }
        if (now - deferred.since >= labelFlushDeadline()) record = deferredDue = true;
        if (record && !deferredDue && selfLocking && !labelTryEachPacket()) {
            record = false;
            recordAtLock = true;
        }
    }
    if (pending.has_value()) {
        if (!labelPacket && packetFlush) submit = true;
        if (now - *pending >= labelFlushDeadline()) submit = pendingDue = true;
    }
    const auto work = Graphics::Recorder::RecordedWorkSinceSubmit();
    const bool capped = !submit && !record && batchCap() != 0 && work >= batchCap();
    const bool reap = !submit && !record && !capped && reapDue && (reapEachPacket() || now - clock.lastReapTry >= ReapInterval);
    if (!submit && !record && !capped && !reap) {
        if (recordAtLock) ++recordTriesSkipped;
        return;
    }
    if (reap) clock.lastReapTry = now;
    bool outright = needsRecord;
    if (LabelBatchSubmit()) {
        const auto grace = 4 * labelFlushDeadline();
        const bool overdue = (deferredDue && now - deferred.since >= grace) || (pendingDue && now - *pending >= grace) || (capped && work >= 2 * batchCap());
        if (overdue) ++overdueLocks;
        outright = outright || overdue;
    } else if (deferredDue || (queue == 0 && (capped || (submit && (!selfLocking || pendingDue))))) {
        outright = true;
    }
    // A submit made while the presenter holds the queue (queueing its presentation) waits for it
    // inside Recorder::Submit, holding the GPU mutex with the draw thread drained, so both threads
    // stall (about 0.2 ms a frame on the S2 menu page). A batch that only needs submitting waits for
    // a later packet instead, unless it is 4 times past its cap or its labels 8 times past their
    // deadline. APS5_NO_PRESENT_DEFER=1 submits at once.
    static const bool presentDefer = std::getenv("APS5_NO_PRESENT_DEFER") == nullptr;
    if (presentDefer && !record && !recordAtLock && (submit || capped) && GuestMemory::QueueMutex().HeldElsewhere()) {
        const bool farPast = (capped && work >= 4 * batchCap()) || (pendingDue && now - *pending >= 8 * labelFlushDeadline());
        if (!farPast) {
            presentDeferrals.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    // The failed tries are only counted for the [labels] line: a locked add per try otherwise.
    const auto failedTry = [&] {
        if (!profile) return;
        ++triesFailed[record ? TryLabel : submit || capped ? TryFlush : TryReap];
        if (selfLocking && record) ++packetLockDeferred;
        else if (selfLocking && submit) ++packetSubmitDeferred;
    };
    // At the draw thread's front a try fails while the draw thread holds records (the mutex's
    // front, GpuMutexType::try_lock): that is known without touching the mutex, which the tries
    // did about 2.8M times per 10 s on the S2 menu page.
    if (!outright) {
        if (const auto* back = FrontDrawThread(); back != nullptr && back->Busy() && !GuestMemory::GpuMutex().HeldByThisThread()) {
            failedTry();
            return;
        }
    }
    std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::defer_lock);

    if (record) GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
    else if (submit || capped) GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);

    if (outright) {
        gpuLock.lock();
    } else if (!gpuLock.try_lock()) {
        failedTry();
        return;
    }
    const auto localDevice = device.Load();
    if (record || recordAtLock) {
        recordDeferredLabels(localDevice.get(), queue);

        if ((!labelPacket && packetFlush) || deferredDue) submit = true;
    }
    if (localDevice == nullptr) return;
    if (submit || capped) {

        if (!Graphics::Recorder::PendingLabelSince().has_value() && (batchCap() == 0 || Graphics::Recorder::RecordedWorkSinceSubmit() < batchCap())) return;
        localDevice->SubmitRecorded(queue == 0);
        if (capped) ++capFlushes;
        else if (!labelPacket && packetFlush) ++packetFlushes;
        else ++deadlineFlushes;
    } else if (reap) {
        localDevice->ReapRecorded();
        ++boundaryReaps;
    }

    static std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    if (!profile) return;
    const auto reportNow = std::chrono::steady_clock::now();
    if (reportNow - lastReport < std::chrono::seconds(10)) return;
    lastReport = reportNow;
    const auto stores = Graphics::Recorder::StoreCounts();
    const auto failed = [](TrySite site) { return static_cast<unsigned long long>(triesFailed[site].load()); };
    AgcDriver::ProfilePrint_nid_no_patch("[labels] batch flushes between packets: %llu at the next packet, %llu by deadline, %llu by size cap (%llu well overdue, waited for the mutex); %llu boundary reaps; %llu submits deferred while the presenter held the queue; record tries skipped (self-locking next) %llu; failed tries: label %llu, flush %llu, reap %llu, poll %llu, idle %llu, capture %llu, poll-record %llu; at the packet's own lock: %llu label groups (%.0f ms), %llu submits (%.0f ms incl. queue 0's reaps), left to it by a failed try: %llu groups, %llu submits; after a capture: %llu groups recorded by a try, %llu packets redone for a queued label over the capture; %llu suspend points; stores: %llu in %llu runs (%llu joined, %llu replaced, %llu WAW barriers, %llu joins refused; %llu runs recorded at submit, %llu in place before a writer or reader); DCC key stores: %llu queued in %llu runs (%llu before a writer, %llu joined a queued range); queued labels: %llu noted (%llu over a recorded entry), %llu same-queue wait hits, %llu groups recorded from a poll loop, %llu from the flush hook (%llu hook accesses, %llu inside a completion skipped)\n", static_cast<unsigned long long>(packetFlushes.load()), static_cast<unsigned long long>(deadlineFlushes.load()), static_cast<unsigned long long>(capFlushes.load()), static_cast<unsigned long long>(overdueLocks.load()), static_cast<unsigned long long>(boundaryReaps.load()), static_cast<unsigned long long>(presentDeferrals.load()), static_cast<unsigned long long>(recordTriesSkipped.load()), failed(TryLabel), failed(TryFlush), failed(TryReap), failed(TryPoll), failed(TryIdle), failed(TryCapture), failed(TryPollRecord), static_cast<unsigned long long>(packetLockRecords.load()), packetLockRecordUs.load() / 1000.0, static_cast<unsigned long long>(packetLockSubmits.load()), packetLockSubmitUs.load() / 1000.0, static_cast<unsigned long long>(packetLockDeferred.load()), static_cast<unsigned long long>(packetSubmitDeferred.load()), static_cast<unsigned long long>(captureTryRecords.load()), static_cast<unsigned long long>(captureRetries.load()), static_cast<unsigned long long>(suspendPoints.load()), static_cast<unsigned long long>(stores.stores), static_cast<unsigned long long>(stores.runs), static_cast<unsigned long long>(stores.joined), static_cast<unsigned long long>(stores.replaced), static_cast<unsigned long long>(stores.wawBarriers), static_cast<unsigned long long>(stores.joinsRefused), static_cast<unsigned long long>(stores.runsAtSubmit), static_cast<unsigned long long>(stores.runsForced), static_cast<unsigned long long>(stores.keyStores), static_cast<unsigned long long>(stores.keyStoreRuns), static_cast<unsigned long long>(stores.keyStoreRunsForWriter), static_cast<unsigned long long>(stores.keyStoresJoined), static_cast<unsigned long long>(stores.queuedNoted), static_cast<unsigned long long>(stores.queuedOverRecorded), static_cast<unsigned long long>(stores.queuedHits), static_cast<unsigned long long>(pollLabelRecords.load()), static_cast<unsigned long long>(hookLabelRecords.load()), static_cast<unsigned long long>(stores.queuedHookRecords), static_cast<unsigned long long>(stores.queuedHookInCompletion));
}

}
