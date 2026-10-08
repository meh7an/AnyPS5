#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace AgcDriver::DriverDetail {

void Driver::RegisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
    require(output != nullptr, "null video output");
    std::lock_guard lock(mutex);
    rethrowFailure();
    checkStopping();
    require(outputs.emplace(handle, output).second, "video output already registered");
}

void Driver::UnregisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
    std::lock_guard lock(mutex);
    const auto it = outputs.find(handle);
    require(it != outputs.end() && it->second == output, "video output registration mismatch");
    outputs.erase(it);
}

namespace {

// APS5_TRACE_FLIP_PACING: where each presentation's time goes, averaged every 5 s. Present runs on
// the presenter thread alone, so the totals need no lock.
struct PresentPacing {
    using Clock = std::chrono::steady_clock;
    enum Stage { Retire, Acquire, Blit, Queue, Fence, Total, Count };
    std::array<double, Count> ms{};
    std::array<double, Count> maxMs{};
    std::uint64_t presents = 0;
    Clock::time_point lastReport = Clock::now();
    Clock::time_point lap = Clock::now();

    static bool Enabled() {
        static const bool enabled = std::getenv("APS5_TRACE_FLIP_PACING") != nullptr;
        return enabled;
    }
    void Start() { lap = Clock::now(); }
    void Mark(Stage stage) {
        const auto now = Clock::now();
        const auto elapsed = std::chrono::duration<double, std::milli>(now - lap).count();
        ms[stage] += elapsed;
        maxMs[stage] = std::max(maxMs[stage], elapsed);
        lap = now;
    }
    void Finish(Clock::time_point started) {
        const auto now = Clock::now();
        const auto elapsed = std::chrono::duration<double, std::milli>(now - started).count();
        ms[Total] += elapsed;
        maxMs[Total] = std::max(maxMs[Total], elapsed);
        ++presents;
        if (now - lastReport < std::chrono::seconds(5)) return;
        const double count = static_cast<double>(presents);
        std::fprintf(stderr, "[pacing] present: %llu in %.1f s; avg/max ms: retire %.2f/%.1f acquire %.2f/%.1f blit %.2f/%.1f queue-present %.2f/%.1f gpu fence %.2f/%.1f total %.2f/%.1f\n", static_cast<unsigned long long>(presents), std::chrono::duration<double>(now - lastReport).count(), ms[Retire] / count, maxMs[Retire], ms[Acquire] / count, maxMs[Acquire], ms[Blit] / count, maxMs[Blit], ms[Queue] / count, maxMs[Queue], ms[Fence] / count, maxMs[Fence], ms[Total] / count, maxMs[Total]);
        ms.fill(0);
        maxMs.fill(0);
        presents = 0;
        lastReport = now;
    }
};

PresentPacing& Pacing() {
    static PresentPacing pacing;
    return pacing;
}

}

void Driver::Present(const PresentationWindow& window, const DisplayBuffer* buffer, bool opaque, void (*gpuReady)(void*), void* context) {

    PerformanceContext timingContext(window.timing.get());
    const bool pacing = PresentPacing::Enabled();
    const auto pacingStart = PresentPacing::Clock::now();
    if (pacing) Pacing().Start();
    PerformanceTimer timing("Driver.Present");
    static thread_local bool pinned = false;
    if (!pinned) {
        pinned = true;
        PinWorkerThread("presenter");
        GuestMemory::MarkPresenterThread();
    }
    CheckFailure();
    require(gpuReady != nullptr && context != nullptr, "missing GPU completion callback");
    require(window.getDrawableSize != nullptr, "missing window drawable size query");

    static const bool syncFlip = std::getenv("APS5_SYNC_FLIP") != nullptr;
    static const std::size_t inFlight = VulkanDevice::FlipInFlight();
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    std::shared_ptr<VulkanDevice> presenting;
    timing.Mark("validate");
    try {
        bool submitted = false;
        bool presentable = false;
        bool trailing = false;
        double waitedMs = 0;
        {
            std::unique_lock replacing(deviceReplacement, std::defer_lock);
            if (const auto current = device.Load(); current == nullptr || current->Window() == nullptr) replacing.lock();
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Present);
            std::lock_guard lock(GuestMemory::GpuMutex());
            timing.Mark("gpu_mutex_wait");
            if (device == nullptr || device->Window() == nullptr) {
                if (device) {
                    device->PrepareForReplacement();
                    replacedDevices.push_back(device);
                }
                device = std::make_shared<VulkanDevice>(&window);
            }
            require(device->Window() == window.context, "presentation window does not match device surface");
            presenting = device;
            timing.Mark("device_setup");
            std::uint32_t drawableWidth = 0;
            std::uint32_t drawableHeight = 0;
            window.getDrawableSize(window.context, &drawableWidth, &drawableHeight);
            presenting->Resize(drawableWidth, drawableHeight);
            timing.Mark("resize");
            presentable = presenting->Presentable();
            if (buffer != nullptr) require(buffer->width == window.width && buffer->height == window.height, "display buffer extent differs from output");
        }

        if (presentable && !syncFlip) {
            if (inFlight != 0) {

                waitedMs = presenting->RetirePresents(presenting->PresentWaitsForSlots(buffer) ? 0 : inFlight);
                timing.Mark("inflight_wait");
            }
            if (pacing) Pacing().Mark(PresentPacing::Retire);
            presentable = presenting->AcquireImage();
            timing.Mark("acquire_image");
            if (pacing) Pacing().Mark(PresentPacing::Acquire);
        }
        if (presentable) {
            // The presentation is queued without the GPU mutex, so the queue workers' draws go on
            // meanwhile (a submit of theirs waits for it). The queue is taken before the GPU mutex
            // goes: nothing comes between the blit and its presentation in queue order.
            // APS5_NO_UNLOCKED_PRESENT=1 queues it under the GPU mutex as well, as before.
            static const bool lockedPresent = std::getenv("APS5_NO_UNLOCKED_PRESENT") != nullptr;
            std::unique_lock queue(GuestMemory::QueueMutex(), std::defer_lock);
            const auto queuePresent = [&] {
                presenting->QueuePresent();
                queue.unlock();
                timing.Mark("queue_present");
                if (pacing) Pacing().Mark(PresentPacing::Queue);
                trailing = !syncFlip;
                submitted = false;
            };
            {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Present);
                std::lock_guard lock(GuestMemory::GpuMutex());
                timing.Mark("gpu_mutex_wait");
                if (buffer != nullptr) {
                    if (syncFlip) {
                        presenting->WaitIdle();
                        timing.Mark("device_idle_wait");
                    }
                    submitted = presenting->PresentDisplayBuffer(*buffer);
                    timing.Mark("present_display_buffer");
                } else {
                    submitted = presenting->PresentClear(window.width, window.height, opaque);
                    timing.Mark("present_clear");
                }
                if (pacing) Pacing().Mark(PresentPacing::Blit);
                if (submitted && syncFlip) {
                    waitedMs = presenting->FinishPresent();
                    timing.Mark("render_fence_wait");
                }
                if (submitted && (syncFlip || inFlight != 0)) {
                    queue.lock();
                    if (lockedPresent) queuePresent();
                }
            }
            if (queue.owns_lock()) queuePresent();
        }
        if (trailing) {
            waitedMs += presenting->FinishPresent();
            timing.Mark("render_fence_wait");
            if (pacing) Pacing().Mark(PresentPacing::Fence);
        }
        if (submitted) {
            waitedMs = presenting->FinishPresent();
            timing.Mark("render_fence_wait");

            std::lock_guard queue(GuestMemory::QueueMutex());
            presenting->QueuePresent();
            timing.Mark("queue_present");
        }
        gpuReady(context);
        timing.Mark("release_and_callback");
        if (pacing) Pacing().Finish(pacingStart);
        if (profile) reportPresents(waitedMs, inFlight);
        CheckFailure();
    } catch (const ProcessShutdown&) {
        throw;
    } catch (...) {
        ReportFailure(std::current_exception());
        throw;
    }
}

void Driver::reportPresents(double waitedMs, std::size_t inFlight) {
    static std::vector<double> waits;
    static VulkanDevice::PresentStatistics previous{};
    static std::uint64_t previousFlips = 0, windowSerial = 0, previousUnsignaled = 0;
    static auto lastReport = std::chrono::steady_clock::now();
    waits.push_back(waitedMs);
    if (waits.size() == 1) windowSerial = flipSerial.load();
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::sort(waits.begin(), waits.end());
    const auto percentile = [&](double fraction) { return waits.empty() ? 0.0 : waits[std::min(waits.size() - 1, static_cast<std::size_t>(fraction * static_cast<double>(waits.size())))]; };
    const auto counts = VulkanDevice::PresentCounts();
    const auto presents = counts.presents - previous.presents;
    const double perPresent = presents != 0 ? 1.0 / static_cast<double>(presents) : 0.0;
    const auto flips = flipsCounted.load() - previousFlips;
    const double perFlip = flips != 0 ? 1.0 / static_cast<double>(flips) : 0.0;
    const auto serial = flipSerial.load();
    AgcDriver::ProfilePrint_nid_no_patch( "[present] %llu presents over 10 s (%zu may trail on the GPU); %s p50 %.2f ms, p90 %.2f ms, max %.2f ms over %zu; per present: GPU busy %.1f ms, idle gaps %.1f ms (%llu batches without a completion record); per flip: %.1f batches recorded (%.1f unsignaled at the flip); per present: %.1f batches ahead of the blit, %.1f after it; after the flip packet: last submit %.1f ms, blit submit %.1f ms; in-place reads overwritten before execution: %llu of %llu checked\n", static_cast<unsigned long long>(presents), inFlight, inFlight != 0 ? "inflight_wait" : "render_fence_wait", percentile(0.5), percentile(0.9), waits.empty() ? 0.0 : waits.back(), waits.size(), (counts.gpuBusyMs - previous.gpuBusyMs) * perPresent, (counts.gpuGapMs - previous.gpuGapMs) * perPresent, static_cast<unsigned long long>(counts.gpuUnread - previous.gpuUnread), static_cast<double>(serial - windowSerial) * perFlip, static_cast<double>(flipBatchesUnsignaled.load() - previousUnsignaled) * perFlip, static_cast<double>(counts.batchesAheadOfBlit - previous.batchesAheadOfBlit) * perPresent, static_cast<double>(counts.batchesAfterFlip - previous.batchesAfterFlip) * perPresent, (counts.lastSubmitAfterFlipMs - previous.lastSubmitAfterFlipMs) * perPresent, (counts.blitSubmitAfterFlipMs - previous.blitSubmitAfterFlipMs) * perPresent, static_cast<unsigned long long>(counts.readsOverwritten - previous.readsOverwritten), static_cast<unsigned long long>(counts.readsChecked - previous.readsChecked));
    previous = counts;
    previousFlips = flipsCounted.load();
    previousUnsignaled = flipBatchesUnsignaled.load();
    waits.clear();
}

void Driver::ReleaseWindow(void* window) {
    std::lock_guard lock(GuestMemory::GpuMutex());
    if (device && device->Window() == window) device.Reset();
}

}
