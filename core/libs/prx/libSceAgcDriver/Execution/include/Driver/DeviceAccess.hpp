#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DEVICEACCESS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DEVICEACCESS_HPP

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace AgcDriver::DriverDetail {

class DevicePointer {
public:
    std::shared_ptr<VulkanDevice> Load() const;
    operator std::shared_ptr<VulkanDevice>() const;
    DevicePointer& operator=(std::shared_ptr<VulkanDevice> value);
    void Reset();
    VulkanDevice* operator->() const;
    explicit operator bool() const;
    bool operator==(std::nullptr_t) const;

private:
    std::atomic<std::shared_ptr<VulkanDevice>> pointer;
};

// Packets use the device shared (lock_shared, once per packet); a device replacement (lock) waits for
// the users inside to leave while new ones wait for it. A packet enters and leaves with one atomic
// instruction each.
class DeviceUseGate {
public:
    void lock_shared();
    void unlock_shared();
    void lock();
    void unlock();

private:
    // The users inside in the low bits, and Replacing from a replacement's lock to its unlock;
    // whoever waits sleeps on the word.
    static constexpr std::uint32_t Replacing = 0x80000000u;
    std::atomic<std::uint32_t> state{0};
};

}

#endif
