#ifndef CORE_LIBS_PRX_LIBKERNEL_FILE_FILE_HPP
#define CORE_LIBS_PRX_LIBKERNEL_FILE_FILE_HPP

#include <cstddef>
#include <cstdint>
#ifdef _WIN32
#include <mutex>
#endif

#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

extern "C" {

int APS5_VABI sceKernelOpen(const char* path, int flags, std::uint16_t mode);
int APS5_VABI sceKernelClose(int d);
std::int64_t APS5_VABI sceKernelRead(int d, void* buf, std::size_t nbytes);
std::int64_t APS5_VABI sceKernelWrite(int d, const void* buf, std::size_t nbytes);
std::int64_t APS5_VABI sceKernelLseek(int d, std::int64_t offset, int whence);
int APS5_VABI sceKernelStat(const char* path, FileStat* sb);
int APS5_VABI sceKernelUnlink(const char* path);
int APS5_VABI sceKernelFcntl(int d, int command, ...);
#ifdef _WIN32
std::int64_t NativePositioned_nid_no_patch(int descriptor, void* buf, std::size_t nbytes, std::int64_t offset, bool write);
#endif

}

bool DescriptorIsOpen_nid_no_patch(int descriptor);
// Titles close descriptor 0 as an unset handle. Closing the standard descriptors is refused, so
// their numbers are never handed to a title's files, where such a close would shut another
// thread's file.
inline bool IsStandardDescriptor_nid_no_patch(int descriptor) {
    return descriptor >= 0 && descriptor <= 2;
}
#ifdef _WIN32
// Held across every use of a descriptor's file offset and across its close. A positioned transfer
// moves the shared offset and then restores it, so a read, write, seek or close of the same
// descriptor on another thread must not run in between. Duplicated descriptors share an offset
// but not this lock.
std::mutex& DescriptorOffsetLock_nid_no_patch(int descriptor);
#endif

#endif
