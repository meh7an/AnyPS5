#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PRESENTATIONSCALER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PRESENTATIONSCALER_HPP

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstdint>

namespace AgcDriver {

class PresentationScaler {
public:
    PresentationScaler(const Graphics::Context& context, VkFormat sourceFormat, VkFormat destinationFormat);
    ~PresentationScaler();
    PresentationScaler(const PresentationScaler&) = delete;
    PresentationScaler& operator=(const PresentationScaler&) = delete;

    void EnsureSourceImage(std::uint32_t width, std::uint32_t height);
    void RecordUpload(VkCommandBuffer commands, VkBuffer uploadBuffer);
    // The source image takes `image`'s texels as they are (a copy between size-compatible formats):
    // a resident display image whose bytes are already in the source's order.
    void RecordCopyFrom(VkCommandBuffer commands, VkImage image, VkImageLayout layout);
    void RecordClear(VkCommandBuffer commands, const VkClearColorValue& color);
    void RecordImage(VkCommandBuffer commands, VkImage image);
    // Fills the source image from another image of the same extent (any blittable format, layout
    // unchanged), leaving the source image ready to blit or read back.
    void RecordBlitInto(VkCommandBuffer commands, VkImage image, VkImageLayout layout, VkFilter filter);
    void RecordBlitInto(VkCommandBuffer commands, VkImage image, VkImageLayout layout, VkFilter filter, std::uint32_t width, std::uint32_t height);
    // Copies the source image's texels (tightly packed rows of its format) into a host-readable buffer
    // and makes them visible to the host once the submission completes.
    void RecordReadback(VkCommandBuffer commands, VkBuffer destination);
    void RecordBlit(VkCommandBuffer commands, VkImage destinationImage, std::uint32_t destinationWidth, std::uint32_t destinationHeight);
    // The aspect-fitting blit of RecordBlit from any image (presenting a resident render target needs
    // no copy into the source image first), through the scaler's halving chain
    // (PresentationHalvings): a linear blit that shrinks the image by more than half each way reads
    // 2x2 texels of every larger block, so fine patterns alias into coarse ones (the title's 4x4
    // dither fades, presented from 4K into a window, became a grid). Each 2x linear blit of the chain
    // averages 2x2 texels, and the last blit shrinks by less than half. APS5_PRESENT_SINGLE_BLIT=1
    // blits once, as before. Returns whether the chain was used.
    bool RecordFittedBlit(VkCommandBuffer commands, VkImage image, VkImageLayout layout, std::uint32_t width, std::uint32_t height, VkFilter filter, VkImage destinationImage, std::uint32_t destinationWidth, std::uint32_t destinationHeight);
    std::uint32_t SourceWidth() const { return sourceWidth; }
    std::uint32_t SourceHeight() const { return sourceHeight; }

private:
    void release() noexcept;
    void releaseChain() noexcept;
    void ensureChain(std::uint32_t width, std::uint32_t height, std::uint32_t levels);

    Graphics::Context context;
    VkFormat sourceFormat;
    std::uint32_t sourceWidth = 0;
    std::uint32_t sourceHeight = 0;
    VkImage sourceImage = VK_NULL_HANDLE;
    VkDeviceMemory sourceMemory = VK_NULL_HANDLE;
    // The halving chain: mip level k holds the presented image at 1/2^(k+1) of its extent, in the
    // source format (the swapchain's), when that format can be a blit destination. One image for
    // every present in flight (its barrier orders their blits); it is re-created only when the
    // presented size or the swapchain extent changes, and VulkanDevice retires the presents that
    // used it first in both cases (they count as shared for its size-change wait; Resize waits idle).
    bool chainable = false;
    std::uint32_t chainWidth = 0;
    std::uint32_t chainHeight = 0;
    std::uint32_t chainLevels = 0;
    VkImage chainImage = VK_NULL_HANDLE;
    VkDeviceMemory chainMemory = VK_NULL_HANDLE;
};

// The 2x halvings before a linear blit of a width x height image into a fitted rectangle of
// fittedWidth x fittedHeight: while both halves still cover the rectangle.
std::uint32_t PresentationHalvings(std::uint32_t width, std::uint32_t height, std::uint32_t fittedWidth, std::uint32_t fittedHeight);

}

#endif
