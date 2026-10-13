#include "ResidentPresent.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/PresentationScaler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GpuColorTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;
using AgcDriver::DisplayBuffer;
using AgcDriver::ResidentPresent;

constexpr std::uint64_t Bgra8 = 0x8000000000000000ull;
constexpr std::uint64_t Rgba8 = 0x8000000022000000ull;
constexpr std::uint64_t TenBit = 0x0100000000000000ull;
constexpr std::uint64_t Rgba16Float = 0xc001000622000000ull;

void decisionTests() {
    using AgcDriver::DisplayTexelFormat;
    using AgcDriver::ResidentPresentPath;
    Require(DisplayTexelFormat(Bgra8) == VK_FORMAT_B8G8R8A8_UNORM && DisplayTexelFormat(Rgba8) == VK_FORMAT_R8G8B8A8_UNORM, "8-bit display texel formats are wrong");
    Require(DisplayTexelFormat(Bgra8 | TenBit) == VK_FORMAT_A2R10G10B10_UNORM_PACK32 && DisplayTexelFormat(Rgba8 | TenBit) == VK_FORMAT_A2B10G10R10_UNORM_PACK32, "10-bit display texel formats are wrong");
    Require(ResidentPresentPath(VK_FORMAT_B8G8R8A8_UNORM, Bgra8, true) == ResidentPresent::Blit && ResidentPresentPath(VK_FORMAT_R8G8B8A8_UNORM, Rgba8, true) == ResidentPresent::Blit, "an 8-bit image in its display's order is not blitted");
    Require(ResidentPresentPath(VK_FORMAT_B8G8R8A8_UNORM, Bgra8, false) == ResidentPresent::Convert, "an 8-bit image that cannot be a blit source is not converted");
    Require(ResidentPresentPath(VK_FORMAT_R8G8B8A8_UNORM, Bgra8, true) == ResidentPresent::Convert && ResidentPresentPath(VK_FORMAT_B8G8R8A8_UNORM, Rgba8, true) == ResidentPresent::Convert, "an 8-bit image in the other order is not converted");
    Require(ResidentPresentPath(VK_FORMAT_R8G8B8A8_SRGB, Rgba8, true) == ResidentPresent::Convert, "an sRGB image is blitted (the blit would decode it)");
    Require(ResidentPresentPath(VK_FORMAT_R8G8B8A8_UINT, Bgra8, true) == ResidentPresent::Convert && ResidentPresentPath(VK_FORMAT_R8G8B8A8_UINT, Rgba8, true) == ResidentPresent::Convert, "an 8-bit integer image (the bytes a compute pass stores) is not converted");
    for (const auto display : {Bgra8 | TenBit, Rgba8 | TenBit}) {
        for (const auto storage : {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32}) {
            Require(ResidentPresentPath(storage, display, true) == ResidentPresent::Convert, "a 10-bit image is blitted (the blit rounds where the guest path truncates)");
        }
        Require(ResidentPresentPath(VK_FORMAT_B8G8R8A8_UNORM, display, true) == ResidentPresent::None, "an 8-bit image presents a 10-bit display");
    }
    Require(ResidentPresentPath(VK_FORMAT_A2B10G10R10_UNORM_PACK32, Bgra8, true) == ResidentPresent::None && ResidentPresentPath(VK_FORMAT_R16G16_SFLOAT, Bgra8, true) == ResidentPresent::None, "an image of another type presents an 8-bit display");
    const auto guest = FindGuestColorTargetFormat(VK_FORMAT_A2R10G10B10_UNORM_PACK32, 4);
    Require(guest.has_value() && guest == FindGuestTextureFormat(VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4), "the blue-low 10-bit color target has no storage format");
}

std::uint32_t memoryType(const Context& context, std::uint32_t bits, VkMemoryPropertyFlags flags) {
    for (std::uint32_t i = 0; i < context.memory.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) != 0 && (context.memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
    }
    throw std::runtime_error("no device-local memory type for the test image");
}

void conversionTest(const Context& context, std::uint64_t pixelFormat, VkFormat format) {
    constexpr std::uint32_t width = 200;
    constexpr std::uint32_t height = 130;
    const auto texelBytes = AgcDriver::DisplayTexelBytes(pixelFormat);
    std::vector<std::byte> words(static_cast<std::size_t>(width) * height * texelBytes);
    std::uint32_t seed = 0x9e3779b9u ^ static_cast<std::uint32_t>(format);
    for (std::size_t i = 0; i < words.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        words[i] = static_cast<std::byte>(seed >> 24u);
    }
    const ColorTargetLayout layout(width, height, ColorTileMode::RenderTarget, texelBytes);
    std::vector<std::byte> storage(layout.Bytes() + 65536);
    auto* aligned = reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(storage.data()) + 65535u) & ~std::uintptr_t{65535});
    std::span<std::byte> guest(aligned, layout.Bytes());
    const DisplayBuffer display{reinterpret_cast<std::uint64_t>(aligned), pixelFormat, width, height};
    Require(AgcDriver::DisplayBufferSize(display) == guest.size(), "test display buffer size differs from the color layout");
    if (texelBytes == 4u) {
        layout.Tile(words, guest);
    } else {
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) std::memcpy(guest.data() + AgcDriver::DisplayBufferOffset(display, x, y), words.data() + (static_cast<std::size_t>(y) * width + x) * texelBytes, texelBytes);
        }
    }
    const auto expected = AgcDriver::DecodeDisplayBuffer(display, guest);

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &imageInfo, nullptr, &image), "vkCreateImage resident present test");
    struct Release {
        const Context& context;
        VkImage& image;
        VkDeviceMemory& memory;
        ~Release() {
            if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
            if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        }
    } release{context, image, memory};
    VkMemoryRequirements requirements{};
    context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType(context, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory resident present test");
    Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory resident present test");

    Buffer upload(context, words.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memcpy(upload.Bytes().data(), words.data(), words.size());
    Buffer fromImage(context, expected.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer fromGuest(context, expected.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    GpuColorTransfer transfer(context);
    const bool redLow = AgcDriver::DisplayRedLow(pixelFormat);
    const bool tenBit = AgcDriver::DisplayTenBit(pixelFormat);
    const bool halfFloat = AgcDriver::DisplayHalfFloat(pixelFormat);
    const auto barrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    const auto readback = [&](VkCommandBuffer commands, Buffer& destination) {
        const VkBufferCopy copy{0, 0, expected.size()};
        context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands, transfer.LinearBuffer(), destination.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
    };
    transfer.Upload(display.address, width, height, ColorTileMode::RenderTarget, texelBytes);
    {
        CommandBatch batch(context);
        const auto commands = batch.Handle();
        VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = image;
        toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransfer);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {width, height, 1};
        context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, upload.Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        auto toGeneral = toTransfer;
        toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
        transfer.Detile(commands, redLow, tenBit, halfFloat);
        readback(commands, fromGuest);
        transfer.DetileImage(commands, image, VK_IMAGE_LAYOUT_GENERAL, width, height, ColorTileMode::RenderTarget, redLow, tenBit, halfFloat);
        readback(commands, fromImage);
        batch.SubmitAndWait();
    }
    const auto name = std::string(halfFloat ? "16-bit float " : tenBit ? "10-bit " : "8-bit ") + (redLow ? "red-low" : "blue-low") + " display from VkFormat " + std::to_string(static_cast<int>(format));
    Require(std::equal(expected.begin(), expected.end(), fromGuest.Bytes().begin()), "the GPU detile differs from the CPU decode: " + name);
    Require(std::equal(expected.begin(), expected.end(), fromImage.Bytes().begin()), "the resident image presents other pixels than guest memory: " + name);
}

// A shrinking present blit goes through 2x halvings first (PresentationHalvings): a 1-texel checker
// shrunk to under a quarter comes out a flat grey, where one linear blit keeps texels of a single
// colour wherever its taps land on them.
void fittedBlitTest(const Context& context) {
    using AgcDriver::PresentationHalvings;
    Require(PresentationHalvings(3840, 2160, 1100, 619) == 1 && PresentationHalvings(3840, 2160, 1920, 1080) == 1 && PresentationHalvings(3840, 2160, 640, 360) == 2, "a 4K image takes the wrong number of halvings");
    Require(PresentationHalvings(3840, 2160, 1921, 1081) == 0 && PresentationHalvings(1920, 1080, 2560, 1440) == 0 && PresentationHalvings(256, 128, 60, 30) == 2, "an image is halved below its fitted rectangle, or not halved above it");
    constexpr std::uint32_t width = 256;
    constexpr std::uint32_t height = 128;
    constexpr std::uint32_t outWidth = 60;
    constexpr std::uint32_t outHeight = 30;
    AgcDriver::PresentationScaler scaler(context, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM);
    scaler.EnsureSourceImage(width, height);
    Buffer upload(context, static_cast<std::size_t>(width) * height * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto value = static_cast<std::byte>((x + y) % 2 == 0 ? 0xff : 0x00);
            auto* texel = upload.Bytes().data() + (static_cast<std::size_t>(y) * width + x) * 4;
            texel[0] = texel[1] = texel[2] = value;
            texel[3] = std::byte{0xff};
        }
    }
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
    imageInfo.extent = {outWidth, outHeight, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &imageInfo, nullptr, &image), "vkCreateImage fitted blit test");
    struct Release {
        const Context& context;
        VkImage& image;
        VkDeviceMemory& memory;
        ~Release() {
            if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
            if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        }
    } release{context, image, memory};
    VkMemoryRequirements requirements{};
    context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType(context, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory fitted blit test");
    Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory fitted blit test");
    Buffer readback(context, static_cast<std::size_t>(outWidth) * outHeight * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    const auto barrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    {
        CommandBatch batch(context);
        const auto commands = batch.Handle();
        scaler.RecordUpload(commands, upload.Handle());
        VkImageMemoryBarrier toDestination{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toDestination.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toDestination.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDestination.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDestination.srcQueueFamilyIndex = toDestination.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDestination.image = image;
        toDestination.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toDestination);
        scaler.RecordBlit(commands, image, outWidth, outHeight);
        auto toSource = toDestination;
        toSource.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSource.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toSource);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {outWidth, outHeight, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.Handle(), 1, &region);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        batch.SubmitAndWait();
    }
    const auto* texels = readback.Bytes().data();
    for (std::size_t i = 0; i < static_cast<std::size_t>(outWidth) * outHeight; ++i) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const auto value = std::to_integer<int>(texels[i * 4 + channel]);
            Require(value >= 112 && value <= 144, "a shrunk 1-texel checker is not flat grey at texel " + std::to_string(i) + ": " + std::to_string(value));
        }
    }
}

}

void RunResidentPresentTests(const Context& context) {
    decisionTests();
    fittedBlitTest(context);
    conversionTest(context, Bgra8 | TenBit, VK_FORMAT_A2B10G10R10_UNORM_PACK32);
    conversionTest(context, Rgba8 | TenBit, VK_FORMAT_A2B10G10R10_UNORM_PACK32);
    conversionTest(context, Bgra8 | TenBit, VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    conversionTest(context, Bgra8, VK_FORMAT_R8G8B8A8_UNORM);
    conversionTest(context, Rgba8, VK_FORMAT_B8G8R8A8_UNORM);
    conversionTest(context, Rgba16Float, VK_FORMAT_R16G16B16A16_SFLOAT);
    std::cout << "resident present format and conversion tests passed\n";
}
