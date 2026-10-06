#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaDescriptorFormat.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

class DepthSurface {
public:
    DepthSurface(const Context& context, const DepthTarget& target) : context(context), target(target) {
        this->context.bufferPool.reset();
        VkFormatProperties properties{};
        context.formatProperties(context.physical, target.format, &properties);
        Require((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0, "depth/stencil format " + std::to_string(target.format) + " cannot be an attachment on this device");
        Require((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0, "depth/stencil format " + std::to_string(target.format) + " cannot be sampled on this device");
        Require(target.extent.width <= context.limits.maxFramebufferWidth && target.extent.height <= context.limits.maxFramebufferHeight, "depth target exceeds framebuffer limits");
        const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT | (target.stencilAddress != 0 ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
        try {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = target.format;
            info.extent = {target.extent.width, target.extent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &image), "vkCreateImage depth");
            VkMemoryRequirements requirements{};
            context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory depth target");
            Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory depth");
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = target.format;
            viewInfo.subresourceRange = {aspects, 0, 1, 0, 1};
            Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView depth");
            auto* recorder = Recorder::Active();
            std::unique_ptr<CommandBatch> batch;
            if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
            const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
            VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            toGeneral.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.image = image;
            toGeneral.subresourceRange = viewInfo.subresourceRange;
            const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
            barrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
            const VkClearDepthStencilValue clear{target.clearDepth, target.clearStencil};
            context.Function<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage")(commands, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &toGeneral.subresourceRange);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
            if (batch) batch->SubmitAndWait();
            else Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
        } catch (...) {
            release();
            throw;
        }
    }
    ~DepthSurface() { release(); }
    DepthSurface(const DepthSurface&) = delete;

    // The surface as slice 0 of a sampled 2D array: the slices' depth planes are copied into the
    // layers of an array image before every use (`slices[i]` null: the layer is cleared).
    std::shared_ptr<Texture> Layered(const std::vector<DepthSurface*>& slices, VkComponentMapping components) {
        const auto layers = static_cast<std::uint32_t>(slices.size());
        const VkImageSubresourceRange all{VK_IMAGE_ASPECT_DEPTH_BIT | (target.stencilAddress != 0 ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u), 0, 1, 0, layers};
        auto* recorder = Recorder::Active();
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
        if (layeredLayers != layers) {
            if (layeredImage != VK_NULL_HANDLE) retired.push_back({layeredImage, layeredMemory});
            layeredTextures.clear();
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = target.format;
            info.extent = {target.extent.width, target.extent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = layers;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &layeredImage), "vkCreateImage layered depth");
            VkMemoryRequirements requirements{};
            context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, layeredImage, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &layeredMemory), "vkAllocateMemory layered depth");
            Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, layeredImage, layeredMemory, 0), "vkBindImageMemory layered depth");
            layeredLayers = layers;
            VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            toGeneral.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.image = layeredImage;
            toGeneral.subresourceRange = all;
            barrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
        }
        constexpr VkAccessFlags any = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, any, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        const auto copy = context.Function<PFN_vkCmdCopyImage>("vkCmdCopyImage");
        const auto clear = context.Function<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage");
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            if (slices[layer] == nullptr) {
                const VkClearDepthStencilValue value{target.clearDepth, target.clearStencil};
                const VkImageSubresourceRange range{all.aspectMask, 0, 1, layer, 1};
                clear(commands, layeredImage, VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
                continue;
            }
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, layer, 1};
            region.extent = {target.extent.width, target.extent.height, 1};
            copy(commands, slices[layer]->image, VK_IMAGE_LAYOUT_GENERAL, layeredImage, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        }
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, any);
        if (batch) batch->SubmitAndWait();
        else Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
        const std::array<std::uint32_t, 4> key{components.r, components.g, components.b, components.a};
        if (const auto found = layeredTextures.find(key); found != layeredTextures.end()) return found->second;
        auto texture = std::make_shared<Texture>(context, layeredImage, target.format, VK_IMAGE_ASPECT_DEPTH_BIT, components, VK_IMAGE_VIEW_TYPE_2D_ARRAY, layers);
        layeredTextures.emplace(key, texture);
        return texture;
    }

    void Transfer(StorageTexture& storage, bool intoStorage) {
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto format = storage.Format();
        const bool texelsMatch = d16 ? (format == VK_FORMAT_R16_UNORM || format == VK_FORMAT_R16_UINT || format == VK_FORMAT_R16_SINT || format == VK_FORMAT_R16_SFLOAT)
                                     : (format == VK_FORMAT_R32_SFLOAT || format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R32_SINT);
        const auto& resource = storage.Descriptor();
        if (!texelsMatch || resource.width != target.extent.width || resource.height != target.extent.height || resource.baseLevel != 0 || resource.baseArray != 0) {
            char text[256];
            std::snprintf(text, sizeof(text), "AGC graphics: storage image access to depth surface 0x%llx (%ux%u, vk format %d) as a %ux%u image of vk format %d, level %u, slice %u is not implemented",
                          static_cast<unsigned long long>(target.address), target.extent.width, target.extent.height, static_cast<int>(target.format), resource.width, resource.height, static_cast<int>(format), resource.baseLevel, resource.baseArray);
            throw std::runtime_error(text);
        }
        auto* recorder = Recorder::Active();
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(target.extent.width) * target.extent.height * (d16 ? 2u : 4u);
        auto buffer = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(bytes), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const VkBufferImageCopy depthRegion{0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {target.extent.width, target.extent.height, 1}};
        const VkBufferImageCopy colorRegion{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {target.extent.width, target.extent.height, 1}};
        const auto toBuffer = context.Resolved(&DeviceFunctions::cmdCopyImageToBuffer, "vkCmdCopyImageToBuffer");
        const auto fromBuffer = context.Resolved(&DeviceFunctions::cmdCopyBufferToImage, "vkCmdCopyBufferToImage");
        constexpr VkAccessFlags any = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, any, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        if (intoStorage) toBuffer(commands, image, VK_IMAGE_LAYOUT_GENERAL, buffer->Handle(), 1, &depthRegion);
        else toBuffer(commands, storage.Image(), VK_IMAGE_LAYOUT_GENERAL, buffer->Handle(), 1, &colorRegion);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        if (intoStorage) fromBuffer(commands, buffer->Handle(), storage.Image(), VK_IMAGE_LAYOUT_GENERAL, 1, &colorRegion);
        else fromBuffer(commands, buffer->Handle(), image, VK_IMAGE_LAYOUT_GENERAL, 1, &depthRegion);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, any);
        if (batch) {
            batch->SubmitAndWait();
            return;
        }
        recorder->Keep(buffer);
        Recorder::CountBarriers(Recorder::CommandClass::Draw, 3);
    }

    void Clear(const DepthTarget& values, VkImageAspectFlags aspects) {
        auto* recorder = Recorder::Active();
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        constexpr VkPipelineStageFlags users = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        constexpr VkAccessFlags access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        RecordMemoryBarrier(context, commands, users, VK_PIPELINE_STAGE_TRANSFER_BIT, access, VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkClearDepthStencilValue clear{values.clearDepth, values.clearStencil};
        const VkImageSubresourceRange range{aspects, 0, 1, 0, 1};
        context.Function<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage")(commands, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, users, VK_ACCESS_TRANSFER_WRITE_BIT, access);
        if (batch) batch->SubmitAndWait();
        else Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
    }
    DepthSurface& operator=(const DepthSurface&) = delete;

    std::shared_ptr<Texture> Sampled(std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components) {
        std::array<std::uint32_t, 12> key{};
        std::copy_n(words.begin(), std::min<std::size_t>(words.size(), 8), key.begin());
        key[8] = components.r;
        key[9] = components.g;
        key[10] = components.b;
        key[11] = components.a;
        if (const auto found = textures.find(key); found != textures.end()) return found->second;
        const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
        const auto format = ResolveTextureFormat(resource.format);
        const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
        const bool singleSliceArray = resource.dimension == TextureDimension::k2DArray && resource.depthOrLastArray == 0;
        if ((format != expected && !depthBits) || (resource.dimension != TextureDimension::k2D && !singleSliceArray) || resource.width != target.extent.width || resource.height != target.extent.height || resource.baseLevel != 0 || resource.lastLevel != 0 || resource.baseArray != 0) {
            char text[448];
            std::snprintf(text, sizeof(text), "AGC graphics: sampling the %s plane of depth surface 0x%llx (%ux%u, vk format %d) as a %ux%u texture of guest format %u (vk %d), tile mode %u, dimension %d, levels %u-%u, slice %u is not implemented (T# %08x %08x %08x %08x %08x %08x %08x %08x)",
                          stencil ? "stencil" : "depth", static_cast<unsigned long long>(target.address), target.extent.width, target.extent.height, static_cast<int>(target.format), resource.width, resource.height, resource.format, static_cast<int>(format),
                          static_cast<unsigned>(resource.tileMode), static_cast<int>(resource.dimension), resource.baseLevel, resource.lastLevel, resource.baseArray, key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7]);
            throw std::runtime_error(text);
        }
        auto texture = std::make_shared<Texture>(context, image, target.format, stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, components, singleSliceArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
        textures.emplace(key, texture);
        return texture;
    }

    const Context context;
    const DepthTarget target;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;

private:
    std::map<std::array<std::uint32_t, 12>, std::shared_ptr<Texture>> textures;
    std::map<std::array<std::uint32_t, 4>, std::shared_ptr<Texture>> layeredTextures;
    VkImage layeredImage = VK_NULL_HANDLE;
    VkDeviceMemory layeredMemory = VK_NULL_HANDLE;
    std::uint32_t layeredLayers = 0;
    std::vector<std::pair<VkImage, VkDeviceMemory>> retired;

    void release() noexcept {
        textures.clear();
        layeredTextures.clear();
        if (layeredImage != VK_NULL_HANDLE) retired.push_back({layeredImage, layeredMemory});
        for (const auto& [retiredImage, retiredMemory] : retired) {
            context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, retiredImage, nullptr);
            context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, retiredMemory, nullptr);
        }
        retired.clear();
        layeredImage = VK_NULL_HANDLE;
        layeredMemory = VK_NULL_HANDLE;
        if (view) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
        if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
        if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        view = VK_NULL_HANDLE;
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
    }
};

bool sameSurface(const DepthTarget& a, const DepthTarget& b) {
    return a.address == b.address && a.stencilAddress == b.stencilAddress && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format;
}

AgcDriver::Mutex& surfacesMutex() {
    static AgcDriver::Mutex mutex;
    return mutex;
}

std::vector<std::unique_ptr<DepthSurface>>& surfaces() {
    static auto* list = new std::vector<std::unique_ptr<DepthSurface>>();
    return *list;
}

std::atomic<std::uint64_t> surfacesGeneration{1};

struct LastDepthSurface {
    VkDevice device = VK_NULL_HANDLE;
    DepthTarget target{};
    VkImageView view = VK_NULL_HANDLE;
    std::uint64_t generation = 0;
};

}

std::uint64_t DepthSliceBytes(VkExtent2D extent, std::uint32_t bytesPerTexel) {
    const std::uint32_t blockWidth = bytesPerTexel == 4 ? 128u : 256u;
    const std::uint32_t blockHeight = bytesPerTexel == 1 ? 256u : 128u;
    const auto width = static_cast<std::uint64_t>((extent.width + blockWidth - 1) / blockWidth * blockWidth);
    const auto height = static_cast<std::uint64_t>((extent.height + blockHeight - 1) / blockHeight * blockHeight);
    return width * height * bytesPerTexel;
}

VkImageView DepthSurfaceView(const Context& context, const DepthTarget& target) {
    struct LastTag {};
    auto& last = ThreadScratch<LastDepthSurface, LastTag>();
    const auto generation = surfacesGeneration.load(std::memory_order_acquire);
    if (last.view != VK_NULL_HANDLE && last.generation == generation && last.device == context.device && sameSurface(last.target, target)) return last.view;
    std::lock_guard lock(surfacesMutex());
    VkImageView view = VK_NULL_HANDLE;
    for (const auto& surface : surfaces()) {
        if (surface->context.device == context.device && sameSurface(surface->target, target)) {
            view = surface->view;
            break;
        }
    }
    if (view == VK_NULL_HANDLE) {
        surfaces().push_back(std::make_unique<DepthSurface>(context, target));
        view = surfaces().back()->view;
    }
    last = {context.device, target, view, generation};
    return view;
}

void ClearDepthSurface(const Context& context, const DepthTarget& target, bool depth, bool stencil) {
    std::lock_guard lock(surfacesMutex());
    for (const auto& surface : surfaces()) {
        if (surface->context.device != context.device || !sameSurface(surface->target, target)) continue;
        surface->Clear(target, (depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0u) | (stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u));
        return;
    }
    surfaces().push_back(std::make_unique<DepthSurface>(context, target));
}

void TransferDepthSurface(const Context& context, std::uint64_t address, StorageTexture& storage, bool intoStorage) {
    std::lock_guard lock(surfacesMutex());
    const auto& list = surfaces();
    const auto found = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) { return surface->context.device == context.device && (surface->target.address == address || surface->target.stencilAddress == address); });
    if (found == list.rend()) return;
    if ((*found)->target.address != address) {
        char text[112];
        std::snprintf(text, sizeof(text), "AGC graphics: storage image access to the stencil plane 0x%llx is not implemented", static_cast<unsigned long long>(address));
        throw std::runtime_error(text);
    }
    (*found)->Transfer(storage, intoStorage);
}

void ClearDepthSurfaces(VkDevice device) {
    std::lock_guard lock(surfacesMutex());
    surfacesGeneration.fetch_add(1, std::memory_order_release);
    std::erase_if(surfaces(), [&](const auto& surface) { return surface->context.device == device; });
}

std::shared_ptr<Texture> DepthSurfaceTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components) {
    if (resource.tileMode != TextureTileMode::kZ64KBX) return nullptr;
    std::lock_guard lock(surfacesMutex());
    const auto& list = surfaces();
    if (resource.dimension == TextureDimension::k2DArray && resource.depthOrLastArray != 0) {
        const auto first = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) { return surface->context.device == context.device && surface->target.address == resource.baseAddress; });
        if (first == list.rend()) return nullptr;
        const auto& target = (*first)->target;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const bool depthBits = words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
        if ((ResolveTextureFormat(resource.format) != (d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT) && !depthBits) || resource.width != target.extent.width || resource.height != target.extent.height || resource.baseLevel != 0 || resource.lastLevel != 0 || resource.baseArray != 0) {
            char text[256];
            std::snprintf(text, sizeof(text), "AGC graphics: sampling depth surface 0x%llx (%ux%u, vk format %d) as a %ux%u array of %u slices of guest format %u, levels %u-%u, base slice %u is not implemented",
                          static_cast<unsigned long long>(target.address), target.extent.width, target.extent.height, static_cast<int>(target.format), resource.width, resource.height, resource.depthOrLastArray + 1u, resource.format, resource.baseLevel, resource.lastLevel, resource.baseArray);
            throw std::runtime_error(text);
        }
        const auto sliceBytes = DepthSliceBytes(target.extent, d16 ? 2u : 4u);
        std::vector<DepthSurface*> slices(static_cast<std::size_t>(resource.depthOrLastArray) + 1u, nullptr);
        for (std::size_t slice = 0; slice < slices.size(); ++slice) {
            const auto address = target.address + slice * sliceBytes;
            const auto found = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) { return surface->context.device == context.device && surface->target.address == address && surface->target.extent.width == target.extent.width && surface->target.extent.height == target.extent.height && surface->target.format == target.format; });
            if (found != list.rend()) slices[slice] = found->get();
        }
        return (*first)->Layered(slices, components);
    }
    const auto found = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) {
        return surface->context.device == context.device && (surface->target.address == resource.baseAddress || (surface->target.stencilAddress != 0 && surface->target.stencilAddress == resource.baseAddress));
    });
    return found == list.rend() ? nullptr : (*found)->Sampled(words, resource, components);
}

bool DepthSurfaceAt(std::uint64_t address) {
    std::lock_guard lock(surfacesMutex());
    return std::any_of(surfaces().begin(), surfaces().end(), [&](const auto& surface) { return surface->target.address == address || surface->target.stencilAddress == address; });
}

}
