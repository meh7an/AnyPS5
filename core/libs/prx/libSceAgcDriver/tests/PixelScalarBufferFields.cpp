#include "BdaAbi.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <spirv/unified1/spirv.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace ShaderRecompiler;

constexpr std::array<std::uint32_t, 4> AddTidBuffer{0x10000000u, 0u, 0x1000u, 0x31816facu};
constexpr std::array<std::uint32_t, 4> UnsetBuffer{0x10000000u, 0x40000002u, 1u, 0xffffffffu};

constexpr std::array<std::uint32_t, 7> ScalarLoad{
    0xf4200100u, 0xfa000000u,
    0xbf8cc07fu,
    0x7e000204u,
    0xf800180fu, 0x00000000u,
    0xbf810000u,
};

constexpr std::array<std::uint32_t, 6> VectorLoad{
    0xe0300000u, 0x80000000u,
    0xbf8c3f70u,
    0xf800180fu, 0x00000000u,
    0xbf810000u,
};

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

SpirvTarget Target() {
    static constexpr std::array<std::uint32_t, 3> capabilities{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess,
    };
    static constexpr std::array<std::string_view, 4> extensions{
        "SPV_EXT_descriptor_indexing", "SPV_KHR_8bit_storage", "SPV_KHR_float_controls", "SPV_KHR_physical_storage_buffer",
    };
    return {0x00401000u, 0x00010300u, 32, BdaAbi::Version, capabilities, extensions, false,
            {1024, 1024, 64}, 1024, 32768, {}, {}};
}

RecompileResult RecompilePixel(std::span<const std::uint32_t> code, std::span<const std::uint32_t> buffer) {
    ShaderPixelStageInfo pixel{};
    pixel.interpolatorCount = 0;
    pixel.inputAddr = 0x2;
    pixel.hasPerspectiveCenterVgpr = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    RecompileRequest request{};
    request.shader = {ShaderStage::Fragment, 0x30000u, code, 0, {}};
    request.context.waveSize = 64;
    request.context.userDataBaseRegister = 0;
    request.context.userData = buffer;
    request.context.pixel = pixel;
    request.target = Target();
    request.layout.pushConstantSizeBytes = 128;
    request.useCache = false;
    return Recompile(request);
}

std::span<const std::uint32_t> BoundBuffer(const RecompileResult& result) {
    for (const auto& binding : result.bindings) {
        if (binding.role == DescriptorRole::GuestBuffers && binding.guestDescriptor.size() >= 4u) return std::span(binding.guestDescriptor).first(4u);
    }
    throw std::runtime_error("no guest buffer binding");
}

void RequireRefused(std::span<const std::uint32_t> code, std::span<const std::uint32_t> buffer, std::string_view reason) {
    try {
        static_cast<void>(RecompilePixel(code, buffer));
    } catch (const std::runtime_error& error) {
        Check(std::string_view(error.what()).find(reason) != std::string_view::npos, std::string("unexpected error: ") + error.what());
        return;
    }
    throw std::runtime_error("a per-lane load recompiled, expected: " + std::string(reason));
}

}

int main() {
    try {
        static_cast<void>(RecompilePixel(ScalarLoad, AddTidBuffer));
        const auto unset = RecompilePixel(ScalarLoad, UnsetBuffer);
        const auto bound = BoundBuffer(unset);
        Check(bound[0] == UnsetBuffer[0] && bound[2] == UnsetBuffer[2], "an unset V# read by a scalar load lost its base or size");
        Check(bound[1] == (UnsetBuffer[1] & ~0x40000000u) && bound[3] == (UnsetBuffer[3] & ~0xc0000000u), "an unset V# read by a scalar load kept its cache swizzle or type bits");
        RequireRefused(VectorLoad, AddTidBuffer, "buffer ADD_TID is only valid for compute shaders");
        RequireRefused(VectorLoad, UnsetBuffer, "buffer descriptor uses an unsupported type");
        std::printf("pixel scalar V# tests passed\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
