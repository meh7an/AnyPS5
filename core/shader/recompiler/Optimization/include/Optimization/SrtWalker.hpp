#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_SRTWALKER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_SRTWALKER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

using SrtMemoryReader = bool (*)(void* userData, std::uint64_t address, std::uint32_t* value);

// The guest addresses a walk dereferenced (Detail::Evaluator::EvaluateRawRead): the leaf read of
// each pure flat slot (IrResourcePlan::pureFlatSlots) as (flat offset, address), set by the
// flattened loop through `leaf`/`leafSlot` around that slot's evaluation, and every other raw read
// address. With `copyNodes` (the raw read nodes of the SRT slots a driver patches as memory copies,
// sorted by node, with their slots) also each such read's (slot, address) in `copies`.
struct SrtReadTrace {
    using CopyNodes = std::vector<std::pair<const IrValue*, std::uint32_t>>;
    const IrValue* leaf = nullptr;
    std::uint32_t leafSlot = 0;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> leaves;
    std::vector<std::uint64_t> otherReads;
    std::shared_ptr<const CopyNodes> copyNodes;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> copies;

    // One raw read of `node` at `address`.
    void NoteRead(const IrValue* node, std::uint64_t address) {
        if (node == leaf) leaves.emplace_back(leafSlot, address);
        else otherReads.push_back(address);
        if (copyNodes == nullptr || copyNodes->empty()) return;
        const auto found = std::lower_bound(copyNodes->begin(), copyNodes->end(), node, [](const CopyNodes::value_type& entry, const IrValue* value) { return std::less<const IrValue*>{}(entry.first, value); });
        if (found != copyNodes->end() && found->first == node) copies.emplace_back(found->second, address);
    }
};

struct SrtRuntime {
    std::span<const std::uint32_t> userData;
    std::uint64_t shaderBase = 0;
    SrtMemoryReader readMemory = nullptr;
    void* userContext = nullptr;
    SrtMemoryReader readSpecializationMemory = nullptr;
    SrtReadTrace* readTrace = nullptr;
};

enum class RuntimeValueType {
    Any,
    Integer
};

class SrtWalker {
public:
    void BuildPlan(IrProgram& program) const;
    [[nodiscard]] bool ValidateRuntimeValue(const IrResourcePlan& program, const IrValue* value, RuntimeValueType type = RuntimeValueType::Any) const;
    void EvaluateUniformValues(const IrResourcePlan& program, std::span<IrValue* const> values, const SrtRuntime& runtime, std::span<std::uint32_t> results) const;
    void EvaluateDescriptorSource(const IrResourcePlan& program, std::uint32_t source, const SrtRuntime& runtime, DescriptorValue& result) const;
    void EvaluateDescriptorSources(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results) const;
    void EvaluateRuntimeSources(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results, std::vector<std::uint32_t>& flat, std::span<const std::uint8_t> cleanFlatSlots, std::vector<std::uint8_t>& activeSources) const;
    void Walk(const IrResourcePlan& program, const SrtRuntime& runtime, std::vector<std::uint32_t>& flat) const;

};

}

#endif
