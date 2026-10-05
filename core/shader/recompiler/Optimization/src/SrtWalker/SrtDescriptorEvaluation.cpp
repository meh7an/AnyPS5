#include "Optimization/SrtWalker/SrtDescriptorEvaluation.hpp"
#include "Optimization/SrtWalker/SrtEvaluator.hpp"
#include "Optimization/SrtWalker/SrtTape.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <string>

namespace ShaderRecompiler::Detail {

namespace {

std::string& failureReason() {
    struct FailureReasonStorage {};
    return HostThreadLocal<std::string, FailureReasonStorage>();
}

std::string DescribeValue(const IrValue* value, std::uint32_t depth) {
    if (value == nullptr) return "null";
    value = value->Resolve();
    std::string text(IrOpcodeName(value->Opcode()));
    if (value->HasImmediate() && value->Type() == IrType::U32) return text + "(" + std::to_string(value->ImmediateU32()) + ")";
    if (depth == 0 || value->ArgumentCount() == 0) return text;
    text += "(";
    for (std::size_t index = 0; index < value->ArgumentCount(); ++index) {
        if (index != 0) text += ", ";
        text += DescribeValue(value->Argument(index), depth - 1);
    }
    return text + ")";
}

bool Fail(std::string reason) {
    failureReason() = std::move(reason);
    return false;
}

const DescriptorSource* Source(const IrResourcePlan& program, std::uint32_t source) {
    if (source >= program.descriptorSources.size()) {
        return nullptr;
    }
    return &program.descriptorSources[source];
}

}

namespace {

bool interpretRuntimeSources(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results, std::vector<std::uint32_t>& flat, bool evaluateFlat, std::span<const std::uint8_t> cleanFlatSlots, std::vector<std::uint8_t>& activeSources);

// APS5_VERIFY_SRT_TAPE: the Evaluator's walk beside the tape's, over a trace of its own; any
// difference in the descriptors, the flat slots, the active sources or the reads traced is reported
// (a few times) and the Evaluator's answer replaces the tape's.
void verifyTape(const IrResourcePlan& program, const SrtRuntime& runtime, std::vector<DescriptorValue>& results, std::vector<std::uint32_t>& flat, std::vector<std::uint8_t>& activeSources, std::size_t leavesBefore, std::size_t readsBefore) {
    SrtReadTrace expectedTrace;
    SrtRuntime interpreted = runtime;
    interpreted.readTrace = runtime.readTrace != nullptr ? &expectedTrace : nullptr;
    std::vector<DescriptorValue> expectedResults;
    std::vector<std::uint32_t> expectedFlat;
    std::vector<std::uint8_t> expectedActive;
    const bool ok = interpretRuntimeSources(program, program.materializationSources, interpreted, expectedResults, expectedFlat, true, program.cleanFlatSlots, expectedActive);
    std::string difference;
    if (!ok) difference = "the Evaluator failed: " + failureReason();
    else if (expectedResults != results) difference = "descriptors";
    else if (expectedFlat != flat) difference = "flat slots";
    else if (expectedActive != activeSources) difference = "active sources";
    else if (auto* trace = runtime.readTrace; trace != nullptr) {
        auto leaves = std::vector(trace->leaves.begin() + static_cast<std::ptrdiff_t>(leavesBefore), trace->leaves.end());
        auto reads = std::vector(trace->otherReads.begin() + static_cast<std::ptrdiff_t>(readsBefore), trace->otherReads.end());
        auto expectedLeaves = expectedTrace.leaves;
        auto expectedReads = expectedTrace.otherReads;
        std::sort(leaves.begin(), leaves.end());
        std::sort(expectedLeaves.begin(), expectedLeaves.end());
        std::sort(reads.begin(), reads.end());
        reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
        std::sort(expectedReads.begin(), expectedReads.end());
        expectedReads.erase(std::unique(expectedReads.begin(), expectedReads.end()), expectedReads.end());
        if (leaves != expectedLeaves) difference = "traced leaf reads";
        else if (reads != expectedReads) difference = "traced reads";
        if (!difference.empty()) {
            trace->leaves.resize(leavesBefore);
            trace->otherReads.resize(readsBefore);
            trace->leaves.insert(trace->leaves.end(), expectedTrace.leaves.begin(), expectedTrace.leaves.end());
            trace->otherReads.insert(trace->otherReads.end(), expectedTrace.otherReads.begin(), expectedTrace.otherReads.end());
        }
    }
    static std::atomic<std::uint64_t> walks {0};
    static std::atomic<std::uint64_t> differences {0};
    const auto walked = walks.fetch_add(1, std::memory_order_relaxed) + 1;
    if (difference.empty()) {
        bool decade = walked >= 10;
        for (auto rest = walked; decade && rest >= 10; rest /= 10) decade = rest % 10 == 0;
        if (decade) std::fprintf(stderr, "[srt-tape] verify: %llu walks, %llu differed\n", static_cast<unsigned long long>(walked), static_cast<unsigned long long>(differences.load(std::memory_order_relaxed)));
        return;
    }
    if (differences.fetch_add(1, std::memory_order_relaxed) < 20) std::fprintf(stderr, "[srt-tape] verify: shader %016llx: the tape's walk differs from the Evaluator's in its %s\n", static_cast<unsigned long long>(program.shaderHash), difference.c_str());
    if (!ok) return;
    results = std::move(expectedResults);
    flat = std::move(expectedFlat);
    activeSources = std::move(expectedActive);
}

}

// The plan's tape (SrtTape) answers the capture walk (the plan's own materialization sources and
// clean flat slots, with the flat walk); anything else, or a walk the tape cannot finish, is the
// Evaluator's.
bool EvaluateRuntimeSourcesImpl(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results, std::vector<std::uint32_t>& flat, bool evaluateFlat, std::span<const std::uint8_t> cleanFlatSlots, std::vector<std::uint8_t>& activeSources) {
    const bool planInputs = evaluateFlat && sources.data() == program.materializationSources.data() && sources.size() == program.materializationSources.size() && cleanFlatSlots.data() == program.cleanFlatSlots.data() && cleanFlatSlots.size() == program.cleanFlatSlots.size();
    if (planInputs) {
        if (const auto* tape = SrtTape::For(program); tape != nullptr) {
            const auto leavesBefore = runtime.readTrace != nullptr ? runtime.readTrace->leaves.size() : 0;
            const auto readsBefore = runtime.readTrace != nullptr ? runtime.readTrace->otherReads.size() : 0;
            if (tape->Run(program, runtime, results, flat, activeSources)) {
                if (SrtTape::Verified()) verifyTape(program, runtime, results, flat, activeSources, leavesBefore, readsBefore);
                return true;
            }
        }
    }
    return interpretRuntimeSources(program, sources, runtime, results, flat, evaluateFlat, cleanFlatSlots, activeSources);
}

namespace {

bool interpretRuntimeSources(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results, std::vector<std::uint32_t>& flat, bool evaluateFlat, std::span<const std::uint8_t> cleanFlatSlots, std::vector<std::uint8_t>& activeSources) {
    failureReason().clear();
    static const bool debug = std::getenv("APS5_SRT_DEBUG") != nullptr;
    if (debug) {
        for (std::size_t slot = 0; slot < program.srtReads.size(); ++slot) std::fprintf(stderr, "[srt] slot %zu = %s"  "\n", slot, DescribeValue(program.srtReads[slot].value, 6).c_str());
    }
    if (!program.srtPlanComplete) {
        return Fail("SRT plan is incomplete");
    }
    if (std::any_of(cleanFlatSlots.begin(), cleanFlatSlots.end(), [](std::uint8_t clean) { return clean != 0u; }) && runtime.readSpecializationMemory == nullptr) {
        return Fail("clean flat slots need specialization memory");
    }
    SrtRuntime cleanRuntime = runtime;
    cleanRuntime.readMemory = runtime.readSpecializationMemory;
    Evaluator cleanEvaluator(program, cleanRuntime);
    Evaluator evaluator(program, runtime, cleanFlatSlots, &cleanEvaluator);
    std::vector<std::uint8_t> active;
    if (evaluateFlat) {
        active.assign(program.descriptorSources.size(), 1u);
    }
    if (evaluateFlat && !program.controlFlow.empty()) {
        for (const auto& block : program.controlFlow) {
            for (const auto source : block.sources) {
                active.at(source) = 0u;
            }
        }
        std::vector<std::uint8_t> visited(program.controlFlow.size());
        std::vector<std::uint32_t> pending {0};
        while (!pending.empty()) {
            const auto index = pending.back();
            pending.pop_back();
            if (visited.at(index)) {
                continue;
            }
            visited[index] = 1u;
            const auto& block = program.controlFlow[index];
            for (const auto source : block.sources) {
                active[source] = 1u;
            }
            std::uint32_t condition = 0;
            const bool cleanEvaluable = block.condition != nullptr && runtime.readSpecializationMemory != nullptr && cleanEvaluator.Evaluate(block.condition, condition);
            if (cleanEvaluable) {
                pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
            } else {
                pending.insert(pending.end(), block.successors.begin(), block.successors.end());
            }
        }
    }
    std::vector<DescriptorValue> evaluated;
    evaluated.reserve(sources.size());
    for (const auto sourceIndex : sources) {
        const auto* source = Source(program, sourceIndex);
        if (source == nullptr) {
            return Fail("descriptor source " + std::to_string(sourceIndex) + " does not exist");
        }
        DescriptorValue value;
        value.dwordCount = source->dwordCount;
        if (!evaluateFlat || active[sourceIndex]) {
            for (std::uint32_t index = 0; index < source->dwordCount; index++) {
                if (!evaluator.Evaluate(source->dwords[index], value.dwords[index])) {
                    std::string detail = DescribeValue(source->dwords[index], 4);
                    const IrValue* dword = source->dwords[index]->Resolve();
                    if (dword->Opcode() == IrOpcode::ReadConst && dword->ArgumentCount() == 2 && dword->Argument(1)->Resolve()->HasImmediate()) {
                        const auto slot = dword->Argument(1)->Resolve()->ImmediateU32();
                        if (slot < program.srtReads.size()) detail += " where slot " + std::to_string(slot) + " = " + DescribeValue(program.srtReads[slot].value, 8);
                    }
                    return Fail("descriptor source " + std::to_string(sourceIndex) + " dword " + std::to_string(index) + ": " + detail);
                }
            }
        }
        evaluated.push_back(value);
    }
    std::vector<std::uint32_t> flattened;
    if (evaluateFlat) {
        flattened.resize(program.srtReads.size());
        for (const auto& read : program.srtReads) {
            const bool clean = read.flatOffset < cleanFlatSlots.size() && cleanFlatSlots[read.flatOffset] != 0u;
            auto& selected = clean ? cleanEvaluator : evaluator;
            // A pure slot's raw read is reachable from no root, so it was not evaluated (nor
            // cached) before this loop: its dereference happens here, once, and is recorded as
            // the slot's leaf; reads nested in its address cone land among the other reads.
            auto* trace = runtime.readTrace;
            const bool pure = trace != nullptr && read.flatOffset < program.pureFlatSlots.size() && program.pureFlatSlots[read.flatOffset] != 0u;
            if (pure) {
                trace->leaf = read.value->Resolve();
                trace->leafSlot = read.flatOffset;
            }
            const bool evaluated = read.flatOffset < flattened.size() && selected.Evaluate(read.value, flattened[read.flatOffset]);
            if (pure) trace->leaf = nullptr;
            if (!evaluated) {
                return Fail(std::string(clean ? "clean " : "") + "SRT read at flat offset " + std::to_string(read.flatOffset) + ": " + DescribeValue(read.value, 4));
            }
        }
    }
    results = std::move(evaluated);
    activeSources = std::move(active);
    if (evaluateFlat) {
        flat = std::move(flattened);
    }
    return true;
}

}

const std::string& RuntimeSourceFailureReason() {
    return failureReason();
}

}
