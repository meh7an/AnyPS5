#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTTAPE_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTTAPE_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/SrtWalker.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace ShaderRecompiler::Detail {

// A resource plan's runtime walk compiled once into a flat node array: the descriptor sources, the
// flat SRT slots and the control-flow conditions EvaluateRuntimeSourcesImpl evaluates on every
// capture, with the Evaluator's semantics (the main, clean and read-first-lane contexts, values
// computed once per context, lazy evaluation, the read trace) but without its per-value pointer
// hashing, Resolve chains and opcode dispatch on the IR. A node is an IR value in one context; the
// structural decisions the Evaluator takes on every evaluation (resolving invariant phis, the
// extract patterns, raw-read classification, clean-slot routing) are taken once here.
// APS5_NO_SRT_TAPE=1 evaluates with the Evaluator as before; APS5_VERIFY_SRT_TAPE=1 runs both and
// reports any difference (the Evaluator's answer is used then).
class SrtTape {
public:
    // The plan's tape, compiled on first use and kept by the plan; null when the plan holds
    // something the tape does not model (a cycle, a malformed instruction), or when disabled.
    static const SrtTape* For(const IrResourcePlan& plan);
    static bool Enabled();
    static bool Verified();

    // EvaluateRuntimeSourcesImpl over the plan's materialization sources and clean flat slots with
    // the flat walk. False when any evaluation the Evaluator would fail fails here (the caller then
    // runs the Evaluator, which fails the same way and says why); the read trace is left as it was.
    bool Run(const IrResourcePlan& plan, const SrtRuntime& runtime, std::vector<DescriptorValue>& results, std::vector<std::uint32_t>& flat, std::vector<std::uint8_t>& activeSources) const;

    std::size_t Nodes() const { return nodes.size(); }

private:
    friend class SrtTapeCompiler;
    static std::shared_ptr<const SrtTape> Compile(const IrResourcePlan& plan);
    enum class Op : std::uint8_t {
        Fail, Const, UserData, ShaderBase, Read, ReadConstBuffer, ExtractHalf, AddCarry, ConstructU64,
        ConvertF32U32, ConvertU32F32, FPMul32, FPTrunc32, FPIsNan32, FPOrdLessThanEqual32, FPOrdGreaterThanEqual32,
        IAdd32, IAdd64, ISub32, ISub64, IMul32, IMul64, UMin32, BitwiseAnd32, BitwiseAnd64, BitwiseOr32, BitwiseXor32, BitwiseNot32,
        ShiftLeftLogical32, ShiftLeftLogical64, ShiftRightLogical32, ShiftRightLogical64, ShiftRightArithmetic32, ShiftRightArithmetic64,
        BitFieldUExtract, BitFieldSExtract, BitFieldInsert, Select, IEqual32, INotEqual32, ULessThan32, UGreaterThan32,
        LogicalAnd, LogicalOr, LogicalXor, LogicalNot
    };
    static constexpr std::uint32_t NoNode = 0xffffffffu;
    struct Node {
        Op op = Op::Fail;
        // Reads through the runtime's specialization reader (the clean context and its nested ones).
        bool clean = false;
        // ExtractHalf / AddCarry: which 32-bit half.
        std::uint8_t component = 0;
        std::uint8_t argumentCount = 0;
        std::array<std::uint32_t, 5> arguments{};
        // Const: the value; UserData: the user data word; the reads: the memory info's offset.
        std::uint64_t immediate = 0;
        // The reads: the read instruction, the read trace's leaf identity.
        const IrValue* source = nullptr;
    };
    struct Frame;
    bool evaluate(Frame& frame, std::uint32_t index, std::uint64_t& result) const;
    bool compute(Frame& frame, const Node& node, std::uint64_t& result) const;

    std::vector<Node> nodes;
    // Per descriptor source, its eight dword roots (NoNode past dwordCount).
    std::vector<std::array<std::uint32_t, 8>> sourceDwords;
    // Per srtReads entry, its root in the context the flat loop evaluates it in.
    std::vector<std::uint32_t> flatRoots;
    // Per control-flow block, its condition's root in the clean context (NoNode without one).
    std::vector<std::uint32_t> conditionRoots;
};

}

#endif
