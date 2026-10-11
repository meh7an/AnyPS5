#include "IntermediateRepresentation/IrProgram.hpp"
#include "IntermediateRepresentation/IrMetadata/StageIO.hpp"
#include "Optimization/ConstantFolder.hpp"
#include "Optimization/SingleLaneAnalysis.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <initializer_list>

using namespace ShaderRecompiler;

namespace {

struct Builder {
    IrProgram program;
    IrBlock* block = nullptr;
    ShaderWorkgroupInputInfo workgroup;

    explicit Builder(std::uint32_t width = 16u, std::uint32_t height = 16u) {
        block = &program.CreateBlock();
        program.BlockOrder().push_back(block);
        program.SetEntryBlock(*block);
        BlockInfo info;
        info.id = 0;
        program.Metadata().blockInfo.push_back(info);
        program.Resources().stage = IrShaderStage::Compute;
        program.SetWaveSize(64u);
        workgroup.threadsNum[0] = width;
        workgroup.threadsNum[1] = height;
        workgroup.threadsNum[2] = 1u;
    }

    IrValue& constant(std::uint32_t value) {
        auto& created = program.CreateValue(IrOpcode::Void, IrType::U32);
        created.SetImmediateU32(value);
        return created;
    }

    IrValue& boolean(bool value) {
        auto& created = program.CreateValue(IrOpcode::Void, IrType::Bool);
        created.SetImmediateBool(value);
        return created;
    }

    IrValue& emit(IrOpcode opcode, IrType type, std::initializer_list<IrValue*> arguments) {
        auto& created = program.CreateValue(opcode, type);
        for (auto* argument : arguments) created.AddArgument(argument);
        block->AppendInstruction(&created);
        return created;
    }

    IrValue& binary(IrOpcode opcode, IrValue& left, IrValue& right) {
        return emit(opcode, IrType::U32, {&left, &right});
    }

    IrValue& lane() {
        return emit(IrOpcode::LaneId, IrType::U32, {});
    }

    IrValue& localId(std::uint32_t axis) {
        return emit(IrOpcode::GetBuiltin, IrType::U32, {&constant(static_cast<std::uint32_t>(StageInputKind::LocalInvocationId)), &constant(axis)});
    }

    IrValue& userData() {
        return emit(IrOpcode::GetUserData, IrType::U32, {});
    }

    // The flat index 16 * y + x, as a byte offset of a dword per lane.
    IrValue& laneDword() {
        auto& row = binary(IrOpcode::IMul32, constant(16u), emit(IrOpcode::BitFieldUExtract, IrType::U32, {&localId(1), &constant(0u), &constant(24u)}));
        return binary(IrOpcode::ShiftLeftLogical32, binary(IrOpcode::IAdd32, row, localId(0)), constant(2u));
    }

    IrValue& ballot(IrValue& predicate, std::uint32_t half) {
        auto& mask = emit(IrOpcode::Ballot, IrType::U32x4, {&predicate});
        return emit(IrOpcode::CompositeExtractU32x4, IrType::U32, {&mask, &constant(half)});
    }

    IrValue& threadBit(IrValue& low, IrValue& high) {
        auto& laneId = lane();
        auto& word = emit(IrOpcode::SelectU32, IrType::U32, {&emit(IrOpcode::ULessThan32, IrType::Bool, {&laneId, &constant(32u)}), &low, &high});
        auto& shift = binary(IrOpcode::BitwiseAnd32, laneId, constant(31u));
        auto& bit = binary(IrOpcode::BitwiseAnd32, binary(IrOpcode::ShiftRightLogical32, word, shift), constant(1u));
        return emit(IrOpcode::INotEqual32, IrType::Bool, {&bit, &constant(0u)});
    }

    IrValue& lds(IrOpcode opcode, IrType type, std::initializer_list<IrValue*> arguments, std::uint32_t offset) {
        MemoryInfo memory;
        memory.kind = ResourceKind::Lds;
        memory.offset = offset;
        const auto index = static_cast<std::uint32_t>(program.Resources().memoryInfo.size());
        program.Resources().memoryInfo.push_back(memory);
        auto& created = emit(opcode, type, arguments);
        created.SetFlags(MemoryFlags{index, 0u});
        return created;
    }

    void keep(IrValue& value) {
        (void)emit(value.Type() == IrType::Bool ? IrOpcode::Reference : IrOpcode::ReferenceU32, IrType::Void, {&value});
    }

    void branch(BranchCondition condition, IrValue& predicate, IrValue* exec) {
        auto& info = program.Metadata().blockInfo.front();
        info.terminator.kind = TerminatorKind::ConditionalBranch;
        info.terminator.condition = condition;
        info.condition = &predicate;
        info.exec = exec;
    }

    void fold() {
        ConstantFolder{}.Fold(program);
    }

    SingleLaneVerdict analyze() const {
        return AnalyzeSingleLane(program, workgroup);
    }
};

bool folds(const IrValue& value, std::uint32_t immediate) {
    const IrValue* resolved = value.Resolve();
    return resolved->HasImmediate() && resolved->Type() == IrType::U32 && resolved->ImmediateU32() == immediate;
}

bool eligible(const SingleLaneVerdict& verdict) {
    if (!verdict.eligible) std::fprintf(stderr, "  rejected: %s\n", verdict.reason);
    return verdict.eligible;
}

bool rejected(const SingleLaneVerdict& verdict, const char* reason) {
    if (verdict.eligible) {
        std::fprintf(stderr, "  accepted, expected \"%s\"\n", reason);
        return false;
    }
    if (std::strstr(verdict.reason, reason) != nullptr) return true;
    std::fprintf(stderr, "  rejected with \"%s\", expected \"%s\"\n", verdict.reason, reason);
    return false;
}

bool run(const char* name, const std::function<bool()>& test) {
    try {
        if (test()) return true;
        std::fprintf(stderr, "%s\n", name);
        return false;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s: %s\n", name, error.what());
        return false;
    }
}

}

int main() {
    bool passed = true;

    passed &= run("a word ORed with its complement folds to all ones", [] {
        Builder b;
        auto& x = b.localId(0);
        auto& ones = b.binary(IrOpcode::BitwiseOr32, b.emit(IrOpcode::BitwiseNot32, IrType::U32, {&x}), x);
        b.keep(ones);
        b.fold();
        return folds(ones, 0xffffffffu);
    });

    passed &= run("bitwise ops on a word and itself, its complement or an absorbing constant fold", [] {
        Builder b;
        auto& x = b.localId(0);
        auto& clear = b.binary(IrOpcode::BitwiseAnd32, x, b.emit(IrOpcode::BitwiseNot32, IrType::U32, {&x}));
        auto& same = b.binary(IrOpcode::BitwiseAnd32, x, x);
        auto& zero = b.binary(IrOpcode::BitwiseXor32, x, x);
        auto& ones = b.binary(IrOpcode::BitwiseOr32, x, b.constant(0xffffffffu));
        auto& none = b.binary(IrOpcode::BitwiseAnd32, b.constant(0u), x);
        for (auto* value : {&clear, &same, &zero, &ones, &none}) b.keep(*value);
        b.fold();
        return folds(clear, 0u) && same.Resolve() == &x && folds(zero, 0u) && folds(ones, 0xffffffffu) && folds(none, 0u);
    });

    passed &= run("the low bit of all ones shifted by a masked amount folds to one, zero shifted to zero", [] {
        Builder b;
        auto& amount = b.localId(0);
        auto& masked = b.binary(IrOpcode::BitwiseAnd32, amount, b.constant(31u));
        auto& bit = b.binary(IrOpcode::BitwiseAnd32, b.binary(IrOpcode::ShiftRightLogical32, b.constant(0xffffffffu), masked), b.constant(1u));
        // SPIR-V leaves a shift by 32 or more undefined: an unmasked amount keeps the bit test.
        auto& unmasked = b.binary(IrOpcode::BitwiseAnd32, b.binary(IrOpcode::ShiftRightLogical32, b.constant(0xffffffffu), amount), b.constant(1u));
        auto& zero = b.binary(IrOpcode::ShiftRightLogical32, b.constant(0u), amount);
        b.keep(bit);
        b.keep(unmasked);
        b.keep(zero);
        b.fold();
        return folds(bit, 1u) && !unmasked.Resolve()->HasImmediate() && folds(zero, 0u);
    });

    passed &= run("a select on a lane's bit of ~exec | exec folds to its true arm", [] {
        Builder b;
        auto& low = b.ballot(b.boolean(true), 0u);
        auto& high = b.ballot(b.boolean(true), 1u);
        auto& allLow = b.binary(IrOpcode::BitwiseOr32, b.emit(IrOpcode::BitwiseNot32, IrType::U32, {&low}), low);
        auto& allHigh = b.binary(IrOpcode::BitwiseOr32, b.emit(IrOpcode::BitwiseNot32, IrType::U32, {&high}), high);
        auto& active = b.localId(0);
        auto& selected = b.emit(IrOpcode::SelectU32, IrType::U32, {&b.threadBit(allLow, allHigh), &active, &b.constant(0u)});
        b.keep(selected);
        b.fold();
        return selected.Resolve() == &active;
    });

    passed &= run("a lane-private LDS array (tid * 4 + k * 1024) runs at one lane per invocation", [] {
        Builder b;
        auto& address = b.laneDword();
        auto& value = b.localId(0);
        (void)b.lds(IrOpcode::WriteSharedU32, IrType::Void, {&address, &value, &b.boolean(true)}, 0u);
        (void)b.lds(IrOpcode::WriteSharedU32, IrType::Void, {&address, &value, &b.boolean(true)}, 1024u);
        b.keep(b.lds(IrOpcode::LoadSharedU32, IrType::U32, {&address, &b.boolean(true)}, 1024u));
        return eligible(b.analyze());
    });

    passed &= run("a two-dimensional private LDS layout (4x + 1024y) runs at one lane", [] {
        Builder b;
        auto& address = b.binary(IrOpcode::IAdd32, b.binary(IrOpcode::ShiftLeftLogical32, b.localId(0), b.constant(2u)), b.binary(IrOpcode::ShiftLeftLogical32, b.localId(1), b.constant(10u)));
        (void)b.lds(IrOpcode::WriteSharedU32, IrType::Void, {&address, &b.localId(0), &b.boolean(true)}, 0u);
        return eligible(b.analyze());
    });

    passed &= run("lanes of one column sharing an LDS word keep two lanes", [] {
        Builder b;
        auto& address = b.binary(IrOpcode::ShiftLeftLogical32, b.localId(0), b.constant(2u));
        (void)b.lds(IrOpcode::WriteSharedU32, IrType::Void, {&address, &b.localId(1), &b.boolean(true)}, 0u);
        return rejected(b.analyze(), "share LDS");
    });

    passed &= run("an LDS read reaching the next lane's dword keeps two lanes", [] {
        Builder b;
        auto& address = b.laneDword();
        (void)b.lds(IrOpcode::WriteSharedU32, IrType::Void, {&address, &b.localId(0), &b.boolean(true)}, 0u);
        b.keep(b.lds(IrOpcode::LoadSharedU32x2, IrType::U32x2, {&address, &b.boolean(true)}, 0u));
        return rejected(b.analyze(), "share LDS");
    });

    passed &= run("LDS addressed by a loaded value keeps two lanes", [] {
        Builder b;
        auto& address = b.userData();
        (void)b.lds(IrOpcode::WriteSharedU32, IrType::Void, {&address, &b.localId(0), &b.boolean(true)}, 0u);
        return rejected(b.analyze(), "not affine");
    });

    passed &= run("LDS writes in a one-wave workgroup keep two lanes", [] {
        Builder b(8u, 8u);
        auto& address = b.binary(IrOpcode::ShiftLeftLogical32, b.binary(IrOpcode::IAdd32, b.binary(IrOpcode::ShiftLeftLogical32, b.localId(1), b.constant(3u)), b.localId(0)), b.constant(2u));
        (void)b.lds(IrOpcode::WriteSharedU32, IrType::Void, {&address, &b.localId(0), &b.boolean(true)}, 0u);
        return rejected(b.analyze(), "one-wave");
    });

    passed &= run("a lane's own ballot bit runs at one lane", [] {
        Builder b;
        auto& predicate = b.emit(IrOpcode::ULessThan32, IrType::Bool, {&b.localId(0), &b.constant(5u)});
        auto& ballot = b.emit(IrOpcode::Ballot, IrType::U32x4, {&predicate});
        auto& low = b.emit(IrOpcode::CompositeExtractU32x4, IrType::U32, {&ballot, &b.constant(0u)});
        auto& high = b.emit(IrOpcode::CompositeExtractU32x4, IrType::U32, {&ballot, &b.constant(1u)});
        auto& mask = b.emit(IrOpcode::BitwiseNot32, IrType::U32, {&low});
        auto& maskHigh = b.emit(IrOpcode::BitwiseNot32, IrType::U32, {&high});
        b.keep(b.emit(IrOpcode::SelectU32, IrType::U32, {&b.threadBit(mask, maskHigh), &b.constant(1u), &b.constant(0u)}));
        return eligible(b.analyze());
    });

    passed &= run("a counted ballot keeps two lanes", [] {
        Builder b;
        auto& low = b.ballot(b.emit(IrOpcode::ULessThan32, IrType::Bool, {&b.localId(0), &b.constant(5u)}), 0u);
        b.keep(b.emit(IrOpcode::BitCount32, IrType::U32, {&low}));
        return rejected(b.analyze(), "lane mask read as a number");
    });

    passed &= run("a lane mask whose halves are different expressions keeps two lanes", [] {
        Builder b;
        auto& low = b.ballot(b.boolean(true), 0u);
        b.keep(b.emit(IrOpcode::SelectU32, IrType::U32, {&b.threadBit(low, b.constant(0u)), &b.constant(1u), &b.constant(0u)}));
        return rejected(b.analyze(), "halves differ");
    });

    passed &= run("the lane id read as data keeps two lanes", [] {
        Builder b;
        b.keep(b.binary(IrOpcode::IAdd32, b.lane(), b.constant(1u)));
        return rejected(b.analyze(), "lane id read as a number");
    });

    passed &= run("a workgroup of at most 32 lanes runs at one lane even reading the lane id", [] {
        Builder b(16u, 2u);
        b.keep(b.binary(IrOpcode::IAdd32, b.lane(), b.constant(1u)));
        return eligible(b.analyze());
    });

    passed &= run("readfirstlane of user data runs at one lane, of a local id keeps two", [] {
        Builder uniform;
        uniform.keep(uniform.emit(IrOpcode::ReadFirstLane, IrType::U32, {&uniform.userData()}));
        Builder varying;
        varying.keep(varying.emit(IrOpcode::ReadFirstLane, IrType::U32, {&varying.localId(0)}));
        return eligible(uniform.analyze()) && rejected(varying.analyze(), "lane read of a lane-varying value");
    });

    passed &= run("a VCC branch on a uniform compare under the branch's exec runs at one lane", [] {
        Builder b;
        auto& exec = b.emit(IrOpcode::INotEqual32, IrType::Bool, {&b.localId(0), &b.constant(3u)});
        auto& compare = b.emit(IrOpcode::ULessThan32, IrType::Bool, {&b.userData(), &b.constant(5u)});
        auto& vcc = b.emit(IrOpcode::LogicalAnd, IrType::Bool, {&exec, &compare});
        auto& condition = b.emit(IrOpcode::LogicalNot, IrType::Bool, {&vcc});
        b.keep(condition);
        b.keep(exec);
        b.branch(BranchCondition::VccZero, condition, &exec);
        return eligible(b.analyze());
    });

    passed &= run("a VCC branch on a lane compare keeps two lanes", [] {
        Builder b;
        auto& exec = b.emit(IrOpcode::INotEqual32, IrType::Bool, {&b.localId(0), &b.constant(3u)});
        auto& compare = b.emit(IrOpcode::ULessThan32, IrType::Bool, {&b.localId(1), &b.constant(5u)});
        auto& vcc = b.emit(IrOpcode::LogicalAnd, IrType::Bool, {&exec, &compare});
        b.keep(vcc);
        b.keep(exec);
        b.branch(BranchCondition::VccNonZero, vcc, &exec);
        return rejected(b.analyze(), "VCC branch on lane-varying bits");
    });

    passed &= run("a VCC branch whose bits came from a narrower exec keeps two lanes", [] {
        Builder b;
        auto& wide = b.emit(IrOpcode::INotEqual32, IrType::Bool, {&b.localId(0), &b.constant(3u)});
        auto& narrow = b.emit(IrOpcode::LogicalAnd, IrType::Bool, {&wide, &b.emit(IrOpcode::ULessThan32, IrType::Bool, {&b.localId(1), &b.constant(5u)})});
        auto& compare = b.emit(IrOpcode::ULessThan32, IrType::Bool, {&b.userData(), &b.constant(5u)});
        auto& vcc = b.emit(IrOpcode::LogicalAnd, IrType::Bool, {&narrow, &compare});
        b.keep(vcc);
        b.keep(wide);
        b.branch(BranchCondition::VccNonZero, vcc, &wide);
        return rejected(b.analyze(), "VCC branch on lane-varying bits");
    });

    passed &= run("a DPP move keeps two lanes", [] {
        Builder b;
        b.keep(b.emit(IrOpcode::DppMoveU32, IrType::U32, {&b.localId(0)}));
        return rejected(b.analyze(), "operation across lanes");
    });

    passed &= run("a barrier in a program with exec branches keeps two lanes", [] {
        Builder b;
        (void)b.emit(IrOpcode::Barrier, IrType::Void, {});
        auto& exec = b.emit(IrOpcode::INotEqual32, IrType::Bool, {&b.localId(0), &b.constant(3u)});
        auto& skip = b.emit(IrOpcode::LogicalNot, IrType::Bool, {&exec});
        b.keep(skip);
        b.branch(BranchCondition::ExecZero, skip, nullptr);
        return rejected(b.analyze(), "barrier");
    });

    if (!passed) return 1;
    std::printf("single lane tests passed\n");
    return 0;
}
