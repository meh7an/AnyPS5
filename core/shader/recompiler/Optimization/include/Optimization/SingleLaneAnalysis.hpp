#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_SINGLELANEANALYSIS_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_SINGLELANEANALYSIS_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"

namespace ShaderRecompiler {

struct SingleLaneVerdict {
    bool eligible = false;
    // What disqualified the program (a static string) and the value it was found at, if any.
    const char* reason = nullptr;
    const IrValue* at = nullptr;
};

// Whether a wave64 compute program gives every lane the same results at one guest lane per
// invocation on a 32-wide host subgroup (SpirvEmitter's laneCount 1) as at two (laneCount 2, the
// whole wave in one subgroup). At one lane each subgroup holds half a wave and runs it as a wave
// whose lanes 32-63 are inactive: ballots fill bits 0-31 and lane ids are 0-31. A workgroup of at
// most 32 lanes is one subgroup either way and always qualifies. Otherwise no lane can tell when:
// - no lane reads another's value by index (readlane and readfirstlane only of uniform values; no
//   writelane, permutes, DPP, swizzles, append/consume, ADD_TID or GPU-selected buffers);
// - lane masks are only combined bit by bit and read one bit per lane (the lane's own bit, the
//   same expression over both 32-bit halves), never as numbers;
// - LDS is lane-private: every access is one affine function of the local invocation id whose
//   footprints no two lanes share, so no lane needs another's lockstep, and the workgroup is more
//   than one wave when it writes LDS (one-wave groups get workgroup barriers per wave);
// - a VCC branch decides the same for every active lane: each term of its predicate is uniform or
//   a term of the exec mask at the branch, so half a wave's "any lane" agrees with the wave's;
// - barriers do not share a program with exec branches, which a half without active lanes skips.
// Exec branches otherwise only skip blocks where a half has no active lane.
[[nodiscard]] SingleLaneVerdict AnalyzeSingleLane(const IrProgram& program, const ShaderWorkgroupInputInfo& workgroup);

}

#endif
