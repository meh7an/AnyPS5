#ifndef CORE_SHADER_RECOMPILER_OPTIMIZATION_RESOURCEPROGRAM_HPP
#define CORE_SHADER_RECOMPILER_OPTIMIZATION_RESOURCEPROGRAM_HPP

#include "Recompiler.hpp"
#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include <cstdint>
#include <memory>

namespace ShaderRecompiler {

[[nodiscard]] IrProgram PrepareResourceProgram(const RecompileRequest& request);
[[nodiscard]] std::shared_ptr<const IrResourcePlan> GetResourcePlan(const RecompileRequest& request);

// What a driver's capture of a request produces: the plan and the materialization of the words the
// runtime read through it. Recompile(request, capture) compiles from these without a second walk.
struct SourceEntry;
struct ResourceCapture {
    std::shared_ptr<const IrResourcePlan> plan;
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
    // The cache entry the plan belongs to; null when the request bypasses the cache.
    std::shared_ptr<SourceEntry> source;
    // The walk's read addresses when the plan has pure flat slots (the leaf of each pure slot,
    // and every other read sorted and deduplicated); empty otherwise.
    SrtReadTrace readTrace;
    // APS5_PROFILE_DRAW: what CaptureResources spent resolving the source (the stage inputs, the
    // key over the code, the plan) before the walk.
    std::uint64_t sourceNanoseconds = 0;
};
[[nodiscard]] std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime);

// The resolved source of a request (its cache entry with the plan built), for a driver that
// memoizes it per registered shader: ResolveSource is what CaptureResources does before the walk
// (the stage input validation, the key over the code, the lookup, the first-sight plan build), and
// the overload below captures over a handle without repeating it. A handle stays valid for every
// request with the same code and the same cache key fields (RecompileCacheKey::ContextHash plus
// the target); the vertex stages' input validation is repeated per capture because it reads V#
// fields the key does not cover. Null for a request that bypasses the cache (useCache false).
struct SourceHandle {
    std::shared_ptr<SourceEntry> source;
};
[[nodiscard]] std::shared_ptr<const SourceHandle> ResolveSource(const RecompileRequest& request);
[[nodiscard]] std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime, const SourceHandle& handle);

// The user data a capture over a source and its results depend on, for a driver memo keyed over
// the user data (the AGC driver's stage memo). Per user word, `keepBits` are the bits the walk or
// the specialization can read: a pointer, a value the walk computes with, a sampler, a descriptor
// word other than an inline buffer's or image's base, anything the analysis does not recognize.
// The other bits reach a result only as copies (an inline V#'s or T#'s base, push or shader data),
// which UserDataPatchesFor names. A key keeps those bits of each word and, for each inline V# in
// `bufferBases` (the user words of its words 0 and 1), whether its base is zero: an empty buffer
// specializes differently. Null when the source has no plan.
struct UserDataKey {
    std::vector<std::uint32_t> keepBits;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bufferBases;
};
[[nodiscard]] std::shared_ptr<const UserDataKey> UserDataKeyFor(const SourceHandle& handle);
// A result word that is a copy of a user word: dword `word` of the guest descriptor of binding
// `binding` (an index into RecompileResult::bindings), or of the push constants (PushConstants).
struct UserDataPatch {
    static constexpr std::uint32_t PushConstants = 0xffffffffu;
    std::uint32_t binding = 0;
    std::uint32_t word = 0;
    std::uint32_t userWord = 0;
};
// The user data copies in a result materialized over `handle`'s source: every result of one variant
// has them in the same places, so they are made once per variant. Null when the result's variant is
// not one of the source's.
[[nodiscard]] std::shared_ptr<const std::vector<UserDataPatch>> UserDataPatchesFor(const SourceHandle& handle, const RecompileResult& result);
// Makes `result`, a result of the request's source and variant materialized over user data that
// differs from the request's only where `patches` copy it, the request's: its user words copied in
// and its vertex attributes resolved from the request's V#s, as Recompile(request, capture) would
// materialize them. The vertex family's stage inputs are validated as a capture validates them.
void PatchOverUserData(const RecompileRequest& request, RecompileResult& result, const std::vector<UserDataPatch>& patches);

}

#endif
