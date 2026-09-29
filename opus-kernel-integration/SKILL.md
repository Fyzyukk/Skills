---
name: opus-kernel-integration
description: Integrate, tune, validate, or review JIT and precompiled-CO OPUS GEMM/BMM kernels in ROCm/aiter. Use for exact-kid registry and dispatch, architecture ports, split-K/workspace changes, tuner/CSV/production replay, CO artifacts, and compatibility migrations; do not use for generic CK/Triton work or isolated kernel-math optimization.
---

# OPUS Kernel Integration

Deliver the smallest complete vertical slice that preserves shipped behavior. Keep one exact-kid execution path, derive capabilities from the canonical registry, and keep selection policy outside the exact launcher.

## Read the relevant reference

- For tuning, CSV schemas, `--run_config`, production-call replay, full-kid GPU sweeps, M padding, performance reporting, or CI diagnosis, read [references/tuning-validation.md](references/tuning-validation.md) before acting.
- For precompiled `.co` kernels, manifests, offline builds, fingerprints, packaging, or loader changes, read [references/co-artifacts.md](references/co-artifacts.md) before acting.
- For multi-commit PR archaeology, reviewer-finding triage, merge reconciliation, commit planning, or deciding which layer owns a regression, read [references/change-patterns.md](references/change-patterns.md). Its PR #4961 ledger records both retained solutions and approaches that were later removed or superseded.

Do not load a reference that is unrelated to the task.

## Runtime architecture

Keep one execution path:

```text
production caller or tuner selects final kid/split_k
  -> opus_gemm() or opus_bmm()
  -> private family launcher
  -> scalar launch plan when metadata/workspace is needed
  -> checked C++ exact-kid table
  -> generated host launcher
  -> JIT kernel or packaged precompiled image
  -> optional reducer
```

Ownership boundaries:

- `aiter/ops/opus/dispatch.py`: checked public exact-kid router; `aiter/ops/opus/__init__.py` only exposes the public surface.
- `gemm_op_a16w16.py` and `gemm_op_a8w8.py`: flat-package family adaptation, tensor/layout validation, workspace materialization, and one backend call. Do not reorganize these modules merely for aesthetics or force unrelated CK/MoE imports to move.
- `launch_plan.py`: immutable scalar plans for resolved kid, ABI split-K, and minimum workspace metadata.
- `policy.py` and high-level callers: tuned lookup, legacy-ID normalization, heuristic selection, padded-M lookup, and fallback.
- `csrc/opus_gemm/opus_gemm_common.py`: canonical aggregated global-kid registry and per-instance capabilities/workarounds.
- `gen_instances.py`, `codegen/`, architecture generators, and `include/gfx*/`: generated dispatch plus device implementation.
- `opus_gemm.cu`, `opus_bmm.cu`, headers, and pybind: checked raw ABI and exact table invocation.
- `csrc/gemm_a16w16/gemm_a16w16_tune.py` plus `csrc/opus_gemm/*tune*.py`: production multi-backend A16 tuning, OPUS candidate helpers, standalone A8/BMM tuning, and saved-config replay.

Production ownership must not be confused:

| Family | Production policy/caller | Exact execution |
|---|---|---|
| A16W16 GEMM | `aiter/tuned_gemm.py` | `opus_gemm` |
| Plain A8W8 no-scale / 1x128x128 blockscale | No separate OPUS production policy currently; preserve the general boundary in `aiter/ops/gemm_op_a8w8.py` | `opus_gemm` |
| MXFP8 BMM | `aiter/ops/batched_gemm_op_a8w8.py` | `opus_bmm` |

## Non-negotiable contracts

- `opus_gemm()` is logical 2D and `opus_bmm()` is logical batch-first 3D. Adapt legacy physical layouts inside the owning family launcher; do not blur ranks at the public API.
- Treat explicit layouts such as bpreshuffle as tensor-content contracts; shape and stride alone cannot prove them.
- Exact APIs execute the requested registered kid or reject it. They do not read tuned CSVs, run heuristics, silently replace the kid, or fall back to another backend.
- Policy validates saved rows before exact launch. A stale or malformed row may warn and continue to another padded row or the established fallback; it must never reach an incompatible exact launch.
- Resolve architecture and CU information from the input Tensor's device, not the process current device.
- Numeric kid bands reserve identity space only. Registration, family, split-K, workspace, dtype, and shape support come from the exact registry instance.
- Registration and build availability are different facts. Trace registered kid -> requested subset/default floor -> generated table -> linked symbol or packaged image. Runtime diagnostics must distinguish a compiled direct kid, a compiled workspace kid, and a registered-but-unavailable kid.
- The strict exact API never performs legacy sibling redirects. Apply required requested-to-actual normalization only in the owning compatibility or policy boundary, then pass the final kid into the exact launcher.
- A default build needs the representative kids exercised by its default tests and production fallback, not every registered tuning candidate. Keep large candidate sets on demand through tuned rows, sidecars, or explicit build requests.
- Keep JIT kids declared in the ordinary registry. A per-instance compiler workaround belongs in explicit registry metadata, not in a numeric-range check or an unrelated caller.
- Cache scalars and immutable metadata only. Never cache Tensor objects, pointers, streams, allocations, or unbounded shape maps; use a bounded cache when production shapes can vary.
- Workspace is caller-provided or a one-call `torch.empty`. C++ validates and consumes it but never allocates, retains, or globally registers it.
- Resolve/converge ABI split-K before sizing automatic workspace. Validate the resolved launch against K-tile constraints; do not size from an unclamped request. Accept a larger caller-owned workspace only when the existing contract permits it.
- Direct kernels reject workspace. Workspace kernels validate dtype, device, contiguity, and capacity. The reducer owns final output conversion when the main kernel writes partials.
- Generate and test direct-output and `D_OUT=void` workspace specializations independently. A split-K specialization may use an explicit per-instance workaround while the same kid retains its tuned split-one implementation.
- Reuse the existing pybind/C ABI and `module_deepgemm_opus`. Do not add a second A16 fast path, workspace manager, public API, or extension module to solve a local issue.
- A required compatibility wrapper calls its owning private family launcher directly; never route out through the public dispatcher and back into the same family.

## Select the smallest change

| Requested change | Default scope |
|---|---|
| New kid in an existing ABI | Registry declaration, architecture generator/device implementation, generated exact dispatch, focused launch test |
| Existing family on a new architecture | Registry, architecture codegen/device implementation, host dispatch, architecture gate, focused tests |
| New split-K/workspace behavior | Registry metadata, scalar launch plan, owning family launcher, checked C++ entry, kernel/reducer, workspace tests |
| Tuner/config integration | OPUS-owned tuner, registry-derived candidates, CSV/replay tests, relevant production caller |
| New logical tensor/scale ABI | One family tag, one router branch, one private launcher, generated C++ dispatch, contract tests |
| Compatibility migration | One narrow wrapper into the owning private launcher plus behavior-parity coverage |

Do not add a public wrapper, copied candidate table, policy branch, sidecar map, or package reorganization unless the contract truly requires it.

For every kid, declare only applicable facts at its source: unique global id, architecture, family tag, tile/alignment and shape limits, layouts/dtypes, output and bias support, direct/split-K behavior, workspace dtype, compile-time split metadata, and narrowly scoped workaround flags.

## Implementation workflow

1. Inspect `git status`, the merge base, current target `main`, and the branch commit history. Preserve unrelated user changes. Trace the actual production caller through policy, exact API, raw binding, registry, and generated launcher before editing.
2. State the contract: logical rank/layout, physical adaptation, dtypes, scales, output, bias, legal shapes, architecture, exact kid semantics, split-K, workspace, and JIT versus CO delivery.
3. Register first. Derive codegen, subset builds, manifests, tuner candidates, validation, and dispatch from the canonical declaration instead of creating parallel facts.
4. Keep selection in policy/callers and execution in the family launcher. Reuse the existing route; ask if the ABI is still ambiguous after inspecting sibling families and upstream behavior.
5. When moving C++ heuristics or shape-driven behavior into Python, preserve the final selected kid/split-K, padded-key order, malformed/missing-config handling, large-tensor guards, and fallback. Use an independent boundary/parity test rather than assertions copied from the new implementation.
6. When `main` changes overlapping OPUS code, merge behavior rather than choosing all of `ours` or `theirs`. Re-run generators and parity/config tests after reconciliation.
7. Add focused regression tests, run the relevant production caller, then audit every changed file and remove layers, branch-only reports, generated HTML, temporary traces, and unrelated documentation.

For a sequence of reviewer fixes, keep a short change ledger while working. For each commit record the observed failure, broken invariant, owning layer, smallest fix, proof, and whether a later commit reworked or reverted it. Do not infer the current design from an old commit without checking its descendants; use the workflow and PR #4961 examples in [references/change-patterns.md](references/change-patterns.md).

## Build and source discipline

- Preserve `__HIP_DEVICE_COMPILE__`/`__HIPCC_RTC__` separation. Keep Torch headers out of per-instance/device translation units and use the checked `aiter_tensor_t` boundary.
- Preserve subset-build support and prove every requested kid produces a generated symbol and exact-table entry. Include any mandatory sidecar families required by the module build.
- Regenerate owned outputs; never hand-edit generated dispatch or copy a binary/symbol under another identity.
- Measure clean-build impact before adding fused host translation units, caches, or compiler flags.
- Comments explain ABI quirks, layout/stride math, hardware limits, or compiler workarounds. Keep review history, benchmark logs, migration narration, and temporary validation reports out of production files.
- One fact has one owner. Remove duplicate validation, candidate sets, workspace formulas, wrappers, and stale compatibility layers.

## Completion gate

- Run formatting/lint, syntax checks, `git diff --check`, focused CPU tests, target-architecture GPU correctness, and a clean relevant subset build.
- Verify registry uniqueness, family/architecture membership, generated symbol coverage, subset inclusion, and exact registered-set versus launched-set equality when claiming full coverage.
- Exercise direct and split-K paths, every supported output dtype, automatic and caller-owned workspace, wrong-workspace rejection, and the reducer where applicable.
- Run the real production caller after exact replay, then measure warmed performance. Report exact architecture, kids, shapes, dtypes, split modes, workspace modes, failures/rejections, and anything not actually verified.
- Testing each kid on one legal shape proves launch coverage, not exhaustive shape correctness. Do not claim >4-GiB validation unless a >4-GiB allocation and access path actually ran.

## Communication

Before editing, state the selected path, contract, and expected files. Afterward, lead with the outcome, list exact files/functions, report commands and results, distinguish kernel failures from intentionally rejected candidates or unavailable runners, and state remaining limitations.
