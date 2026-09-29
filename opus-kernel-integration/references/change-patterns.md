# OPUS change history and review patterns

Read this reference when reconstructing a long-running OPUS branch, responding to reviewer findings, planning commits, reconciling upstream changes, or diagnosing a regression whose symptom crosses policy, registry, codegen, launch, and test layers.

The commit ledger below is derived from PR #4961. It is a forensic index, not an alternate specification. Current source and the contracts in `SKILL.md` remain authoritative. A row marked **superseded**, **removed**, or **merge-only** must not be copied as current design.

## Record each meaningful commit

For every OPUS commit, capture these six facts in the commit message, review notes, or this ledger:

1. **Observed failure:** the user-visible error, wrong result, performance regression, build failure, or review concern.
2. **Broken invariant:** the contract that was violated, such as exact-kid identity, workspace ownership, architecture scoping, or tuned-row validity.
3. **Owning layer:** policy, family adapter, launch plan, registry, generated host table, device implementation, CO tooling, or tests.
4. **Modification pattern:** the smallest coherent change and the intentionally untouched layers.
5. **Proof:** focused CPU/meta test, generated-symbol comparison, clean subset build, target-GPU numerical test, production replay, or CI evidence.
6. **Lifecycle:** retained, refined by a later commit, reverted, removed as branch-only material, or introduced only by an upstream merge.

Useful history commands:

```bash
git status --short --branch
git merge-base HEAD upstream/main
git log --reverse --first-parent upstream/main..HEAD
git show --stat --summary <commit>
git show --first-parent <merge-commit>
git range-diff <old-base>...<old-head> <new-base>...<new-head>
```

For a merge commit, inspect both the merge result and the first-parent delta. A conflict-free merge can still change generated tables, tuned rows, JIT behavior, or package ownership.

## Diagnose by the broken boundary

### Public API and family ownership

Symptoms include a 2D GEMM accepting a 3D physical contract, generic A8W8 APIs selecting OPUS from missing scales or output dtype, CK/MoE imports changing solely because of package nesting, or compatibility wrappers recursing through the public dispatcher.

Modification pattern:

- Keep `opus_gemm` logical 2D and `opus_bmm` logical batch-first 3D.
- Put physical shape adaptation in the owning family module.
- Keep shared exact dispatch in `dispatch.py`; keep family modules flat unless a real ownership boundary requires nesting.
- Restore generic API signatures before adding OPUS-specific convenience behavior.
- Let a compatibility wrapper call its private family launcher directly.

Proof should cover signatures, rank errors, imports, fake/meta behavior, and the actual legacy caller.

### Exact execution versus compatibility policy

Symptoms include one entry point redirecting a legacy kid while another rejects it, an exact API consulting a tuned CSV, or an invalid saved row silently reaching a raw launcher.

Modification pattern:

- Keep the public exact API strict: execute the requested registered kid or reject it.
- Normalize legacy local ids, sibling redirects, padded-key lookup, heuristics, and fallback in policy or the owning compatibility wrapper.
- Resolve to one final `(kid, split_k)` before exact launch and workspace allocation.
- If historical behavior redirected only in a shape-driven API, preserve it there rather than weakening exact semantics globally.

Proof should test explicit ids, tuned rows, heuristic misses, invalid rows, and the strict exact path as separate cases.

### Registry versus current build availability

Symptoms include Python planning a valid workspace for a registered kid while C++ reports `workspace=None`, or a tuner enumerating a kid that has no generated symbol in the current subset build.

There are four distinct states:

```text
registered -> requested for this build -> emitted/linked -> loadable at runtime
```

Modification pattern:

- Keep registry capability metadata canonical.
- Derive the default compile floor and requested subset from registry identities.
- Query generated direct and workspace tables separately at runtime.
- Report `not compiled in the current module` when neither table contains the kid; do not infer that a miss in the workspace table means direct output.
- Add only representative default-tested families to the default floor. Leave large tuning spaces on demand.

Proof should compare registered, requested, generated, and launched sets. A fresh codegen test is required because an incremental build can hide a missing default kid.

### Split-K and workspace planning

Symptoms include short-K rejection despite a valid converged split, oversized automatic workspaces, reducer dtype mismatches, or different behavior for automatic and caller-owned workspace.

Modification pattern:

- Separate requested split-K, converged/ABI split-K, and workspace capacity.
- Apply K-tile and parity limits to the value actually launched.
- Size automatic workspace from the converged value.
- Permit larger caller-owned capacity only when the established ABI allows it.
- Keep workspace allocation in Python and validation/consumption in C++.

Proof should include short K, explicit and automatic split-K, automatic and caller workspace, wrong dtype/device/contiguity/capacity, each output dtype, and reducer execution.

### Tuned CSV and replay

Symptoms include cross-architecture collisions, local ids interpreted as global ids, malformed rows aborting production, shape-only replay claiming exact-kid coverage, or a tuner running on an unsupported architecture.

Modification pattern:

- Preserve the full lookup identity, especially `gfx`.
- Normalize an established legacy id only for the relevant backend rows and only at the compatibility boundary.
- Validate against an exact registry instance rather than a numeric band.
- Warn and continue through the established padded-row/fallback order for stale production rows.
- Make saved-kid replay call the exact API; keep shape-only production replay separate.
- Reject unsupported architectures before reading input, compiling, or tuning.

Proof should cover duplicate keys, mixed backends, malformed numeric fields, stale ids, exact saved arguments, production lookup hits, and M-padding semantics.

### Python hot paths and caches

Symptoms include per-call module lookup, repeated dtype conversion, an unbounded shape dictionary, cached tensors/workspaces, or launcher objects duplicated in every cache entry.

Modification pattern:

- Cache no-argument launcher/module resolution separately.
- Cache only bounded scalar launch plans and immutable metadata.
- Never cache tensors, pointers, streams, allocations, or unbounded ragged-shape maps.
- Preserve fake/meta and `torch.compile` behavior when optimizing eager execution.

Proof should inspect cache bounds and contents and compare eager, fake/meta, and compiled calls.

### ABI, device, and stream state

Symptoms include first-call execution on the process current device instead of the tensor device, missing HIP stream propagation, duplicated C ABI and pybind paths, or removing `develop=True` under the false assumption that it controls rebuilding.

Modification pattern:

- Resolve device state from input tensors.
- Treat `develop=True` as an `aiter_tensor_t`/stream adapter where that is its actual implementation; `AITER_REBUILD` controls rebuilding.
- Prefer the existing checked pybind/raw ABI unless a measured need justifies a C ABI.
- If a temporary fast C ABI is added, test first-call priming, device restoration, non-default streams, graph capture, and exception restoration; remove it if it duplicates the established path without durable benefit.

### Codegen and compiler specializations

Symptoms include a `D_OUT=void` workspace instantiation failing while the direct specialization compiles, mixed-architecture symbols leaking into one build, or a generated table omitting a requested kid.

Modification pattern:

- Put compiler workarounds in per-instance registry metadata.
- Generate direct-output and workspace specializations independently.
- Preserve one global kid identity across specializations.
- Keep host and device passes separated and validate mixed-architecture guards.
- Regenerate dispatch; never hand-edit emitted tables.

### Precompiled CO artifacts

Symptoms include a manifest/image mismatch, a declared kid without an image, a loader finding the wrong symbol, or a source-only change incorrectly rewriting binary identity.

Modification pattern:

- Keep one manifest-driven identity for kid, symbol, image, launcher, and metadata.
- Filter missing images before runtime registration.
- Keep release codegen host-only for CO families.
- Preserve the existing exact API, policy, and caller-owned workspace.
- Reconcile artifact tooling with upstream rather than maintaining a branch-only parallel verifier when upstream owns the format.

Proof should cover manifest/image/build-info parity, fingerprints when supported, missing-image behavior, packaged and override loading, exact launch, reducer, and supported output dtypes.

### Merge, scope, tests, and CI

Modification pattern:

- Merge behavior, not whole files. Re-run generators after any overlap in registry, JIT, tuner, or generated dispatch.
- Remove generated reports, machine-specific runbooks, redundant benchmarks, and branch-only tests before final review.
- Keep representative regression tests in maintained suites; do not create a duplicate test file for one reviewer scenario.
- Tests should encode behavior, not branch commit hashes or temporary implementation names.
- Separate code failures from runner queues, cache/artifact service timeouts, and expected skipped workflows.

## PR #4961 commit ledger

### Foundation and architecture

| Commit | Problem area | Modification pattern and durable lesson | Lifecycle |
|---|---|---|---|
| `ebf718fa` | Task1/Task2 had diverged from upstream | Reapplied the validated final state on a new upstream base: Torch-owned split-K workspace plus caller-resolved exact-kid execution. Reapply final semantics, not obsolete intermediate patches. | Foundation; later refined |
| `e2002dcb` | Documentation volume obscured the implementation | Consolidated task notes and removed duplicated narrative. Keep stable contracts near maintained code and avoid review-report sprawl. | Retained principle |
| `977dd7a5` | Validation was not reproducible | Added a detailed Task1/Task2 validation workflow. A runbook can help during development, but it is not automatically a production artifact. | Later removed |
| `6aca6356` | Upstream MXFP8 tuned data and the new global kid namespace disagreed | Preserved the upstream CSV, normalized legacy OPUS ids in memory, limited translation to OPUS rows, and deduplicated only on columns present in the schema. Preserve shipped data; adapt at a documented boundary. | Refined later in `92b36fbb` |
| `d18b8049` | First C-ABI priming could launch on the wrong current GPU | Guarded the priming pybind call with the input tensor device and added multi-device restoration coverage. Device and stream state come from the tensor, not process-global state. | C ABI later removed; lesson retained |
| `79456ddd` | Temporary validation documents remained in the patch | Removed task-only reports after extracting useful information. Do not ship investigation narration by default. | Cleanup |
| `995f343f` | Python dispatch overhead and repeated resolution | Cached launcher resolution and scalar plans while preserving fake/compile paths. Separate stable function caches from bounded shape caches; never cache tensors. | Retained and later bounded further |
| `76beb3c3` | Upstream gfx1250 split-K work used a different integration architecture | Ported policy tags, cluster grid rounding, reducer dispatch, mixed-arch guards, and tuned configs into the exact-kid/caller-workspace design without restoring old allocation or shape dispatch. Merge semantics across ownership boundaries. | Refined by `6ea8d5b6` |
| `8f12a00c` | gfx942/gfx1250 validation coverage needed a plan | Documented target-specific validation before execution. Keep the plan useful during development but remove branch-only reports later. | Later removed |
| `17d895dd` | A16 C-ABI fast path added measurable Python overhead | Reused thread-local descriptors and removed redundant error queries, then compared correctness and ABBA performance. Optimize only after measuring the real boundary. | C ABI later removed |
| `be54e9bc` | Selection and execution were mixed | Moved architecture heuristics and tuned validation into a policy module while preserving generic caller fallback. Policy selects; exact family launchers execute. | Retained; module evolved into `policy.py` |
| `bce5c477` | GEMM and BMM public semantics were blurred | Split logical 2D GEMM and batch-first 3D BMM adapters and updated callers/tests. Adapt legacy physical layouts inside the family owner. | Retained |
| `5f7adaba` | Launch planning and tuner routes were duplicated or misowned | Centralized immutable launch planning, repaired production-versus-debug tuner routes, and aligned tests with the actual caller. One fact and one route per family. | Core design retained |

### Scope reduction and upstream reconciliation

| Commit | Problem area | Modification pattern and durable lesson | Lifecycle |
|---|---|---|---|
| `059b2b48` | The integration touched generic JIT, CK, and unrelated callers | Reverted broad changes and kept OPUS adaptation within its owning modules. A smaller diff is a correctness tool because ownership becomes reviewable. | Retained |
| `366c8a9c` | Generated HTML reports inflated the PR | Removed generated reports. Keep reproducible commands/results, not rendered investigation artifacts. | Cleanup |
| `aa295b1c` | Upstream changed tuned data, JIT core, and tuner code | Reconciled overlapping behavior selectively instead of choosing all of either side. Recheck schemas and codegen after merges. | Merge reconciliation |
| `c7034f9d` | Test and benchmark duplication made failures ambiguous | Removed redundant suites and microbenchmarks. Prefer maintained entry points and a small set of diagnostic benchmarks. | Cleanup |
| `b621a6cb` | Broad suites repeated the same assertions | Kept representative regressions covering distinct contracts. Coverage should map to invariants, not file count. | Cleanup/refocus |
| `a081f330` | Interface tests overlapped after consolidation | Removed duplicate API/tuner tests and updated docs. Each behavior should have one clear owner. | Cleanup |
| `4bf3bfd8` | Eager A16 dispatch repeated work and legacy workspace metadata had two owners | Simplified the eager path and removed the old workspace registry in favor of canonical launch planning. Avoid sidecar capability maps. | Retained |
| `a3b76325` | Important paths lacked end-to-end validation | Added correctness, graph, workspace, tuning, and documentation coverage. Later pruning shows that useful exploratory coverage must be folded into maintained suites. | Partly retained, partly pruned |
| `9e7ae8eb` | MI308X execution needed an operational guide | Added a hardware-specific validation guide. | Removed by `5c7adcd0` |
| `2af893dc` | Upstream synchronization | Merged upstream and required conflict audit across OPUS-owned files. A merge commit is an integration checkpoint, not proof of behavior. | Merge-only |
| `da3d008b` | Second upstream synchronization | Repeated the merge checkpoint after upstream advanced. Inspect first-parent delta and regenerate affected outputs. | Merge-only |
| `5c7adcd0` | Machine-specific guide did not belong in production | Removed the MI308X guide after validation knowledge was captured elsewhere. | Cleanup |
| `4df32752` | Branch-only test files would create long-term maintenance cost | Removed dedicated files after retaining representative cases in established suites. | Cleanup |

### Compiler, compatibility, CO, and formal review

| Commit | Problem area | Modification pattern and durable lesson | Lifecycle |
|---|---|---|---|
| `95d7f012` | Python style and lint failures across the refactor | Applied formatter/linter-compatible rewrites without changing contracts. Run style checks before expensive target builds. | Retained |
| `8cf11982` | gfx950 MXFP8 split-K `D_OUT=void` specialization failed compilation | Added narrowly scoped per-instance/codegen handling and a regression while leaving the same kid's direct specialization intact. Compiler workarounds belong in explicit metadata, not numeric ranges. | Retained |
| `f6268f43` | New regression test failed formatting | Formatted only the affected test. Keep mechanical cleanup separate from semantic fixes when practical. | Cleanup |
| `7a43edb8` | A16 test entry failed when invoked without CLI arguments | Restored a safe no-argument/default path. CLI tools and test modules must be import-safe and have explicit `main()` behavior. | Retained |
| `d9e0f7a9` | Compatibility wrappers and tuner routes bypassed the intended family ownership | Restored the shape-driven compatibility API, corrected legacy shim routing, and clarified which tuner/replay path is authoritative. Compatibility enters the owning family directly. | Refined by later review fixes |
| `2f847d1a` | 219 gfx1250 CO kernels needed integration without a parallel runtime | Added manifest-driven registry entries, host launchers, packaged images, loader/build tooling, tuned rows, and tests while retaining the exact API and caller-owned workspace. | Retained core; tooling later reconciled |
| `92b9626b` | A duplicate obsolete BF16 tuning table could shadow current data | Removed the full stale table rather than preserving unmatched rows. Delete obsolete sources of truth atomically. | Retained |
| `e3d9dd89` | Remaining BF16 tables did not match the imported CO work | Aligned shipped dsv3/dsv4 rows with the source integration. Config changes are part of runtime behavior and require replay. | Retained, subject to later upstream updates |
| `6df2316d` | Upstream synchronization after CO work | Merged upstream and audited OPUS conflicts. | Merge-only |
| `534a0fc2` | Must-fix review: heavy package init, OPUS leakage into generic A8W8, weak fixed-contract access, Unicode runtime text, questioned `develop=True`, missing gfx1250 CO launch coverage | Moved private dispatch out of `__init__`, restored generic A8W8 signatures, tightened required attributes, kept the ABI/stream adapter with evidence, and added CO numerical/loader coverage. Address each reviewer item at its owning layer rather than mechanically applying every suggestion. | Retained; package layout refined next |
| `d59aa485` | The first review fix added unnecessary `gemm/` and `moe/` package nesting | Returned family modules to the flat package and restored CK/MoE imports. Structural refactors must justify every downstream path change. | Retained |
| `41bdd498` | Should-fix review: duplicated architecture strings and weak CO provenance | Reused shared architecture constants and added source/artifact fingerprint verification. Centralize identity constants; make artifacts auditable when the project owns that mechanism. | Constants retained; provenance mechanism later aligned with upstream |
| `341813aa` | Follow-up style failures | Applied focused Python style fixes after the review series. | Cleanup |
| `c4223109` | Upstream synchronization | Merged main before further semantic review fixes. Re-evaluate prior assumptions after every merge. | Merge-only |

### Policy correctness and simplification

| Commit | Problem area | Modification pattern and durable lesson | Lifecycle |
|---|---|---|---|
| `92b36fbb` | MXFP8 BMM cached unbounded ragged shapes and invalid tuned ids aborted instead of falling back | Replaced the dict with bounded scalar `lru_cache`, cached launchers separately, normalized legacy ids, validated exact registry membership, and warned/skipped stale rows. Numeric kid ranges are not capability checks. | Retained |
| `2cbd16c7` | A16 policy migration lacked independent parity and stale-row behavior was unclear | Added a frozen pre-PR heuristic reference, boundary sweep, compatibility route tests, and warn-once fallback for invalid tuned rows. Prove final `(kid, split_k)` behavior independently of the new implementation. | Retained |
| `0aa9c71a` | A comment-only source change made the recorded CO source fingerprint stale | Updated source provenance while keeping the verified artifact digest unchanged. Source and binary identities are separate. | Superseded by upstream tooling alignment |
| `cc7af9de` | Thin helpers duplicated logic and an A8 backend condition overcomplicated routing | Inlined one-use validation and simplified the tuned backend branch without changing ownership. Remove wrappers that add no contract boundary. | Retained |
| `1e41b664` | A second A16 C-ABI path duplicated the checked pybind launcher | Removed C-ABI code, device/stream guards, descriptors, and tests; retained one exact launch path. A measured optimization should still be deleted if its complexity outweighs durable value. | Retained current architecture |
| `e986c562` | Upstream FlyDSL TopK availability regression | Pulled an unrelated upstream fix needed for test collection. Do not treat it as OPUS architecture or include it in an OPUS-only commit. | Upstream-only |
| `515f0621` | Upstream synchronization | Merged main after simplifying the launch path. Audit any OPUS-adjacent changes brought by the merge. | Merge-only |
| `61466d8d` | Attempted removal of an unrelated `setup.py` comment delta | Tried to reduce scope, but the next commit showed the baseline had been interpreted incorrectly. Verify parent state before cleanup. | Reverted |
| `6665c85e` | Previous cleanup was incorrect | Reverted `61466d8d` instead of hiding the mistake in a mixed commit. Use explicit reverts when review history benefits. | Revert; no net OPUS behavior |
| `4aa1ead4` | Packaging files still carried comment-only branch noise | Removed only the nonfunctional deltas after rechecking the correct baseline. | Cleanup |
| `cf0d9bc7` | Policy parity tests embedded historical SHA markers | Removed commit identifiers while preserving independent expected behavior. Tests should document contracts, not archaeology. | Retained |
| `ec73ed52` | Upstream changed BF16 tuned rows | Merged main and refreshed expected shipped-row outcomes while keeping the independent heuristic oracle intact. | Merge plus reconciliation |

### Split-K, tuner replay, and final upstream integrations

| Commit | Problem area | Modification pattern and durable lesson | Lifecycle |
|---|---|---|---|
| `5e5869d5` | gfx942 short-K planning compared conservative workspace capacity with the K-tile limit | Moved the limit to the effective launch split-K and added short-K coverage. Validate launch semantics independently from allocation capacity. | Interim; refined by `3c617854` |
| `3c617854` | Automatic gfx942 workspace still used the pre-convergence split count | Made the planner return the converged ABI split-K and sized automatic workspace from it, while accepting a larger valid caller workspace. | Retained |
| `ac6ea5ce` | Plain A8W8 and MXFP8 BMM lacked complete tuning/replay workflows | Added registry-derived candidates, exact saved-kid replay, separate shape-only production replay, M-padding checks, and correctness-gated results. Each family needs its own real route. | Retained |
| `a65d5ffe` | Plain A8W8 tuner could start on unsupported architectures | Rejected non-gfx950 before reading files or compiling. Fail early enough that output cannot be mistaken for coverage. | Retained |
| `6ea8d5b6` | Upstream gfx1250 split-K changed FP32 partials, reducer row limits, and large-M behavior | Merged behavior into registry, policy, planner, tuner, and generated launchers; enforced `max_m` and retained direct CO paths for large M. Cross-layer capability changes require a vertical slice. | Retained |
| `e30142f1` | Branch-specific CO tooling and provenance checks diverged from newly merged upstream ownership | Restored upstream builder/manifest metadata and removed the parallel CI verifier while preserving exact-kid integration and existing contract tests. Prefer the upstream owner over a duplicate mechanism. | Retained current tooling |
| `a05fe49e` | Merge resolution had changed OPUS JIT cluster-launch flags/escaping | Restored upstream configuration and verified architecture/version gates. Compare evaluated compiler arguments, not only JSON spelling. | Retained |
| `1f24221f` | Main advanced with transactional JIT cache changes; review found CO batch and duplicate split-K coverage issues | Reconciled JIT cache publication, allowed batched direct CO launches while retaining workspace batch-one limits, removed the duplicate gfx1250 test file, kept the row-limit case in the main suite, and included always-emitted MXFP8 ids in compiled metadata. | Current committed head before follow-up |

## Follow-up findings after `1f24221f`

These changes were present in the working tree when this reference was created and should receive their final commit hash after commit:

1. **gfx1250 default clusterlaunch coverage:** a registered workspace kid (`GFX1250_CLUSTERLAUNCH_KID_OF[(16, 32, 128, 2, 1)]`, currently 20100) was absent from the default compile floor. Add the registry-derived identity, not a numeric literal, and prove fresh codegen emits it.
2. **Compiled-table error classification:** a workspace-table miss was incorrectly treated as proof that a kid was direct. Query direct and workspace tables separately on gfx942/gfx950/gfx1250; reject a dual hit as an invariant violation and a double miss as `not compiled in the current OPUS module`.
3. **gfx942 compatibility redirect:** `gemm_a16w16_opus` passed explicit and tuned ids straight to exact launch, bypassing historical non-exact-N redirects. Route those compatibility selections through `resolve_a16w16_tuned_candidate`; keep strict `opus_gemm`/`opus_bmm` rejection unchanged, and leave kid 10216 rejected because it has no sibling.
4. **Comment drift:** update old `20500+` clusterlaunch comments to the actual `20100+` band, but do not use the band as a capability predicate.

Required proof for this follow-up:

- fresh gfx1250 codegen contains the mapped clusterlaunch kid;
- a registered but omitted gfx950 workspace kid reports `not compiled`, not `workspace=None`;
- strict gfx942 exact planning rejects non-exact N for 10210/10213/10216;
- the compatibility entry redirects explicit and tuned 10210/10213 and rejects 10216;
- formatting, lint, syntax, `git diff --check`, focused CPU/meta tests, relevant clean builds, and available target-GPU tests are reported separately.

## Commit planning checklist

Before creating the next commit:

- Exclude unrelated submodule positions, local skill files, generated reports, and investigation notes unless the user explicitly includes them.
- Review the staged diff, not only the working-tree diff.
- Keep source, focused regression tests, and contract documentation together when they describe one behavior change.
- Split unrelated compiler, policy, artifact, or cleanup changes unless they are required for one complete vertical slice.
- State which target architectures were actually run. A CPU/meta proof, a successful compile, and a target-GPU numerical launch are different claims.
- After push, verify the remote head SHA and classify any failed check as code, scheduling, cache/artifact infrastructure, or expected skip before changing source again.
