# OPUS tuning and validation

Read this reference when the task includes tuning, CSV integration, saved-kid replay, production-path verification, padding, performance, full-kid sweeps, or PR CI diagnosis.

## Family routes

Do not substitute a nearby family because its shapes look similar.

| Family | Tuner | Saved fields | Replay behavior |
|---|---|---|---|
| A16W16 GEMM | `csrc/gemm_a16w16/gemm_a16w16_tune.py --libtype opus` | `gfx`, full shape/dtype/bias key, `libtype=opus`, `solidx`, `splitK` | `--run_config` benchmarks production `aiter/tuned_gemm.py::gemm_a16w16`; verify the saved exact pair separately |
| Plain A8W8 no-scale/blockscale | `csrc/opus_gemm/opus_gemm_a8w8_tune.py` | `gfx`, `cu_num`, shape/dtype, `scaleAB`, `libtype=opus`, `kernelId`, `splitK=0` | `--run_config` calls exact `opus_gemm`; there is currently no separate OPUS production policy |
| MXFP8 BMM | `csrc/opus_gemm/opus_bmm_mxscale_tune.py` | `gfx`, `b,m,n,k`, `libtype=opus`, `kernelId`, `splitK` | Saved fields call exact `opus_bmm`; shape-only replay uses `aiter/ops/batched_gemm_op_a8w8.py` production policy |

For plain gfx950 A8W8, `scaleAB=False` is no-scale and `scaleAB=True` is 1x128x128 blockscale. Do not treat it as MXFP8 BMM. Do not reroute the established general A8W8 interface merely to create a production path for the tuner.

Bpreshuffled blockscale is another explicit family/layout. Its high-level general A8W8 caller enters OPUS only for a valid `libtype=opus` row. If an architecture registers no OPUS bpreshuffle kid, report it unavailable; executing a CK/FlyDSL fallback is not OPUS coverage.

`csrc/opus_gemm/opus_gemm_tune.py` is the debug-only single-shape A16 runner and the candidate/helper implementation imported by the production multi-backend tuner. Do not present it as the production A16 tuning entry point.

## Tuner contract

1. Gate unsupported architectures before reading input files, compiling, tuning, or replaying. A gfx950-only tuner must reject gfx942 rather than produce misleading output.
2. Keep modules import-safe: parsing and execution belong under `main()`.
3. Preserve `gfx` and the complete runtime lookup key in every CSV. Architecture is part of identity; omitting it can select a valid kid from the wrong architecture.
4. Enumerate candidates from the canonical registry and assert completeness for the intended family/tags. Filter using declared shape, dtype, scale-layout, split-K, and workaround metadata.
5. New output should store global registered kids. Normalize an established legacy local-id format only at a documented compatibility boundary, then resolve the exact registry instance.
6. Use an independent FP32 Torch reference. For scaled FP8, use nonuniform scales so layout/indexing errors are observable. Poison or clear preallocated output before correctness checks.
7. Preserve a nonzero mismatch when serializing rounded ratios; a rare mismatch must not become `0.0000` and pass an exact gate accidentally.
8. Time warmed kernel calls only. Exclude data generation, reference computation, allocation, output poisoning, CSV work, and profiler startup unless the measurement explicitly targets them.

An intentionally broad tuning policy may generate incompatible `(kid, splitK, shape)` combinations. Launcher rejection is acceptable only when the row cannot become a winner and the final report separates valid timings, policy rejections, timeouts, numerical failures, and launch/compile failures. Repeated predictable rejections are a candidate-policy efficiency issue, not a kernel correctness failure.

## Saved CSV replay

Do not assume every tuner gives `--run_config` the same meaning:

- A16 production tuning intentionally replays through `gemm_a16w16`. Install/load the generated CSV, prove production lookup selects the saved `solidx/splitK`, and separately call exact `opus_gemm` when exact-pair proof is required.
- Plain A8W8 saved replay calls exact `opus_gemm`.
- MXFP8 BMM rows containing `libtype/kernelId/splitK` call exact `opus_bmm`; legacy shape-only replay remains a production-policy test.

For an exact saved-row replay:

- Require and validate `libtype=opus`, the exact saved kid, saved split-K, family/layout, architecture, output dtype, scales, and shape compatibility.
- Call `opus_gemm` or `opus_bmm` with that saved kid and split-K. Do not silently run production heuristics instead.
- Reject stale, unregistered, wrong-family, wrong-scale, or incompatible saved rows with a useful error.
- If a shape-only replay mode is supported, keep it explicitly separate and let that mode exercise production policy. Do not conflate it with saved-kid replay.
- Unit-test the arguments passed to the exact API. On target hardware, separately run the production caller with the generated CSV and confirm the selected kid. Use a GPU trace when proof of the exact kernel/reducer matters.

Tune success alone is incomplete. The minimum end-to-end proof is:

```text
shape CSV -> registry candidates -> correctness-gated timing -> tuned CSV
          -> exact saved-kid replay -> production caller lookup -> observed hit
```

## Split-K and workspace validation

- Cover split-one/direct behavior and every production-supported multi-split mode. Do not infer multi-split support from a kid band or a name.
- Validate the main partial-output kernel and reducer. A trace of only the main kernel is insufficient for a split-K claim.
- Compare automatic workspace with caller-owned workspace bitwise when both use the same path.
- Test wrong workspace dtype, device, contiguity, and capacity. Direct-only kernels must reject workspace or unsupported split-K.
- Test every supported final output dtype because the reducer owns conversion.
- For compiler-workaround kids whose `D_OUT=void` specialization differs from split-one, compile and run both specializations. Verify the workaround does not change the kid identity or its direct-path tuning.
- Distinguish requested split-K, converged/ABI split-K, and workspace capacity in tests. Short-K cases must validate and allocate from the converged launch value; an oversized caller workspace may remain valid without changing the launch.

## M padding is three different behaviors

Never equate these:

1. **Padded-key lookup:** policy searches an M=64 CSV row for logical M=53.
2. **Logical-M launch:** the selected kernel really accepts M=53 through OOB/masking behavior.
3. **Physical padding:** input and scale tensors are allocated/padded to M=64, launched as M=64, and cropped back to 53.

For an M=53/M=64 check:

- First determine the exact kid's M alignment and OOB capability from registry metadata and actual launcher behavior.
- If it has no OOB support, passing an M=53 tensor to an M=64 config must reject or safely fall back; physically pad the tensor and all M-indexed scales before exact launch.
- Compare the cropped 53 rows with the direct/reference result and require padded output rows to be zero when zero padding promises that behavior.
- Exercise the production lookup separately. Prove whether it hits the saved kid, falls back, or uses a logical-M-capable kernel.
- For BMM, test both token-major and batch-first physical views used by the real caller.

## Full registered-kid sweep

When claiming full architecture coverage:

- Snapshot the canonical registered set by architecture and family/tag.
- Build a matching subset and prove generated-symbol/exact-table coverage. Account for mandatory sidecar kids that a module always builds.
- Launch every registered kid on at least one legal, useful shape; choose per-kid shapes rather than forcing one invalid common shape.
- Cover supported output dtypes, bias-aware variants, direct/split paths, workspace ownership, and alignment guards. For BMM, verify every kid's declared `m_align` with one legal and one illegal M where practical.
- Compare the registered and launched sets exactly and report missing/extra ids.
- A declared, disabled, image-missing, or zero-registration family is not launch coverage. Report it unavailable instead of counting a non-OPUS fallback.
- Store per-row correctness and warmed timing, but do not rank different kids when their validation shapes differ.
- State exclusions such as compiler-specialized split modes, direct-only kernels, unavailable artifacts, or untested >4-GiB boundaries.

## Policy and compatibility tests

- When policy moves from generated C++ to Python, keep an independent translation of the old final behavior and sweep decision boundaries. Do not import or mirror the new implementation as the reference.
- Test shipped tuned rows as a family-wide merge, including duplicate full keys and cross-architecture collisions. Run `python3 -m unittest op_tests.tuning_tests.test_config_shape_collision -v` when schemas or loaders change.
- A stale saved row must warn once per relevant row and continue through the established padded-row/fallback order. Exact APIs still reject stale ids directly.
- Test real transposed/K-contiguous views and padded leading dimensions. A convenient contiguous substitute does not reproduce production.

## PR CI diagnosis

Separate code failure from scheduling:

- A job showing `Waiting for a runner to pick up this job`, no steps, and empty/zero `runner_id` is queued for infrastructure capacity, not a test failure.
- Confirm the workflow condition expanded to `true`, then report the requested self-hosted runner label and queue duration. Do not remove/re-add labels or rerun immediately; that can reset the queue.
- A `labeled` PR event may create new skipped records for unrelated workflows. Preserve the distinction between those expected skips and earlier successful required checks.
- If the queue persists beyond the project's normal provisioning window, ask a runner administrator to check the exact runner labels, group access, autoscaler health, and GPU quota.
