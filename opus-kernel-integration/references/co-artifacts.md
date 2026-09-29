# OPUS precompiled CO integration

Read this reference only when the task changes precompiled `.co` kernels, their declaration/build pipeline, packaged assets, or runtime loader.

## Identity and ownership

- Declare a precompiled variant once in `csrc/opus_gemm/gen_co/co_kernels.json`. The offline builder and registry loader consume that declaration; do not add a manual parallel runtime table.
- A CO kid remains an ordinary exact kid in its existing logical family. Do not add a CO-only public API, policy path, workspace owner, heuristic dispatcher, or sidecar symbol map.
- The instance name must deterministically own the global kid, generated host launcher, ELF kernel symbol, `.co` filename, build-info record, and runtime registry entry.
- Runtime registry, codegen, tuner, and config validation consume only entries whose required image exists. Only the offline artifact producer may inspect declared-but-not-yet-built entries.
- Choose CO for a concrete delivery requirement such as compiler support or flags that release JIT cannot preserve. Do not use CO merely to avoid normal registry/codegen integration.

## Build and packaging

- Rebuild images and `build_info.json` with `csrc/opus_gemm/gen_co/build_co.py`; never hand-edit generated launchers, rename an image, or copy one under another symbol.
- Keep CO codegen host-only. Release hipcc must not instantiate the offline-only device implementation.
- Preserve runtime resolution through `OPUS_GEN_CO_DIR/<running arch>/<symbol>.co` and wheel inclusion of every registered packaged image.
- Commit only the manifest, required bare ELF images, and reproducibility metadata. Exclude temporary TUs, assembly, disassembly, profiler traces, and investigation reports.
- For clustered kernels, preserve cluster-launch support. Round only the physical grid to whole clusters while retaining logical shape and stride metadata.

## Reproducibility metadata

Treat `build_info.json` as part of the artifact contract:

- Verify one-to-one correspondence among manifest entries, registered kids, generated launchers, tracked images, build-info records, and unique symbols.
- Record deterministic source and artifact fingerprints. A source-only change may update the source fingerprint while the verified binary digest remains unchanged; do not rebuild or alter the artifact digest without a binary change.
- Validate recorded compiler/toolchain inputs, kernarg size, LDS, launch bounds, cluster geometry, and relevant device flags against host traits.
- Remember that JIT flag filtering may silently drop unsupported flags; verify the effective offline compiler command when a flag is the reason for CO delivery.

## Validation

- Run CPU integration coverage for manifest/image/build-info parity, duplicate identities, source/artifact fingerprints, host-only codegen, missing-image filtering, symbol uniqueness, and launch-plan rejection.
- On target hardware, launch through both the packaged path and an `OPUS_GEN_CO_DIR` override when loader or packaging behavior changed.
- Exercise exact-kid numerical correctness, split-one/direct behavior, every supported multi-split path, reducer, output dtypes, bias rules, and automatic/caller workspace.
- Verify a missing or mismatched image keeps the kid out of runtime/tuner/generated dispatch instead of deferring failure until first launch.
- When integrating an upstream CO family, preserve the existing final policy behavior and tuned tables. Merge semantics and regenerate; do not restore removed legacy C++ shape dispatch or allocator-owned workspace.
