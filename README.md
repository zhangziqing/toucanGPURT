# Toucan GPU Runtime: Nanhu Ready V1

This branch starts at upstream `132daee` and integrates the exact checked and
final-only runtime sources accepted in the RTX PRO 6000 wide32 experiments.
The original snapshots were hash-verified before import. Shared device hooks and
runner sources remain byte-identical; the narrow adaptations listed below have
host checks, but **this integrated branch has not yet been compiled with NVCC or
executed on a GPU**.

## Build Contract

Requires Python 3.10+, GNU `ar`, and a C++20-capable CUDA toolkit supporting the
selected architecture. Git is optional for repository provenance. The accepted source
build used NVCC 13.2.78, `sm_120`, `-O3`, `-lineinfo`, and a 64-register runtime
limit. The script preserves these settings, with architecture selected explicitly.

```sh
python3 build_nanhu.py --abi wide32 --arch sm_86 \
  --output build/wide32-sm_86 --nvcc /usr/local/cuda/bin/nvcc
python3 build_nanhu.py --abi legacy16 --arch sm_86 \
  --output build/legacy16-sm_86 --nvcc /usr/local/cuda/bin/nvcc
```

Use `--arch sm_120` with a supporting toolkit for the RTX PRO 6000 Blackwell;
`sm_86` is the RTX 3090 target. A build does not reserve a GPU, initialize CUDA,
run a probe, or launch evaluation. A different architecture needs a separate
output directory. Do not run concurrent builds into the same directory.

Equivalent Make entry point:

```sh
make nanhu ABI=wide32 ARCH=sm_86 BUILD_DIR=build/wide32-sm_86 NVCC=/usr/local/cuda/bin/nvcc
```

For `--output BUILD`, the output contract is:

```json
{"abi":"wide32","arch":"sm_86","binaries":{"checked":"checked/nanhu-checked","batch_final":"final/nanhu-final","resources":"final/resource-probe"}}
```

The harness only needs these relative paths joined to `BUILD`. The additional
top-level identity keys below are for audit consumers:

| Manifest key | Binary path relative to BUILD |
| --- | --- |
| `checked_binary` | `checked/nanhu-checked` |
| `final_binary` | `final/nanhu-final` |
| `resource_probe` | `final/resource-probe` |

`BUILD/build.json` is written atomically only after all three binaries succeed.
Each key contains an absolute `path`, `bytes`, and `sha256`. The manifest also
records ABI, architecture, NVCC path/version, source revision and dirty status,
source hashes, exact commands, per-command build time, and log identities.
Git revision and status are recorded only when `git rev-parse --show-toplevel`
resolves to this runtime's exact source root. Standalone exports and copies
nested in another repository use `null` for both fields, retaining source hashes
including `provenance/accepted-sources.json` when supplied. The enclosing
repository's revision is never attributed to an exported runtime.
`plan.json`, `commands.json`, and `command-NN.log` retain build diagnostics.
A failed rebuild invalidates the previous success manifest.

`--dry-run` prints the command plan without writing files or invoking tools.
It supplies binary paths, but not artifact hashes or a successful build status.

The original generic runtime remains available via `make ABI=legacy16 ARCH=sm_86`
or `make ABI=wide32 ARCH=sm_120`. Its library and `test_run` now live under
`build/ABI-ARCH/` to avoid stale objects crossing ABI or architecture boundaries.
The generic target has no Nanhu device hooks; use `make nanhu` for these runners.

## ABI Selection

`TOUCAN_WIDE_POOL` must be defined consistently in the runtime and runner. The
build script controls this for every translation unit. The matching compiler
fork's `nanhu-ready-v1` branch is a dedicated strict-two-round wide32 snapshot:
32-bit fields and `TCWIDE1` serialization are built in, not selected by this
macro. Its output requires `--abi wide32`; `--abi legacy16` instead loads
previously generated legacy16 designs.

| Build ABI | Local offsets/counts | Design serialization | Pool bound |
| --- | --- | --- | --- |
| `legacy16` (default) | `uint16_t` | Original headerless layout | Less than 65535 B |
| `wide32` | `uint32_t` | Eight-byte `TCWIDE1\0` prefix | At most 98304 B |

This preserves the accepted limits, not a new claim that every 16-bit address is
usable. The middle-level LUT encoding remains four bytes in both modes. Symbol
files retain their original format. Design serialization uses the original
native C++ layout, sizes and endianness, tested here on x86_64 little-endian Linux;
it is not a portable interchange format or a parser for untrusted files.

The wide reader rejects headerless designs. The legacy reader checks the prefix
and rejects `TCWIDE*` before interpreting lengths; it therefore requires seekable
input. There is no implicit conversion or shared ABI cache. Layout fingerprints
cover the size, alignment and every public member offset of serialized descriptors
against the original legacy16 and wide32 headers.

## Nanhu Inputs and Invocation

The model and workloads are supplied by the separate ready-to-run bundle, not by
this runtime source repository. Each `DESIGN` directory must contain
`GPUSimDesign.bin` and `GPUSimIOSymbols.bin` of the selected ABI. When packaging
wide output named `GPUSimDesign.wide32.bin`, create the required filename as a
copy or symlink; the runner's accepted filename contract is unchanged.

Use the matching `nanhu-watch.tsv` next to each workload image, rebound by name
from that design's symbols. Numeric watch offsets must not be copied between
partitions. All six retained watches are required for full checked acceptance;
the inherited runner permits absent watch metadata, so the harness must enforce
the bundle's watch manifest. Use fresh output directories for every invocation.

```sh
build/wide32-sm_86/checked/nanhu-checked DESIGN WORKLOAD/checked-loop-1000.bin OUTPUT/checked 20000 1000
build/wide32-sm_86/final/nanhu-final DESIGN WORKLOAD/checked-loop-1000.bin OUTPUT/final 10645 1000 batch-final
build/wide32-sm_86/final/resource-probe DESIGN
```

The probe initializes the selected design and queries resources at its actual
dynamic shared-memory size, but does not launch evaluation. It is an explicit
GPU operation, separate from building. It reports all three kernels, occupancy,
thread count, cooperative grid and batch kernel dispatch.

The final runner supports `batch-final` and `single-final` only for these fixed
loop1/loop1000 endpoints:

| Workload | Total cycles | Trap cycle | Retired | RAM reads |
| --- | --- | --- | --- | --- |
| loop1 | 8511 | 8507 | 19 | 120 |
| loop1000 | 10645 | 10641 | 8011 | 108 |

Both use 64 reset cycles, first commit cycle 631, no RAM writes, three flash
reads, and all 32 GPRs checked against the independent fixed-workload oracle.
The GPU device model reserves 16 GiB of RAM, separate from design allocations.
The pinned single-clock/four-helper alias assumptions remain unchanged.

Checked mode emits commit/device traces and selected watch observations, with
oracle PC/instruction columns explicitly marked as not observed DUT values.
Final mode downloads only the endpoint and cumulative counters, plus validation
pool dumps after timing. It does not collect per-cycle traces. Run checked
acceptance first, then unprofiled batch timings; profile separate runs. Do not
compare entire state pools byte-for-byte across partition layouts.

## Provenance and Local Verification

`provenance/accepted-sources.json` records both evidence archive hashes, original
source hashes and the complete list of adapted snapshot files. Checked32 and
final32 runtime snapshots were byte-identical; only their build definitions and
runner differ. No separate runtime source tree is needed.

The adaptations are the ABI type alias and bounds, serialization selection and
cross-ABI guard, ABI-neutral load error, and isolated build options. The accepted
GPU device/final-cycle hooks, scheduling, launch resource decisions, runner and
oracle semantics are unchanged. No new per-cycle work is added to wide32.

Local checks on November 20, 2026 used GCC 13.3.0, Python 3.12.3 and Make 4.3:

```sh
python3 tests/run_host_checks.py
python3 tests/check_build.py
git diff --check
```

Passed: both frozen ABI layouts, same-ABI serialization round trips, cross-ABI
rejection, wide bytecode packer/evaluator boundaries with ASan/UBSan, host device
memory semantics, launch-resource test doubles, runner syntax in both ABIs, and
four ABI/architecture command plans. Fake tools exercise artifact identities and
failed-build invalidation only; they do not validate CUDA compilation.

The sandbox uses ptrace, so the actual sanitizer command was
`ASAN_OPTIONS=detect_leaks=0 python3 tests/run_host_checks.py`. ASan/UBSan remained
enabled; leak detection was not run. NVCC compilation, driver resource probes,
GPU correctness, timing, and profiling of this integrated branch: **not run**.
Previously accepted RTX PRO 6000 evidence is source provenance, not a substitute
for revalidating the integrated build on the target GPU.
