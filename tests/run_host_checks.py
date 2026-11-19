#!/usr/bin/env python3
"""Host checks only: no CUDA compiler, driver, or GPU execution is implied."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(*command, **kwargs):
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def main():
    provenance = json.loads((ROOT / "provenance/accepted-sources.json").read_text())
    for name, record in provenance["imports"].items():
        if name not in provenance["adapted_paths"]:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record["sha256"], name
    with tempfile.TemporaryDirectory(prefix="toucan-host-") as temporary:
        out = Path(temporary)
        for abi in ("legacy16", "wide32"):
            flags = ["-DTOUCAN_WIDE_POOL"] if abi == "wide32" else []
            exe = out / abi
            run("g++", "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all", "-fno-pie", "-no-pie", *flags,
                "-I", ROOT / "include", ROOT / "tests/abi.cpp", ROOT / "src/Serializer.cpp", "-o", exe)
            with (out / (abi + ".bin")).open("wb") as stream:
                run(exe, stdout=stream)
            layout = out / (abi + "-layout")
            run("g++", "-std=c++20", *flags, "-I", ROOT / "include",
                ROOT / "tests/layout.cpp", "-o", layout)
            actual = subprocess.check_output([layout], text=True)
            assert actual == (ROOT / "provenance" / (abi + "-layout.txt")).read_text(), abi
        run(out / "wide32", out / "legacy16.bin")
        run(out / "legacy16", out / "wide32.bin")
        exe = out / "devices"
        run("g++", "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all", "-fno-pie", "-no-pie",
            "-I", ROOT / "include", "-I", ROOT / "runner", ROOT / "tests/devices.cpp", "-o", exe)
        run(exe)
        sim = (ROOT / "src/SimEval.cu").read_text()
        spans = [("#define NETLIST_ALIGNMENT", "typedef struct {\n  uint32_t netlistOffset"),
                 ("__device__ __forceinline__ __int128 sign_extend_128", "/*\n__device__"),
                 ("__device__ void evalSingleMicroPart", "// Note: launch in 1D")]
        (out / "eval.inc").write_text("\n".join(
            sim[sim.index(start):sim.index(end, sim.index(start))] for start, end in spans))
        exe = out / "wide-eval"
        run("g++", "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all", "-fno-pie", "-no-pie", "-DTOUCAN_WIDE_POOL",
            "-I", ROOT / "include", "-I", out, ROOT / "tests/check_wide_pool.cpp",
            ROOT / "src/Serializer.cpp", "-o", exe)
        run(exe)
        sim = (ROOT / "src/ToucanSim.cu").read_text()
        (out / "setup.inc").write_text(sim[sim.index("int ToucanSimulator::setupGPU"):
                                          sim.index("int ToucanSimulator::init")])
        exe = out / "resources"
        run("g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-DCHECK_WIDE_POOL", "-I", out, ROOT / "tests/check_resources.cpp", "-o", exe)
        run(exe)
        # Parse the real runners with a minimal CUDA API double. This is not NVCC.
        (out / "cuda_runtime_api.h").write_text('''#pragma once
#include <cstddef>
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaMemcpyDeviceToHost = 1, cudaMemcpyHostToDevice = 2;
const char *cudaGetErrorString(cudaError_t);
cudaError_t cudaMemcpy(void *, const void *, size_t, int);
''')
        for abi in ("legacy16", "wide32"):
            flags = ["-DTOUCAN_WIDE_POOL"] if abi == "wide32" else []
            for name in ("checked", "final"):
                run("g++", "-std=c++20", "-DNANHU_GPU_DEVICES", *flags, "-fsyntax-only",
                    "-I", out, "-I", ROOT / "include", ROOT / "runner" / (name + ".cpp"))
    print("PASS host ABI layouts, serializer isolation, evaluator boundaries, resources and runner syntax")
    print("CUDA compilation and GPU execution: not run")


if __name__ == "__main__":
    main()
