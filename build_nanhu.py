#!/usr/bin/env python3
"""Build both accepted Nanhu paths without running CUDA kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def identity(path):
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def source_git_metadata():
    unavailable = {"source_revision": None, "source_status": None}
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True)
        if top.returncode or Path(top.stdout.strip()).resolve() != ROOT:
            return unavailable
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True)
        if revision.returncode or status.returncode:
            return unavailable
    except OSError:
        return unavailable
    return {"source_revision": revision.stdout.strip(), "source_status": status.stdout}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abi", choices=("legacy16", "wide32"), required=True)
    parser.add_argument("--arch", choices=("sm_86", "sm_120"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT or out in ROOT.parents:
        parser.error("--output must not be the source directory or its ancestor")
    nvcc = shutil.which(args.nvcc)
    if not nvcc and not args.dry_run:
        parser.error("NVCC not found: " + args.nvcc)
    nvcc = nvcc or args.nvcc
    ar = shutil.which("ar") or "ar"
    common = ["-std=c++20", "-arch=" + args.arch, "-O3", "-lineinfo",
              "-DNANHU_GPU_DEVICES", "-I" + str(ROOT / "include")]
    if args.abi == "wide32":
        common.append("-DTOUCAN_WIDE_POOL")
    sources = sorted((ROOT / "src").glob("*.cpp")) + sorted((ROOT / "src").glob("*.cu"))
    commands = []
    binaries = {}
    for mode in ("checked", "final"):
        directory = out / mode
        objects = [directory / (source.stem + ".o") for source in sources]
        flags = common + (["-DNANHU_FINAL_ONLY"] if mode == "final" else [])
        for source, obj in zip(sources, objects):
            commands.append([nvcc, *flags, "--maxrregcount=64", "-c", str(source), "-o", str(obj)])
        library = directory / "libToucanGPURT.a"
        commands.append([ar, "rcs", str(library), *map(str, objects)])
        binary = directory / ("nanhu-" + mode)
        commands.append([nvcc, *common, "-I" + str(ROOT / "runner"),
                         str(ROOT / "runner" / (mode + ".cpp")), str(library), "-o", str(binary)])
        binaries[mode + "_binary"] = binary
        if mode == "final":
            probe = directory / "resource-probe"
            commands.append([nvcc, *common, str(ROOT / "runner/resources.cu"),
                             str(library), "-o", str(probe)])
            binaries["resource_probe"] = probe
    plan = {"schema": 1, "abi": args.abi, "arch": args.arch,
            "binaries": {"checked": "checked/nanhu-checked", "batch_final": "final/nanhu-final",
                         "resources": "final/resource-probe"},
            "commands": commands, **{key: {"path": str(path)} for key, path in binaries.items()},
            "gpu_executed": False}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    # Do not leave a previously successful manifest after a failed rebuild.
    (out / "build.json").unlink(missing_ok=True)
    for mode in ("checked", "final"):
        (out / mode).mkdir(exist_ok=True)
        (out / mode / "libToucanGPURT.a").unlink(missing_ok=True)
    tracked = [ROOT / "build_nanhu.py", ROOT / "provenance/accepted-sources.json", *sources]
    for folder in ("include", "runner"):
        tracked += sorted((ROOT / folder).glob("*"))
    source_ids = {str(path.relative_to(ROOT)): identity(path) for path in tracked if path.is_file()}
    version = subprocess.check_output([nvcc, "--version"], text=True).strip()
    records = []
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for index, command in enumerate(commands):
        print(subprocess.list2cmdline(command), flush=True)
        started = time.monotonic()
        log = out / (f"command-{index:02d}.log")
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        records.append({"command": command, "seconds": time.monotonic() - started,
                        "returncode": result.returncode, "log": identity(log)})
        (out / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError("Build failed; see " + str(log))
    if source_ids != {name: identity(ROOT / name) for name in source_ids}:
        raise RuntimeError("Source changed during build; build.json was not written")
    manifest = {**plan, "status": "built", "nvcc": {"path": nvcc, "version": version},
                **source_git_metadata(),
                "sources": source_ids, "build_steps": records,
                **{key: identity(path) for key, path in binaries.items()}}
    temporary = out / "build.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, out / "build.json")
    print(out / "build.json")


if __name__ == "__main__":
    main()
