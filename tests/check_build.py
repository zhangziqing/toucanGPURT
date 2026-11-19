#!/usr/bin/env python3
"""Verify the build contract using dry runs and explicit fake-tool fixtures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def build(*args, runtime=ROOT, **kwargs):
    return subprocess.run([sys.executable, runtime / "build_nanhu.py", *map(str, args)],
                          cwd=runtime, capture_output=True, text=True, **kwargs)


def main():
    with tempfile.TemporaryDirectory(prefix="toucan-build-contract-") as temporary:
        base = Path(temporary)
        for abi in ("legacy16", "wide32"):
            for arch in ("sm_86", "sm_120"):
                out = base / (abi + "-" + arch)
                result = build("--abi", abi, "--arch", arch, "--output", out, "--dry-run", check=True)
                plan = json.loads(result.stdout)
                assert not out.exists() and not plan["gpu_executed"]
                assert plan["binaries"] == {"checked": "checked/nanhu-checked",
                    "batch_final": "final/nanhu-final", "resources": "final/resource-probe"}
                for key, path in (("checked_binary", "checked/nanhu-checked"),
                                  ("final_binary", "final/nanhu-final"),
                                  ("resource_probe", "final/resource-probe")):
                    assert Path(plan[key]["path"]) == out / path
                for command in plan["commands"]:
                    if command[1] == "rcs":
                        continue
                    assert "-arch=" + arch in command
                    assert ("-DTOUCAN_WIDE_POOL" in command) == (abi == "wide32")
                    assert "-DNANHU_GPU_DEVICES" in command
                    if "-c" in command:
                        assert "--maxrregcount=64" in command
                        final = str(out / "final") in command[-1]
                        assert ("-DNANHU_FINAL_ONLY" in command) == final
                make = subprocess.check_output(
                    ["make", "-n", "all", "ABI=" + abi, "ARCH=" + arch], cwd=ROOT, text=True)
                assert str(Path("build") / (abi + "-" + arch)) in make
                assert ("-DTOUCAN_WIDE_POOL" in make) == (abi == "wide32")
        tools = base / "tools"
        tools.mkdir()
        fake = '''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
if "--version" in sys.argv:
    print("FAKE NVCC build-contract fixture; not a CUDA compiler")
elif os.environ.get("FAIL_FIXTURE") and "-c" in sys.argv:
    sys.exit(23)
else:
    if os.environ.get("MUTATE_SOURCE") and "-c" in sys.argv:
        source = Path(os.environ["MUTATE_SOURCE"])
        source.write_bytes(source.read_bytes() + b"\\n")
    output = sys.argv[2] if sys.argv[1] == "rcs" else sys.argv[sys.argv.index("-o") + 1]
    Path(output).write_bytes(b"FAKE build-contract artifact\\n")
'''
        for name in ("nvcc", "ar"):
            path = tools / name
            path.write_text(fake)
            path.chmod(0o755)
        env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"]}
        out = base / "fixture"
        args = ("--abi", "wide32", "--arch", "sm_86", "--output", out, "--nvcc", tools / "nvcc")
        build(*args, env=env, check=True)
        manifest = json.loads((out / "build.json").read_text())
        assert manifest["status"] == "built" and not manifest["gpu_executed"]
        assert "FAKE" in manifest["nvcc"]["version"]
        assert manifest["source_revision"] == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        assert isinstance(manifest["source_status"], str)
        assert manifest["binaries"] == {"checked": "checked/nanhu-checked",
            "batch_final": "final/nanhu-final", "resources": "final/resource-probe"}
        assert len(manifest["build_steps"]) == 15
        for key in ("checked_binary", "final_binary", "resource_probe"):
            record = manifest[key]
            path = Path(record["path"])
            assert record["bytes"] == path.stat().st_size
            assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        for embedded in (False, True):
            parent = base / ("unrelated-repo" if embedded else "standalone")
            parent.mkdir()
            if embedded:
                subprocess.run(["git", "-c", "init.templateDir=", "init", "-q", str(parent)], check=True)
                subprocess.run(["git", "-c", "user.name=Build Fixture", "-c", "user.email=fixture@example.invalid",
                                "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "fixture"],
                               cwd=parent, check=True)
            runtime = parent / "runtime"
            runtime.mkdir()
            shutil.copy2(ROOT / "build_nanhu.py", runtime / "build_nanhu.py")
            for folder in ("include", "src", "runner", "provenance"):
                shutil.copytree(ROOT / folder, runtime / folder)
            output = parent / "output"
            packaged_args = ("--abi", "wide32", "--arch", "sm_86", "--output", output,
                             "--nvcc", tools / "nvcc")
            build(*packaged_args, runtime=runtime, env=env, check=True)
            packaged = json.loads((output / "build.json").read_text())
            assert packaged["status"] == "built" and len(packaged["build_steps"]) == 15
            assert packaged["source_revision"] is None and packaged["source_status"] is None
            provenance = runtime / "provenance/accepted-sources.json"
            assert packaged["sources"]["provenance/accepted-sources.json"]["sha256"] == hashlib.sha256(
                provenance.read_bytes()).hexdigest()
            for key in ("checked_binary", "final_binary", "resource_probe"):
                record = packaged[key]
                assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
            changed = build(*packaged_args, runtime=runtime, env={**env, "MUTATE_SOURCE": str(provenance)})
            assert changed.returncode and "Source changed during build" in changed.stderr
            assert not (output / "build.json").exists()
        failed = build(*args, env={**env, "FAIL_FIXTURE": "1"})
        assert failed.returncode and not (out / "build.json").exists()
        assert json.loads((out / "commands.json").read_text())[-1]["returncode"] == 23
        assert build("--abi", "bad", "--arch", "sm_86", "--output", out).returncode
        assert build("--abi", "legacy16", "--arch", "sm_86", "--output", ROOT).returncode
    print("PASS four ABI/architecture plans, Make isolation, manifest hashes and failed-build invalidation")
    print("PASS exact-root Git provenance, standalone exports, unrelated parents and frozen source rejection")
    print("Fake compiler used only for build-contract checks; CUDA compilation: not run")


if __name__ == "__main__":
    main()
