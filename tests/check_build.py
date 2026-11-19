#!/usr/bin/env python3
"""Verify the build contract using dry runs and explicit fake-tool fixtures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def build(*args, **kwargs):
    return subprocess.run([sys.executable, ROOT / "build_nanhu.py", *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, **kwargs)


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
        assert manifest["binaries"] == {"checked": "checked/nanhu-checked",
            "batch_final": "final/nanhu-final", "resources": "final/resource-probe"}
        assert len(manifest["build_steps"]) == 15
        for key in ("checked_binary", "final_binary", "resource_probe"):
            record = manifest[key]
            path = Path(record["path"])
            assert record["bytes"] == path.stat().st_size
            assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        failed = build(*args, env={**env, "FAIL_FIXTURE": "1"})
        assert failed.returncode and not (out / "build.json").exists()
        assert json.loads((out / "commands.json").read_text())[-1]["returncode"] == 23
        assert build("--abi", "bad", "--arch", "sm_86", "--output", out).returncode
        assert build("--abi", "legacy16", "--arch", "sm_86", "--output", ROOT).returncode
    print("PASS four ABI/architecture plans, Make isolation, manifest hashes and failed-build invalidation")
    print("Fake compiler used only for build-contract checks; CUDA compilation: not run")


if __name__ == "__main__":
    main()
