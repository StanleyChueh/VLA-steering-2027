"""Run manifests: every experimental run writes one JSON describing exactly what produced its results.

Kept dependency-free (stdlib only) so it can be imported from all three environments
(py3.8 LIBERO-CF client, py3.10 SCALE, py3.11 openpi server).
"""

import datetime
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_REPOS = ("scale", "libero-cf", "mechanistic-steering-vlas", "vla-explain", "deps/LIBERO")


def _run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001 - provenance capture must never kill a run
        return f"<unavailable: {e}>"


def git_sha(path):
    sha = _run(["git", "rev-parse", "HEAD"], cwd=path)
    dirty = _run(["git", "status", "--porcelain"], cwd=path)
    return {"sha": sha, "dirty": bool(dirty)}


def gpu_info():
    query = "index,name,driver_version,memory.total,memory.used"
    out = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    cuda = ""
    for line in _run(["nvidia-smi"]).splitlines():
        if "CUDA Version" in line:
            cuda = line.split("CUDA Version:")[1].strip(" |")
    return {
        "nvidia_smi": out.splitlines(),
        "driver_cuda_version": cuda,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def framework_versions():
    versions = {}
    for mod in ("torch", "jax", "jaxlib", "flax", "transformers", "numpy", "robosuite", "mujoco", "tensorflow"):
        if mod in sys.modules:
            versions[mod] = getattr(sys.modules[mod], "__version__", "?")
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "version"):
        versions["torch_cuda"] = getattr(torch.version, "cuda", None)
    return versions


def write_manifest(out_dir, *, config, checkpoint, condition, seed, task, initial_state_indices, extra=None):
    """Write <out_dir>/manifest.json and return its path. Call after heavy imports so versions are captured."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "cwd": os.getcwd(),
        "host": platform.node(),
        "python": sys.version,
        "our_repo": git_sha(REPO_ROOT),
        "external_repos": {r: git_sha(REPO_ROOT / "external" / r) for r in EXTERNAL_REPOS},
        "checkpoint": checkpoint,
        "condition": condition,
        "seed": seed,
        "task": task,
        "initial_state_indices": list(initial_state_indices),
        "config": config,
        "gpu": gpu_info(),
        "frameworks": framework_versions(),
        "extra": extra or {},
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
