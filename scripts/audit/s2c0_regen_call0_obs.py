"""S2c-0: regenerate the S2b call-0 observations of CF-Spatial tasks 0-12 x initial states 20-29 (no policy).

Runs in envs/libero-cf-client. S2b stored hashes, not observations. This replays the S2b client's exact sequence
(np.random.seed(7) once; tasks in ascending order, one env per task built by main_cf.py's _get_libero_env with seed 7;
for each state 20..29: env.reset(), set_init_state, scene fingerprint, 10 dummy wait steps, main_cf.py
preprocessing) and saves the call-0 policy input. Each scene is accepted only if its static geometry / sim-state
fingerprint AND its call-0 observation hash equal the S2b records of all four S2b runs (B and S, seeds 0 and 1).
States >= 30 are refused.

  MUJOCO_GL=egl envs/libero-cf-client/bin/python scripts/audit/s2c0_regen_call0_obs.py --out results/S2c0/obs
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO_ROOT / "configs/libero/cf"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from openpi_client import image_tools  # noqa: E402

from src.wrappers.geometry import scene_fingerprint  # noqa: E402

DUMMY = [0.0] * 6 + [-1.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s2b", default=str(REPO_ROOT / "results/S2b"))
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default=",".join(map(str, range(13))))
    p.add_argument("--init-start", type=int, default=20)
    p.add_argument("--episodes", type=int, default=10)
    a = p.parse_args()
    if a.init_start + a.episodes > 30:
        raise SystemExit("refusing: states >= 30 are reserved")
    spec = importlib.util.spec_from_file_location("main_cf", REPO_ROOT / "external/libero-cf/eval/main_cf.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)

    ref = {}
    for seed in (0, 1):
        for cond in ("B", "S"):
            for line in open(pathlib.Path(a.s2b) / f"seed{seed}" / cond / "episodes.jsonl"):
                r = json.loads(line)
                key = (r["task_id"], r["initial_state_index"])
                g = {k: v for k, v in r["geometry"].items() if "sha" in k}
                ref.setdefault(key, []).append((g, r["policy_calls"][0]["obs_sha"], r["policy_calls"][0]["t"]))

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(7)
    suite = benchmark.get_benchmark_dict()["libero_cf_spatial"]()
    report = []
    for task_id in map(int, a.tasks.split(",")):
        task = suite.get_task(task_id)
        inits = suite.get_task_init_states(task_id)
        env, prompt = cf._get_libero_env(task, 256, 7)
        for ep in range(a.init_start, a.init_start + a.episodes):
            env.reset()
            obs = env.set_init_state(inits[ep])
            geo = {k: v for k, v in scene_fingerprint(env, include_values=False).items() if "sha" in k}
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY)
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
            wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
            state = np.concatenate((obs["robot0_eef_pos"], cf._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
            sha = hashlib.sha256(img.tobytes() + wrist.tobytes() + np.asarray(state, np.float64).tobytes()).hexdigest()[:16]
            refs = ref[(task_id, ep)]
            row = {"task": task_id, "init": ep, "obs_sha": sha, "n_refs": len(refs),
                   "geometry_match_all": all(g == geo for g, _, _ in refs),
                   "obs_sha_match_all": all(s == sha for _, s, _ in refs)}
            report.append(row)
            np.savez(out / f"task{task_id:02d}_init{ep:03d}.npz", image=img, wrist_image=wrist, state=state,
                     prompt=np.asarray(prompt), obs_sha=np.asarray(sha))
            print(row, flush=True)
        env.close()
    summ = {"n": len(report), "geometry_match": sum(r["geometry_match_all"] for r in report),
            "obs_sha_match": sum(r["obs_sha_match_all"] for r in report), "rows": report}
    (out / "regen_report.json").write_text(json.dumps(summ, indent=1))
    print({k: v for k, v in summ.items() if k != "rows"})
    sys.exit(0 if summ["obs_sha_match"] == summ["n"] == summ["geometry_match"] else 1)


if __name__ == "__main__":
    main()
