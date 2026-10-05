"""S2c-0 controlled language query of the frozen pi0.5 (configs/cag/S2c0_prereg.yaml). No rollouts.

Runs in envs/openpi-cf with the deterministic XLA flags. For every regenerated S2b call-0 observation
(results/S2c0/obs, hash-verified against S2b) and every noise seed (CRN master seeds 0, 1 = S2b call 0; 1000..1007
secondary), the SAME noise tensor is passed through the public `noise=` argument for every language condition:
  f, b, o1, o2 : openpi Policy.infer(obs with that benchmark instruction)          -> pre-guidance chunk
  u            : LIBERO-CF CounterfactualActionGuidedPolicy.infer(obs with l_f)      -> its language-masked pass
A pass-through recorder on the jitted sampler captures the model-space chunk (and the noise it received) of every
pass; nothing is changed. Checks per query: noise fingerprint identical across conditions; CAG's conditioned pass
bit-identical to the plain l_f chunk; for seeds 0/1, A_f (env) == S2b B call-0 chunk == S2b S call-0 a_conditioned and
A_u (env) == S2b S call-0 a_unconditioned, bit-for-bit.

  XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  CUDA_VISIBLE_DEVICES=0 envs/openpi-cf/bin/python scripts/run_cag/s2c0_language_query.py --out results/S2c0/chunks
"""

import argparse
import importlib.util
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

from src.wrappers import crn  # noqa: E402
from src.wrappers.manifest import write_manifest  # noqa: E402

SERVE_TF = REPO_ROOT / "external/libero-cf/openpi-cf/scripts/serve_policy_TF.py"
SEEDS = [0, 1] + list(range(1000, 1008))
CONDS = ["f", "b", "o1", "o2", "u"]


def _cag_class():
    spec = importlib.util.spec_from_file_location("serve_policy_TF", SERVE_TF)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.CounterfactualActionGuidedPolicy


class Recorder:
    def __init__(self, policy):
        self.fn = policy._sample_actions
        self.calls = []
        policy._sample_actions = self

    def __call__(self, rng, observation, **kw):
        out = self.fn(rng, observation, **kw)
        self.calls.append({"noise": np.asarray(kw["noise"]), "out": np.asarray(out)})
        return out


def bit(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main():
    for f in ("--xla_gpu_autotune_level=0", "--xla_gpu_deterministic_ops=true"):
        assert f in os.environ.get("XLA_FLAGS", ""), f"XLA_FLAGS must contain {f}"
    p = argparse.ArgumentParser()
    p.add_argument("--obs", default=str(REPO_ROOT / "results/S2c0/obs"))
    p.add_argument("--s2b", default=str(REPO_ROOT / "results/S2b"))
    p.add_argument("--lang", default=str(REPO_ROOT / "configs/cag/S2c0_language_conditions.json"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lang = json.loads(pathlib.Path(a.lang).read_text())["tasks"]

    s2b = {}
    for seed in (0, 1):
        for cond in ("B", "S"):
            for line in open(pathlib.Path(a.s2b) / f"seed{seed}" / cond / "episodes.jsonl"):
                r = json.loads(line)
                c0 = r["policy_calls"][0]
                rec = {"chunk": r["predicted_chunks"][0], "noise_sha": c0["crn"]["noise_sha"], "obs_sha": c0["obs_sha"]}
                if cond == "S":
                    rec["a_cond"] = c0["diag"]["guidance"]["a_conditioned"]
                    rec["a_uncond"] = c0["diag"]["guidance"]["a_unconditioned"]
                s2b[(seed, cond, r["task_id"], r["initial_state_index"])] = rec

    cfg = _config.get_config("pi05_libero")
    base = _policy_config.create_trained_policy(cfg, "gs://openpi-assets/checkpoints/pi05_libero")
    cag = _cag_class()(base, guidance_scale=1.5, uncond_empty_prompt=False)
    rec = Recorder(base)
    state0 = np.zeros(8, np.float32)
    checks = []
    for task in range(13):
        L = lang[str(task)]
        prompts = {"f": L["l_f"], "b": L["l_b"], "o1": L["l_o"][0], "o2": L["l_o"][1]}
        for init in range(20, 30):
            d = np.load(pathlib.Path(a.obs) / f"task{task:02d}_init{init:03d}.npz")
            assert str(d["prompt"]) == L["l_f"], (task, str(d["prompt"]))
            obs = {"observation/image": d["image"], "observation/wrist_image": d["wrist_image"],
                   "observation/state": d["state"]}
            model = np.zeros((len(SEEDS), len(CONDS), 10, 7), np.float32)
            env = np.zeros((len(SEEDS), len(CONDS), 10, 7), np.float64)
            noise_sha = []
            row = {"task": task, "init": init, "obs_sha": str(d["obs_sha"])}
            ok_noise = ok_cagcond = True
            for si, seed in enumerate(SEEDS):
                noise = crn.crn_noise(seed, task, init, 0)
                noise_sha.append(crn.fingerprint(noise))
                for ci, c in enumerate(CONDS):
                    rec.calls.clear()
                    if c == "u":
                        res = cag.infer(dict(obs, prompt=L["l_f"]), noise=noise)
                        cond_pass, unc = rec.calls[0], rec.calls[1]
                        ok_cagcond &= bit(cond_pass["out"][0, :, :7], model[si, 0])
                        ok_noise &= bit(cond_pass["noise"], noise) and bit(unc["noise"], noise)
                        raw = unc["out"][0]
                        env[si, ci] = base._output_transform({"state": state0, "actions": raw.astype(np.float32).copy()})["actions"]
                    else:
                        res = base.infer(dict(obs, prompt=prompts[c]), noise=noise)
                        raw = rec.calls[0]["out"][0]
                        ok_noise &= bit(rec.calls[0]["noise"], noise)
                        env[si, ci] = res["actions"]
                    model[si, ci] = raw[:, :7]
            row["noise_identical_across_conditions"] = bool(ok_noise)
            row["cag_cond_pass_eq_plain_f"] = bool(ok_cagcond)
            for si, seed in enumerate((0, 1)):
                B, S = s2b[(seed, "B", task, init)], s2b[(seed, "S", task, init)]
                row[f"seed{seed}"] = {
                    "obs_sha_eq_s2b": B["obs_sha"] == row["obs_sha"],
                    "noise_sha_eq_s2b": B["noise_sha"] == noise_sha[si] == S["noise_sha"],
                    "A_f_eq_s2b_B_chunk": bit(env[si, 0], np.asarray(B["chunk"], np.float64)),
                    "A_f_eq_s2b_S_a_cond": bit(env[si, 0], np.asarray(S["a_cond"], np.float64)),
                    "A_u_eq_s2b_S_a_uncond": bit(env[si, 4], np.asarray(S["a_uncond"], np.float64)),
                }
            checks.append(row)
            np.savez(out / f"task{task:02d}_init{init:03d}.npz", model=model, env=env, seeds=np.asarray(SEEDS),
                     conds=np.asarray(CONDS), prompts=np.asarray([prompts[c] for c in CONDS[:4]]),
                     noise_sha=np.asarray(noise_sha))
            print(task, init, {k: v for k, v in row.items() if k not in ("task", "init", "obs_sha")}, flush=True)
    flat = lambda k: sum(r[f"seed{s}"][k] for r in checks for s in (0, 1))  # noqa: E731
    summ = {"n_scenes": len(checks),
            "noise_identical_across_conditions": sum(r["noise_identical_across_conditions"] for r in checks),
            "cag_cond_pass_eq_plain_f": sum(r["cag_cond_pass_eq_plain_f"] for r in checks),
            **{k: flat(k) for k in ("obs_sha_eq_s2b", "noise_sha_eq_s2b", "A_f_eq_s2b_B_chunk",
                                    "A_f_eq_s2b_S_a_cond", "A_u_eq_s2b_S_a_uncond")},
            "n_seed_scene_checks": 2 * len(checks)}
    summ["PASS"] = (summ["noise_identical_across_conditions"] == summ["cag_cond_pass_eq_plain_f"] == len(checks)
                    and all(summ[k] == 2 * len(checks) for k in ("obs_sha_eq_s2b", "noise_sha_eq_s2b", "A_f_eq_s2b_B_chunk",
                                                                  "A_f_eq_s2b_S_a_cond", "A_u_eq_s2b_S_a_uncond")))
    (out / "query_checks.json").write_text(json.dumps({"summary": summ, "rows": checks}, indent=1))
    write_manifest(out, config=vars(a), checkpoint="gs://openpi-assets/checkpoints/pi05_libero", condition="S2c0-language-query",
                   seed={"crn_master_seeds": SEEDS, "call_index": 0}, task={"suite": "libero_cf_spatial", "task_ids": list(range(13))},
                   initial_state_indices=range(20, 30), extra={"summary": summ, "xla_flags": os.environ.get("XLA_FLAGS")})
    print(json.dumps(summ, indent=1))
    sys.exit(0 if summ["PASS"] else 1)


if __name__ == "__main__":
    main()
