"""S2d validation of the oracle reference-guidance policy, BEFORE any S2d rollout.

Runs in envs/openpi-cf with the deterministic XLA flags, on regenerated S2b call-0 observations
(results/S2c0/obs, hash-verified against S2b), CRN call-0 noise of seeds 0 and 1:
  V1  deterministic repetition: ref_norm / ref_raw actions bit-identical on repeat (fresh policy objects);
  V2  every pass (faithful, masked, reference) receives the identical CRN noise tensor (fingerprint);
  V3  faithful pass == openpi Policy.infer(l_f) == S2b B call-0 chunk, bit-for-bit (env space);
  V4  masked pass == LIBERO-CF CAG's masked pass; reference pass == Policy.infer(l_r), bit-for-bit;
  V5  ref_norm intervention norm (7 action dims, model space) == MASK's ||0.5 (A_f - A_u)|| at the call;
  V6  returned action == output_transform(A_f + s_mask * d) (ref_norm) / (A_f + 0.5 (A_f - A_r)) (ref_raw);
  V7  the server stack (CRNPolicy(ProvenancePolicy(ReferenceGuidedPolicy))) returns the same action as V6.
Writes the returned actions so a second process can be compared bitwise (--compare-to).

  XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  CUDA_VISIBLE_DEVICES=0 envs/openpi-cf/bin/python scripts/audit/s2d_reference_validation.py --out results/S2d_validation/a
"""

import argparse
import importlib.util
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

from src.wrappers import crn  # noqa: E402
from src.wrappers.reference_guidance import EPS, ReferenceGuidedPolicy  # noqa: E402

SCENES = [(1, 20), (4, 23), (9, 21), (12, 26)]


def cag_class():
    spec = importlib.util.spec_from_file_location("tf", REPO / "external/libero-cf/openpi-cf/scripts/serve_policy_TF.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.CounterfactualActionGuidedPolicy


def bit(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class Recorder:
    def __init__(self, policy):
        self.fn = policy._sample_actions
        self.calls = []
        policy._sample_actions = self

    def __call__(self, rng, observation, **kw):
        out = self.fn(rng, observation, **kw)
        self.calls.append({"noise": np.asarray(kw["noise"]), "out": np.asarray(out)})
        return out


def main():
    for f in ("--xla_gpu_autotune_level=0", "--xla_gpu_deterministic_ops=true"):
        assert f in os.environ.get("XLA_FLAGS", "")
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--compare-to", default=None)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = json.loads((REPO / "configs/cag/S2d_reference_prompts.json").read_text())["tasks"]
    s2b_b = {}
    for seed in (0, 1):
        for line in open(REPO / f"results/S2b/seed{seed}/B/episodes.jsonl"):
            r = json.loads(line)
            s2b_b[(seed, r["task_id"], r["initial_state_index"])] = np.asarray(r["predicted_chunks"][0], np.float64)

    cfg = _config.get_config("pi05_libero")
    base = _policy_config.create_trained_policy(cfg, "gs://openpi-assets/checkpoints/pi05_libero")
    cag = cag_class()(base, guidance_scale=1.5, uncond_empty_prompt=False)
    rec = Recorder(base)
    rows, saved = [], {}
    z8 = np.zeros(8, np.float32)
    for task, init in SCENES:
        d = np.load(REPO / f"results/S2c0/obs/task{task:02d}_init{init:03d}.npz")
        obs = {"observation/image": d["image"], "observation/wrist_image": d["wrist_image"],
               "observation/state": d["state"], "prompt": str(d["prompt"])}
        for seed in (0, 1):
            noise = crn.crn_noise(seed, task, init, 0)
            for kind in ("source", "other"):
                lr = refs[str(task)][kind]
                for mode in (("ref_norm", "ref_raw") if kind == "source" else ("ref_norm",)):
                    tag = f"t{task}_i{init}_s{seed}_{kind}_{mode}"
                    pol1 = ReferenceGuidedPolicy(base, mode=mode)
                    pol2 = ReferenceGuidedPolicy(base, mode=mode)
                    rec.calls.clear()
                    o1 = pol1.infer(dict(obs, s2d_ref_prompt=lr), noise=noise)
                    passes = list(rec.calls)
                    o2 = pol2.infer(dict(obs, s2d_ref_prompt=lr), noise=noise)
                    a_f, a_r = passes[0]["out"][0], passes[1]["out"][0]
                    rec.calls.clear()
                    b_act = base.infer(dict(obs), noise=noise)["actions"]
                    rec.calls.clear()
                    r_act = base.infer(dict(obs, prompt=lr), noise=noise)["actions"]
                    r_plain = rec.calls[0]["out"][0]
                    row = {"tag": tag,
                           "V1_repeat_bit": bit(o1["actions"], o2["actions"]),
                           "V2_same_noise_all_passes": all(bit(c["noise"], noise) for c in passes),
                           "V3_A_f_eq_B_infer": bit(np.asarray(o1["diag"]["s2d"]["a_conditioned"]), b_act),
                           "V3_A_f_eq_S2b_B_call0": bit(np.asarray(o1["diag"]["s2d"]["a_conditioned"]), s2b_b[(seed, task, init)]),
                           "V4_A_r_eq_plain_ref": bit(a_r, r_plain)}
                    if mode == "ref_norm":
                        a_u = passes[2]["out"][0]
                        rec.calls.clear()
                        cag.infer(dict(obs), noise=noise)
                        row["V4_A_u_eq_CAG_masked"] = bit(a_u, rec.calls[1]["out"][0])
                        s_mask = float(np.linalg.norm(0.5 * (a_f - a_u)[:, :7]))
                        row["V5_norm_match_abs_err"] = abs(o1["diag"]["s2d"]["intervention_norm"] - s_mask)
                        r = (a_f - a_r)[:, :7]
                        exp = a_f.copy()
                        exp[:, :7] = a_f[:, :7] + s_mask * r / (np.linalg.norm(r) + EPS)
                    else:
                        exp = a_f + 0.5 * (a_f - a_r)
                    exp_env = base._output_transform({"state": z8, "actions": exp})["actions"]
                    row["V6_action_recon_max_abs_err"] = float(np.max(np.abs(exp_env - o1["actions"])))
                    # V7 server stack
                    from src.wrappers.serve_cf_policy import ProvenancePolicy  # noqa: PLC0415
                    srv = crn.CRNPolicy(ProvenancePolicy(ReferenceGuidedPolicy(base, mode=mode), base, mode, 1.5), 0 if seed == 0 else 1)
                    so = srv.infer(dict(obs, s2d_ref_prompt=lr, crn={"master_seed": seed, "task_id": task, "init_id": init, "call_index": 0},
                                        diag_meta={}))
                    row["V7_server_stack_bit"] = bit(so["actions"], o1["actions"])
                    row["V7_noise_sha_eq_crn"] = so["crn"]["noise_sha"] == so["diag"]["s2d"]["noise_sha"]
                    rows.append(row)
                    saved[tag] = o1["actions"]
                    print(row, flush=True)
    np.savez(out / "actions.npz", **saved)
    summ = {"n": len(rows)}
    for k in rows[0]:
        if k.startswith("V") and isinstance(rows[0][k], bool):
            summ[k] = sum(r[k] for r in rows if k in r)
    norm_rows = [r for r in rows if "V5_norm_match_abs_err" in r]
    summ["V4_A_u_eq_CAG_masked"] = sum(r["V4_A_u_eq_CAG_masked"] for r in norm_rows)
    summ["n_ref_norm"] = len(norm_rows)
    summ["V5_max_abs_err"] = max(r["V5_norm_match_abs_err"] for r in norm_rows)
    summ["V6_max_abs_err"] = max(r["V6_action_recon_max_abs_err"] for r in rows)
    if a.compare_to:
        other = np.load(a.compare_to)
        summ["cross_process_bit_identical"] = [sum(bit(saved[k], other[k]) for k in saved), len(saved)]
    summ["PASS"] = (all(summ[k] == len(rows) for k in summ if k.startswith(("V1", "V2", "V3", "V7")))
                    and summ["V4_A_r_eq_plain_ref"] == len(rows) and summ["V4_A_u_eq_CAG_masked"] == len(norm_rows)
                    and summ["V5_max_abs_err"] < 1e-5 and summ["V6_max_abs_err"] < 1e-6
                    and (not a.compare_to or summ["cross_process_bit_identical"][0] == len(saved)))
    (out / "validation.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))
    sys.exit(0 if summ["PASS"] else 1)


if __name__ == "__main__":
    main()
