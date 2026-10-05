"""S2b common-random-number (CRN) validation, run BEFORE any S2b rollout (reports/S2B_RNG_VALIDATION.md).

Runs in envs/openpi-cf on one GPU, on real LIBERO-CF observations from already-used initial states (state 0 of
tasks 0/6/12; no S2b state is touched). Checks, with the policy objects exactly as served:

  N1  deterministic repeat: crn_noise(tuple) is bit-identical on repeat.
  N2  every key field matters: changing master_seed / task / init / call changes the noise (no shared element).
  N3  marginal distribution preserved: CRN noise vs native openpi draws (jax.random.normal on split keys),
      moments + two-sample KS; and no correlation between noise of neighbouring tuples.
  N4  episode-length independence: the noise for (m, task, init, call) does not depend on how many calls were
      made before (stateless derivation; exercised by generating streams after different-length prefixes).
  P1  stock path (instrumentation never installed): B = openpi Policy.infer(obs, noise=crn),
      S = LIBERO-CF CounterfactualActionGuidedPolicy.infer(obs, noise=crn). Repeat -> bit-identical actions.
  P2  master_seed 0 vs 1 -> different actions.
  P3  B's sampler output == S's conditioned-branch sampler output, bit-for-bit, for the same (obs, tuple); B's
      returned env action == output_transform(S conditioned branch); S's cond and uncond passes receive the same
      noise tensor (the public CAG process); guided = uncond + w (cond - uncond).
  P4  the policy's own RNG stream (base._rng) is irrelevant once noise is supplied (perturbing it changes nothing).
  P5  production path (pi05_attention.install() + DiagnosticPolicy + CRNPolicy, i.e. the S2b server stack, fed the
      CRN tuple through obs["crn"]) returns actions bit-identical to the stock path; diagnostic noise fingerprints
      equal the CRN fingerprint; diagnostic re-runs reproduce each branch bit-for-bit.

`--stock-only` runs N*/P1-P4 and writes the stock actions, so a second process (or the other GPU) can be compared
for cross-process determinism with `--compare-to`.

  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 envs/openpi-cf/bin/python \
      scripts/audit/s2b_crn_validation.py --out results/S2b_rng_validation/gpu0
"""

import argparse
import importlib.util
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from scipy import stats  # noqa: E402

from src.wrappers import crn  # noqa: E402
from src.wrappers.manifest import write_manifest  # noqa: E402

SERVE_TF_PATH = REPO_ROOT / "external/libero-cf/openpi-cf/scripts/serve_policy_TF.py"
OBS = [
    ("results/S0C/obs_cf_spatial_t0_i0.npz", 0, 0),
    ("results/S2_identity/obs_cf_spatial_t6_i0.npz", 6, 0),
    ("results/S2_identity/obs_cf_spatial_t12_i0.npz", 12, 0),
]
CALLS = (0, 1, 7)  # call indices exercised per observation


def _cag_class():
    spec = importlib.util.spec_from_file_location("libero_cf_serve_policy_TF", SERVE_TF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CounterfactualActionGuidedPolicy


def _load_obs(path):
    d = np.load(REPO_ROOT / path)
    return {
        "observation/image": d["image"], "observation/wrist_image": d["wrist_image"],
        "observation/state": d["state"], "prompt": str(d["prompt"]),
    }


def _bit(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def noise_checks():
    r = {}
    t = dict(master_seed=0, task_id=3, init_id=25, call_index=4)
    r["N1_repeat_bit_identical"] = _bit(crn.crn_noise(**t), crn.crn_noise(**t))
    base = crn.crn_noise(**t)
    changed = {}
    for f in crn.KEY_FIELDS:
        u = dict(t)
        u[f] += 1
        o = crn.crn_noise(**u)
        changed[f] = {"differs": not _bit(base, o), "shared_elements": int(np.sum(base == o))}
    r["N2_field_changes"] = changed
    r["N2_master_seed_0_vs_1_differs"] = changed["master_seed"]["differs"]

    # N3: marginal distribution. 1300 S2b-shaped tuples (13 tasks x 10 states x 10 calls) vs native draws.
    tuples = [(m, k, i, c) for m in (0,) for k in range(13) for i in range(20, 30) for c in range(10)]
    pool = np.concatenate([crn.crn_noise(*x).ravel() for x in tuples])
    key = jax.random.key(0)
    native = []
    for _ in range(len(tuples)):  # openpi Policy.infer: self._rng, sample_rng = split(self._rng); normal(sample_rng)
        key, sub = jax.random.split(key)
        native.append(np.asarray(jax.random.normal(sub, crn.NOISE_SHAPE), np.float32).ravel())
    native = np.concatenate(native)
    ks = stats.ks_2samp(pool, native)
    ks_norm = stats.kstest(pool, "norm")
    neigh = [np.corrcoef(crn.crn_noise(0, 3, 25, c).ravel(), crn.crn_noise(0, 3, 25, c + 1).ravel())[0, 1] for c in range(50)]
    seeds = [np.corrcoef(crn.crn_noise(0, k, 25, 0).ravel(), crn.crn_noise(1, k, 25, 0).ravel())[0, 1] for k in range(13)]
    r["N3_marginal"] = {
        "n_values": int(pool.size), "dtype": str(pool.dtype), "shape_per_call": list(crn.NOISE_SHAPE),
        "crn": {"mean": float(pool.mean()), "std": float(pool.std()), "skew": float(stats.skew(pool)),
                "excess_kurtosis": float(stats.kurtosis(pool))},
        "native": {"mean": float(native.mean()), "std": float(native.std()), "skew": float(stats.skew(native)),
                   "excess_kurtosis": float(stats.kurtosis(native))},
        "ks_2samp_crn_vs_native": {"stat": float(ks.statistic), "p": float(ks.pvalue)},
        "ks_crn_vs_N01": {"stat": float(ks_norm.statistic), "p": float(ks_norm.pvalue)},
        "corr_neighbouring_call_index": {"mean": float(np.mean(neigh)), "max_abs": float(np.max(np.abs(neigh)))},
        "corr_master_seed_0_vs_1": {"mean": float(np.mean(seeds)), "max_abs": float(np.max(np.abs(seeds)))},
        "expected_abs_corr_scale_1_over_sqrt_320": float(1 / np.sqrt(320)),
    }
    # N4: stream for episode (init 21) after a 3-call vs a 40-call previous episode (init 20).
    def run(prev_len):
        _ = [crn.crn_noise(0, 5, 20, c) for c in range(prev_len)]
        return [crn.fingerprint(crn.crn_noise(0, 5, 21, c)) for c in range(12)]
    r["N4_episode_length_independent"] = run(3) == run(40)
    r["PASS"] = (
        r["N1_repeat_bit_identical"] and all(v["differs"] and v["shared_elements"] == 0 for v in changed.values())
        and abs(r["N3_marginal"]["crn"]["mean"]) < 0.01 and abs(r["N3_marginal"]["crn"]["std"] - 1) < 0.01
        and ks.pvalue > 0.001 and r["N4_episode_length_independent"]
    )
    return r


class Recorder:
    """Pure pass-through around the jitted sampler: records (noise, output) of each call; changes nothing."""

    def __init__(self, policy):
        self.fn = policy._sample_actions
        self.calls = []
        policy._sample_actions = self

    def __call__(self, rng, observation, **kw):
        out = self.fn(rng, observation, **kw)
        self.calls.append({"noise": np.asarray(kw.get("noise")), "out": np.asarray(out)})
        return out


def stock_checks(args, observations):
    cfg = _config.get_config("pi05_libero")
    base = _policy_config.create_trained_policy(cfg, args.checkpoint)
    cag = _cag_class()(base, guidance_scale=args.guidance_scale, uncond_empty_prompt=False)
    rec = Recorder(base)
    saved, r = {}, {"P1": [], "P2": [], "P3": [], "P4": []}
    for (path, task, init), obs in zip(OBS, observations):
        for call in CALLS:
            tag = f"t{task}_i{init}_c{call}"
            noise = crn.crn_noise(0, task, init, call)
            rec.calls.clear()
            b1 = base.infer(dict(obs), noise=noise)["actions"]
            b_raw = rec.calls[-1]["out"]
            b2 = base.infer(dict(obs), noise=noise)["actions"]
            rec.calls.clear()
            s1 = cag.infer(dict(obs), noise=noise)["actions"]
            s_cond, s_uncond = rec.calls[0], rec.calls[1]
            s2 = cag.infer(dict(obs), noise=noise)["actions"]
            r["P1"].append({"tag": tag, "B_repeat_bit": _bit(b1, b2), "S_repeat_bit": _bit(s1, s2)})
            b_seed1 = base.infer(dict(obs), noise=crn.crn_noise(1, task, init, call))["actions"]
            s_seed1 = cag.infer(dict(obs), noise=crn.crn_noise(1, task, init, call))["actions"]
            r["P2"].append({"tag": tag, "B_differs": not _bit(b1, b_seed1), "S_differs": not _bit(s1, s_seed1),
                            "B_max_abs_diff": float(np.max(np.abs(b1 - b_seed1)))})
            state = np.zeros(8, np.float32)
            cond_env = np.asarray(base._output_transform({"state": state, "actions": np.asarray(s_cond["out"][0], np.float32).copy()})["actions"])
            uncond_env = np.asarray(base._output_transform({"state": state, "actions": np.asarray(s_uncond["out"][0], np.float32).copy()})["actions"])
            r["P3"].append({
                "tag": tag,
                "B_sampler_out_eq_S_cond_sampler_out_bit": _bit(b_raw, s_cond["out"]),
                "B_env_action_eq_S_cond_env_bit": _bit(b1, cond_env),
                "S_cond_uncond_same_noise_bit": _bit(s_cond["noise"], s_uncond["noise"]),
                "S_noise_eq_crn_bit": _bit(s_cond["noise"], noise),
                "S_cond_ne_uncond": not _bit(s_cond["out"], s_uncond["out"]),
                "guided_reconstruction_max_abs_err": float(np.max(np.abs(uncond_env + args.guidance_scale * (cond_env - uncond_env) - s1))),
            })
            base._rng = jax.random.key(987654321)  # perturb the policy's own stream
            b_p = base.infer(dict(obs), noise=noise)["actions"]
            s_p = cag.infer(dict(obs), noise=noise)["actions"]
            r["P4"].append({"tag": tag, "B_bit": _bit(b1, b_p), "S_bit": _bit(s1, s_p)})
            saved[f"B_{tag}"], saved[f"S_{tag}"] = b1, s1
            saved[f"Bseed1_{tag}"], saved[f"Sseed1_{tag}"] = b_seed1, s_seed1
    r["P1_PASS"] = all(x["B_repeat_bit"] and x["S_repeat_bit"] for x in r["P1"])
    r["P2_PASS"] = all(x["B_differs"] and x["S_differs"] for x in r["P2"])
    r["P3_PASS"] = all(
        x["B_sampler_out_eq_S_cond_sampler_out_bit"] and x["B_env_action_eq_S_cond_env_bit"]
        and x["S_cond_uncond_same_noise_bit"] and x["S_noise_eq_crn_bit"] and x["S_cond_ne_uncond"]
        and x["guided_reconstruction_max_abs_err"] < 1e-5 for x in r["P3"]
    )
    r["P4_PASS"] = all(x["B_bit"] and x["S_bit"] for x in r["P4"])
    base._sample_actions = rec.fn
    return r, saved


def production_checks(args, observations, stock_saved):
    from src.instrumentation import pi05_attention as attn  # noqa: PLC0415

    attn.install()
    from src.instrumentation.pi05_diagnostics import DiagnosticPolicy  # noqa: PLC0415

    cfg = _config.get_config("pi05_libero")
    r = {"per_call": []}
    for mode in ("vanilla", "cag_tf"):
        inst = _policy_config.create_trained_policy(cfg, args.checkpoint)
        pol = inst if mode == "vanilla" else _cag_class()(inst, guidance_scale=args.guidance_scale, uncond_empty_prompt=False)
        w = None if mode == "vanilla" else args.guidance_scale
        server = crn.CRNPolicy(DiagnosticPolicy(pol, inst, mode=mode, guidance_scale=w, attention=True), master_seed=0)
        for (path, task, init), obs in zip(OBS, observations):
            for call in CALLS:
                tag = f"t{task}_i{init}_c{call}"
                o = dict(obs)
                o["crn"] = {"master_seed": 0, "task_id": task, "init_id": init, "call_index": call}
                o["diag_meta"] = {}
                out = server.infer(o)
                ref = stock_saved[("B_" if mode == "vanilla" else "S_") + tag]
                br = out["diag"]["branches"]
                row = {
                    "mode": mode, "tag": tag,
                    "actions_bit_identical_to_stock": _bit(out["actions"], ref),
                    "crn_noise_sha": out["crn"]["noise_sha"],
                    "diag_noise_sha_eq_crn": all(b["noise_sha"] == out["crn"]["noise_sha"] for b in br.values()),
                    "diag_x0_bit_identical": all(b["x0_bit_identical"] for b in br.values()),
                }
                if mode == "cag_tf":
                    g = out["diag"]["guidance"]
                    row["a_conditioned_eq_stock_B_bit"] = _bit(np.asarray(g["a_conditioned"], np.float64), stock_saved["B_" + tag])  # env actions are float64
                    row["reconstruction_max_abs_err"] = g["reconstruction_max_abs_err"]
                    att = out["diag"]["attention"]["cond"]
                    mass_sum = np.asarray(att["obs_mass"]) + np.asarray(att["lang_mass"]) - np.asarray(att["prefix_mass"])
                    row["mass_identity_max_abs_err"] = float(np.max(np.abs(mass_sum)))
                r["per_call"].append(row)
        # a call without the CRN tuple must be refused
        try:
            server.infer(dict(observations[0]))
            r[f"{mode}_missing_tuple_refused"] = False
        except ValueError:
            r[f"{mode}_missing_tuple_refused"] = True
    pc = r["per_call"]
    r["P5_PASS"] = (
        all(x["actions_bit_identical_to_stock"] and x["diag_noise_sha_eq_crn"] and x["diag_x0_bit_identical"] for x in pc)
        and all(x.get("a_conditioned_eq_stock_B_bit", True) for x in pc)
        and all(x.get("reconstruction_max_abs_err", 0) < 1e-5 for x in pc)
        and r["vanilla_missing_tuple_refused"] and r["cag_tf_missing_tuple_refused"]
    )
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--guidance-scale", type=float, default=1.5)
    p.add_argument("--stock-only", action="store_true")
    p.add_argument("--compare-to", default=None, help="stock_actions.npz of another process to compare bitwise")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    observations = [_load_obs(o[0]) for o in OBS]
    res = {"noise": noise_checks()}
    res["stock"], saved = stock_checks(args, observations)
    np.savez(out / "stock_actions.npz", **saved)
    if args.compare_to:
        other = np.load(args.compare_to)
        same = {k: _bit(saved[k], other[k]) for k in saved}
        res["cross_process"] = {"compared": args.compare_to, "n": len(same), "n_bit_identical": int(sum(same.values()))}
    if not args.stock_only:
        res["production"] = production_checks(args, observations, saved)
    res["PASS"] = (
        res["noise"]["PASS"] and all(res["stock"][f"P{i}_PASS"] for i in (1, 2, 3, 4))
        and (args.stock_only or res["production"]["P5_PASS"])
        and (not args.compare_to or res["cross_process"]["n_bit_identical"] == res["cross_process"]["n"])
    )
    write_manifest(out, config=vars(args), checkpoint=args.checkpoint, condition="S2b-CRN-validation",
                   seed={"crn_stream_tag": crn.STREAM_TAG}, task={"obs": [o[0] for o in OBS]},
                   initial_state_indices=[0], extra={"PASS": res["PASS"]})
    (out / "crn_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: (v if k == "PASS" else {kk: vv for kk, vv in v.items() if "PASS" in kk}) for k, v in res.items() if k != "cross_process"}, indent=2))
    print("cross_process:", res.get("cross_process"))
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
