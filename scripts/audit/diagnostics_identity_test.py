"""Regression test: S2 diagnostics (guidance-magnitude + action-expert attention logging) do not change actions.

Runs in envs/openpi-cf on one GPU. For vanilla pi0.5 and CAG-TF (omega=1.5) and a sequence of real LIBERO-CF
observations (each fed twice, so the policy RNG stream advances over 2N calls):
  A : stock path, exactly as served before S2 (openpi Policy / LIBERO-CF CounterfactualActionGuidedPolicy), traced
      BEFORE pi05_attention.install()  -> instrumentation never loaded;
  B : production S2 path: install(), fresh policy, DiagnosticPolicy(attention=True) with guidance logging, raw probs
      written for the first call.
Same checkpoint, same starting policy RNG key. Requires, per call: actions(A) bit-identical to actions(B); the RNG
key after the sequence identical; every diagnostic re-run x0 bit-identical to its branch output; CAG reconstruction
uncond + w(cond - uncond) == returned actions.

  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 envs/openpi-cf/bin/python \
      scripts/audit/diagnostics_identity_test.py --obs results/S0C/obs_cf_spatial_t0_i0.npz ... --out results/S2_identity
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

from src.instrumentation import pi05_attention as attn  # noqa: E402
from src.wrappers.manifest import write_manifest  # noqa: E402

SERVE_TF_PATH = REPO_ROOT / "external/libero-cf/openpi-cf/scripts/serve_policy_TF.py"


def _cag_class():
    spec = importlib.util.spec_from_file_location("libero_cf_serve_policy_TF", SERVE_TF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CounterfactualActionGuidedPolicy


def _load_obs(path):
    d = np.load(path)
    return {
        "observation/image": d["image"],
        "observation/wrist_image": d["wrist_image"],
        "observation/state": d["state"],
        "prompt": str(d["prompt"]),
    }


def _bit(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", nargs="+", required=True)
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--guidance-scale", type=float, default=1.5)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq = [_load_obs(o) for o in args.obs] * 2
    cfg = _config.get_config("pi05_libero")
    CAG = _cag_class()

    def wrap(base, mode):
        return base if mode == "vanilla" else CAG(base, guidance_scale=args.guidance_scale, uncond_empty_prompt=False)

    # A: stock (instrumentation not installed yet)
    stock = _policy_config.create_trained_policy(cfg, args.checkpoint)
    actions_a, keys_a = {}, {}
    for mode in ("vanilla", "cag_tf"):
        stock._rng = jax.random.key(args.rng_seed)
        pol = wrap(stock, mode)
        actions_a[mode] = [pol.infer(dict(o))["actions"] for o in seq]
        keys_a[mode] = np.asarray(jax.random.key_data(stock._rng))

    # B: production S2 path
    attn.install()
    from src.instrumentation.pi05_diagnostics import DiagnosticPolicy  # noqa: PLC0415

    inst = _policy_config.create_trained_policy(cfg, args.checkpoint)
    original_sample = inst._sample_actions
    res = {"n_calls_per_mode": len(seq), "modes": {}}
    for mode in ("vanilla", "cag_tf"):
        inst._sample_actions = original_sample
        inst._rng = jax.random.key(args.rng_seed)
        dp = DiagnosticPolicy(wrap(inst, mode), inst, mode=mode, guidance_scale=None if mode == "vanilla" else args.guidance_scale)
        outs = []
        for i, o in enumerate(seq):
            o = dict(o)
            o["diag_meta"] = {"save_raw_path": str(out / "raw" / f"raw_{mode}_call{i}")} if i == 0 else {}
            outs.append(dp.infer(o))
        actions_b = [r["actions"] for r in outs]
        per_call = [_bit(a, b) for a, b in zip(actions_a[mode], actions_b)]
        x0 = [br["x0_bit_identical"] for r in outs for br in r["diag"]["branches"].values()]
        m = {
            "actions_bit_identical_per_call": per_call,
            "actions_max_abs_diff": max(float(np.max(np.abs(a - b))) for a, b in zip(actions_a[mode], actions_b)),
            "rng_after_sequence_identical": _bit(keys_a[mode], np.asarray(jax.random.key_data(inst._rng))),
            "diag_x0_bit_identical_all": all(x0),
            "n_branch_checks": len(x0),
            "noise_sha_first_call": {k: v["noise_sha"] for k, v in outs[0]["diag"]["branches"].items()},
            "raw_probs_file": outs[0]["diag"]["branches"]["cond"].get("raw_probs_file"),
        }
        att = outs[0]["diag"]["attention"]["cond"]
        m["attention_shapes"] = {k: list(np.shape(v)) for k, v in att.items()}
        m["obs_valid_key_count"] = int(att["obs_valid_key_count"])
        m["vlm_valid_key_count"] = int(att["vlm_valid_key_count"])
        m["obs_normalized_entropy_final_step_by_k"] = np.asarray(att["obs_normalized_entropy"])[-1].mean(axis=(0, 1)).round(4).tolist()
        if mode == "cag_tf":
            g = [r["diag"]["guidance"] for r in outs]
            m["guidance_reconstruction_max_abs_err"] = max(x["reconstruction_max_abs_err"] for x in g)
            m["guidance_translation_norm_first_call"] = np.round(g[0]["translation_norm"], 5).tolist()
            m["guidance_normalized_combined_norm_first_call"] = np.round(g[0]["normalized_combined_norm"], 4).tolist()
        m["PASS"] = (
            all(per_call)
            and m["rng_after_sequence_identical"]
            and m["diag_x0_bit_identical_all"]
            and m.get("guidance_reconstruction_max_abs_err", 0.0) < 1e-5
        )
        res["modes"][mode] = m
    res["PASS"] = all(m["PASS"] for m in res["modes"].values())
    write_manifest(
        out, config=vars(args), checkpoint=args.checkpoint, condition="S2-diagnostics-identity",
        seed={"policy_rng": args.rng_seed}, task={"obs": args.obs}, initial_state_indices=[0], extra=res,
    )
    (out / "identity_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
