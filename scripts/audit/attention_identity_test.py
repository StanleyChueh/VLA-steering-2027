"""S0C identity test: does exposing pi0.5 action-expert attention change the produced action?

Runs in envs/openpi-cf on one GPU. For one real LIBERO-CF observation and a fixed noise tensor:
  A  : stock openpi policy, stock gemma.Attention, jitted sample_actions            (instrumentation never loaded)
  A2 : same as A, second call                                                     (run-to-run determinism)
  B  : after install(): fresh jit of the same sample_actions, capture OFF          (control path, patched class)
  C  : diagnostic sampler, capture ON, Euler loop unrolled -> x_0 and attention probs
  P  : end-to-end Policy.infer(obs) twice with the policy RNG reset to the same key: patch installed vs not.
Requires max|A-B| == 0 and max|A-C| == 0 (bit-identical), and reports probs shape/memory/sanity.

  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 envs/openpi-cf/bin/python \
      scripts/audit/attention_identity_test.py --obs results/S0C/obs_cf_spatial_t0_i0.npz --out results/S0C
"""

import argparse
import json
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.shared import nnx_utils  # noqa: E402
from openpi.training import config as _config  # noqa: E402

from src.instrumentation import pi05_attention as attn  # noqa: E402
from src.wrappers.manifest import write_manifest  # noqa: E402


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def bit_identical(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", required=True)
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = pathlib.Path(args.out)

    d = np.load(args.obs)
    obs = {
        "observation/image": d["image"],
        "observation/wrist_image": d["wrist_image"],
        "observation/state": d["state"],
        "prompt": str(d["prompt"]),
    }
    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), args.checkpoint)
    model = policy._model
    inputs = policy._input_transform(jax.tree.map(lambda x: x, obs))
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(inputs)
    noise = jax.random.normal(jax.random.key(args.noise_seed), (1, model.action_horizon, model.action_dim))
    rng = jax.random.key(123)
    res = {}

    # A / A2: stock path
    a = np.asarray(policy._sample_actions(rng, observation, noise=noise))
    a2 = np.asarray(policy._sample_actions(rng, observation, noise=noise))
    policy._rng = jax.random.key(7)
    p_stock = policy.infer(obs)["actions"]

    # B: patched class, capture OFF, freshly traced
    attn.install()
    sample_b = nnx_utils.module_jit(model.sample_actions)
    b = np.asarray(sample_b(rng, observation, noise=noise))
    policy2 = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), args.checkpoint)
    policy2._rng = jax.random.key(7)
    p_patched = policy2.infer(obs)["actions"]

    # C: diagnostic sampler with capture ON
    diag = attn.jit_diagnostic(model)
    t0 = time.monotonic()
    c, probs, prefix_mask = diag(rng, observation, noise)
    c = np.asarray(c)
    probs.block_until_ready()
    first_ms = (time.monotonic() - t0) * 1000
    t0 = time.monotonic()
    c_again, probs_again, _ = diag(rng, observation, noise)
    probs_again.block_until_ready()
    diag_ms = (time.monotonic() - t0) * 1000
    t0 = time.monotonic()
    np.asarray(policy._sample_actions(rng, observation, noise=noise))
    ctrl_ms = (time.monotonic() - t0) * 1000

    probs_np = np.asarray(probs[:, :, 0].astype(jnp.float32))  # [steps, L, K, G, T, S]
    pm = np.asarray(prefix_mask[0])
    row_sums = probs_np.sum(-1)
    lay = attn.token_layout(pm)
    masked_cols = np.setdiff1d(np.arange(lay["s_prefix"]), np.concatenate([lay["image_valid"], lay["language_valid"]]))

    res["shapes"] = {"actions": list(a.shape), "probs": list(probs.shape), "probs_dtype": str(probs.dtype)}
    res["identity"] = {
        "A_vs_A2_bit_identical": bit_identical(a, a2),
        "A_vs_B_max_abs": max_abs(a, b),
        "A_vs_B_bit_identical": bit_identical(a, b),
        "A_vs_C_max_abs": max_abs(a, c),
        "A_vs_C_bit_identical": bit_identical(a, c),
        "C_vs_C2_bit_identical": bit_identical(c, np.asarray(c_again)),
        "policy_infer_stock_vs_patched_max_abs": max_abs(p_stock, p_patched),
        "policy_infer_stock_vs_patched_bit_identical": bit_identical(p_stock, p_patched),
    }
    res["probs_sanity"] = {
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "mass_on_masked_prefix_cols_max": float(probs_np[..., masked_cols].sum(-1).max()) if len(masked_cols) else 0.0,
        "n_image_valid": int(len(lay["image_valid"])),
        "n_language_valid": int(len(lay["language_valid"])),
        "n_prefix": int(lay["s_prefix"]),
    }
    res["cost"] = {
        "probs_bytes_per_query_bf16": int(np.prod(probs.shape) * 2),
        "control_ms": ctrl_ms,
        "diagnostic_ms_warm": diag_ms,
        "diagnostic_ms_first_call_incl_compile": first_ms,
        "peak_gpu_bytes": int(jax.devices()[0].memory_stats().get("peak_bytes_in_use", -1)),
    }
    s = attn.summarize(probs_np, pm)
    res["entropy_preview"] = {
        "H_obs_final_step_mean_by_k": s["H_obs"][-1].mean(axis=(0, 1)).round(4).tolist(),
        "kwts_Enorm_final_step_by_k": s["kwts_Enorm"][-1].round(4).tolist(),
        "obs_mass_final_step_mean_by_k": s["obs_mass"][-1].mean(axis=(0, 1)).round(4).tolist(),
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "attention_identity_probs_final_step.npz", probs_final=probs_np[-1].astype(np.float16), prefix_mask=pm)
    write_manifest(
        out, config=vars(args), checkpoint=args.checkpoint, condition="S0C-identity",
        seed={"noise_seed": args.noise_seed, "rng": 123, "policy_rng": 7}, task={"obs": args.obs},
        initial_state_indices=[0], extra=res,
    )
    (out / "attention_identity_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
