"""Serve vanilla pi0.5 (condition B) or pi0.5 + CAG-TF (condition S) for LIBERO-CF evaluation.

Runs in envs/openpi-cf. The policy objects are built by LIBERO-CF's own, unmodified code:
  - B : openpi `create_trained_policy(...)`                       == openpi-cf/scripts/serve_policy.py
  - S : `CounterfactualActionGuidedPolicy(base, guidance_scale=w)` == openpi-cf/scripts/serve_policy_TF.py
This wrapper only adds provenance to each response under the key "server_info": the policy RNG key *before*
the call (which fully determines the sampled flow-matching noise), the call index, latency, and peak GPU
memory. It does not touch the RNG stream, the noise, or the returned actions.

With --diagnostics (S2 pilot) the policy is wrapped by src/instrumentation/pi05_diagnostics.DiagnosticPolicy
instead, which additionally returns, under "diag": per-branch noise fingerprints, CAG guidance magnitude
(a_cond, a_uncond, translation/rotation/gripper/normalised norms) and action-expert attention entropies, with the
returned actions unchanged (scripts/audit/diagnostics_identity_test.py).

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 envs/openpi-cf/bin/python \
      src/wrappers/serve_cf_policy.py --mode cag_tf --guidance-scale 1.5 --port 8000
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import pathlib
import time

import jax
import numpy as np
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SERVE_TF_PATH = REPO_ROOT / "external/libero-cf/openpi-cf/scripts/serve_policy_TF.py"


def _load_cag_class():
    spec = importlib.util.spec_from_file_location("libero_cf_serve_policy_TF", SERVE_TF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CounterfactualActionGuidedPolicy


def _key_fingerprint(key) -> str:
    data = np.asarray(jax.random.key_data(key)) if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key) else np.asarray(key)
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def _peak_gpu_bytes() -> int | None:
    try:
        return int(jax.devices()[0].memory_stats().get("peak_bytes_in_use"))
    except Exception:  # noqa: BLE001
        return None


class ProvenancePolicy:
    """Pass-through wrapper: identical infer() result plus a `server_info` dict."""

    def __init__(self, policy, base_policy, mode, guidance_scale):
        self._policy = policy
        self._base = base_policy
        self._mode = mode
        self._guidance_scale = guidance_scale
        self._calls = 0
        self.metadata = dict(policy.metadata)
        self.metadata.update({"wrapper_mode": mode, "wrapper_guidance_scale": guidance_scale})

    def infer(self, obs, **kwargs):
        key_before = _key_fingerprint(self._base._rng)
        start = time.monotonic()
        out = self._policy.infer(obs, **kwargs)
        wall_ms = (time.monotonic() - start) * 1000
        out["server_info"] = {
            "call_index": self._calls,
            "rng_key_before": key_before,
            "mode": self._mode,
            "guidance_scale": self._guidance_scale,
            "wall_ms": wall_ms,
            "peak_gpu_bytes": _peak_gpu_bytes(),
        }
        self._calls += 1
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "cag_tf"], required=True)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--diagnostics", action="store_true", help="guidance + attention logging (S2)")
    args = parser.parse_args()
    if args.mode == "cag_tf" and args.guidance_scale is None:
        parser.error("--guidance-scale is required for cag_tf (paper: 1.5; LIBERO-CF README: 2.0)")

    if args.diagnostics:
        import sys  # noqa: PLC0415

        sys.path.insert(0, str(REPO_ROOT))
        from src.instrumentation import pi05_attention  # noqa: PLC0415
        from src.instrumentation.pi05_diagnostics import DiagnosticPolicy  # noqa: PLC0415

        pi05_attention.install()  # before the policy is first traced; capture OFF on the control path
    train_config = _config.get_config(args.config)
    base = _policy_config.create_trained_policy(train_config, args.checkpoint)
    if args.mode == "vanilla":
        policy = base
    else:
        policy = _load_cag_class()(base, guidance_scale=args.guidance_scale, uncond_empty_prompt=False)
    if args.diagnostics:
        wrapped = DiagnosticPolicy(policy, base, mode=args.mode, guidance_scale=args.guidance_scale, attention=True)
    else:
        wrapped = ProvenancePolicy(policy, base, args.mode, args.guidance_scale)
    logging.info("Serving %s", json.dumps(wrapped.metadata, default=str))

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="0.0.0.0", port=args.port, metadata=wrapped.metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
