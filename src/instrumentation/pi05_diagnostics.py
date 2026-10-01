"""Per-call diagnostics for pi0.5 (vanilla) and pi0.5 + CAG-TF, without changing the returned action.

`DiagnosticPolicy` wraps the policy object that the server would otherwise serve (openpi `Policy` for B, LIBERO-CF's
unmodified `CounterfactualActionGuidedPolicy` for S). It replaces `base_policy._sample_actions` by a recorder that
calls the ORIGINAL jitted sampler with the exact same arguments and returns its result untouched; the policy's own
`infer()` code (RNG splitting, noise, CAG combination, output transforms) runs unmodified. Side computations:

  * branch outputs: every `_sample_actions` call made inside one `infer()` is recorded. Vanilla makes one call
    ("cond"); CAG-TF makes two, in a fixed order: conditioned, then language-masked ("uncond")
    (serve_policy_TF.py: `cond_actions = ...; uncond_actions = ...`).
  * guidance magnitude (CAG only): a_cond, a_uncond mapped through the policy's own output transform (env action
    space, 10 x 7) and their difference split into translation (dims 0-2), rotation (3-5), gripper (6), plus the L2
    norm of the difference in the model's quantile-normalised action space (dims 0-6). A reconstruction check
    confirms uncond + w * (cond - uncond) == returned actions.
  * action-expert attention: the diagnostic sampler (src/instrumentation/pi05_attention.py) is re-run with the same
    (rng, observation, noise) and must reproduce the branch output bit-for-bit (`x0_bit_identical`, logged per
    call); attention entropies are reduced on the GPU, raw probs are optionally written to disk.
  * policy-noise fingerprint: sha256 of the flow-matching noise tensor actually used.

The returned `actions` are always the ones produced by the unmodified policy code; diagnostics run after the fact on
copies of its inputs. Requires `pi05_attention.install()` before the policy is first traced if attention is wanted
(capture OFF on the control path is bit-identical, see reports/S3_ATTENTION_FEASIBILITY.md).
"""

import hashlib
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.instrumentation import pi05_attention as attn

IMAGE_TOKENS = attn.IMAGE_TOKENS
NUM_IMAGES = 3


def _sha16(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()[:16]


def _bit_identical(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@jax.jit
def _entropy_summary(probs, prefix_mask):
    """probs: [steps, L, B=1, K, G, T, S] (bf16); prefix_mask: [1, S_p]. All reductions in float32.

    For each (step, layer, head, k):
      obs : distribution over the VALID image keys (image tokens with prefix_mask True), renormalised;
      vlm : distribution over all valid prefix keys (images + valid language), renormalised;
    raw entropy H = -sum p ln p, normalised H / ln(valid_key_count) (valid_key_count > 1 here: 512 / ~525).
    obs_mass = un-renormalised attention mass on valid image keys.
    """
    p = probs[:, :, 0].astype(jnp.float32)  # [steps, L, K, G, T, S]
    steps, L, K, G, T, S = p.shape
    p = p.reshape(steps, L, K * G, T, S)
    s_p = prefix_mask.shape[-1]
    valid = prefix_mask[0].astype(bool)
    idx = jnp.arange(s_p)
    m_obs = (idx < NUM_IMAGES * IMAGE_TOKENS) & valid
    m_vlm = valid
    out = {}
    pre = p[..., :s_p]
    for name, m in (("obs", m_obs), ("vlm", m_vlm)):
        q = jnp.where(m, pre, 0.0)
        mass = q.sum(-1)
        q = q / jnp.maximum(mass[..., None], 1e-30)
        h = -jnp.sum(jnp.where(q > 0, q * jnp.log(jnp.where(q > 0, q, 1.0)), 0.0), axis=-1)
        n = m.sum()
        out[f"{name}_raw_entropy"] = h
        out[f"{name}_normalized_entropy"] = jnp.where(n > 1, h / jnp.log(jnp.maximum(n, 2).astype(jnp.float32)), jnp.nan)
        out[f"{name}_valid_key_count"] = n
        if name == "obs":
            out["obs_mass"] = mass
    # "Knowing When to Stop" Eq. 1-2 reference quantity (head mean, layer sum, over all S_p prefix columns).
    w = pre.mean(axis=2).sum(axis=1)  # [steps, T, S_p]
    w = w / w.sum(-1, keepdims=True)
    e = -jnp.sum(jnp.where(w > 0, w * jnp.log(jnp.where(w > 0, w, 1.0)), 0.0), axis=-1)
    out["kwts_E"] = e
    out["kwts_N"] = jnp.asarray(s_p)
    return out


class DiagnosticPolicy:
    """Pass-through policy: identical `actions`, plus `server_info` and `diag` entries in the response.

    The client may add `obs["diag_meta"] = {"save_raw_path": str | None, ...}`; it is removed before the wrapped
    policy sees the observation.
    """

    def __init__(self, policy, base_policy, *, mode, guidance_scale, attention=True):
        self._policy = policy
        self._base = base_policy
        self._mode = mode
        self._guidance_scale = guidance_scale
        self._attention = attention
        self._calls = 0
        self._records = None
        self._stock_sample = base_policy._sample_actions
        base_policy._sample_actions = self._recording_sample
        self._diag = attn.jit_diagnostic(base_policy._model) if attention else None
        self.metadata = dict(policy.metadata)
        self.metadata.update(
            {"wrapper_mode": mode, "wrapper_guidance_scale": guidance_scale, "wrapper_attention": attention}
        )

    # -- recorder ---------------------------------------------------------------------------------------------
    def _recording_sample(self, rng, observation, **kwargs):
        out = self._stock_sample(rng, observation, **kwargs)
        if self._records is not None:
            self._records.append({"rng": rng, "observation": observation, "noise": kwargs.get("noise"), "out": out})
        return out

    # -- per-call diagnostics -----------------------------------------------------------------------------------
    def _branch_diag(self, rec, save_raw_path):
        model = self._base._model
        noise = rec["noise"]
        if noise is None:  # vanilla: noise is drawn inside sample_actions from `rng` (pi0.py L234-235)
            noise_used = jax.random.normal(rec["rng"], (1, model.action_horizon, model.action_dim))
        else:
            noise_used = noise
        d = {"noise_sha": _sha16(noise_used), "rng_sha": _sha16(jax.random.key_data(rec["rng"]))}
        if not self._attention:
            return d, None
        t0 = time.monotonic()
        x0, probs, prefix_mask = self._diag(rec["rng"], rec["observation"], noise)
        summ = jax.device_get(_entropy_summary(probs, prefix_mask))
        d["x0_bit_identical"] = _bit_identical(x0, rec["out"])
        d["x0_max_abs_diff"] = float(np.max(np.abs(np.asarray(x0, np.float64) - np.asarray(rec["out"], np.float64))))
        if save_raw_path is not None:
            raw = np.asarray(probs[:, :, 0]).view(np.uint16)  # exact bf16 bits, [steps, L, K, G, T, S]
            np.save(save_raw_path, raw)
            d["raw_probs_file"] = str(save_raw_path)
            d["raw_probs_layout"] = "uint16 view of bf16 probs [denoise_step, layer, kv_head, q_group, action_k, key]"
        d["diag_ms"] = (time.monotonic() - t0) * 1000
        summ = {k: np.asarray(v) for k, v in summ.items()}
        summ["prefix_mask"] = np.asarray(prefix_mask[0])
        return d, summ

    def _guidance(self, recs, returned_actions):
        state = np.zeros(8, dtype=np.float32)  # LiberoOutputs ignores state; Unnormalize only touches actions
        model_space = {}
        env_space = {}
        for name, r in zip(("cond", "uncond"), recs):
            a = np.asarray(r["out"][0], dtype=np.float32)
            model_space[name] = a
            env_space[name] = np.asarray(self._base._output_transform({"state": state, "actions": a.copy()})["actions"])
        w = self._guidance_scale
        delta = env_space["cond"] - env_space["uncond"]  # [H, 7]
        delta_norm_space = model_space["cond"][:, :7] - model_space["uncond"][:, :7]
        recon = env_space["uncond"] + w * delta
        return {
            "guidance_scale": w,
            "a_conditioned": env_space["cond"].tolist(),
            "a_unconditioned": env_space["uncond"].tolist(),
            "translation_norm": np.linalg.norm(delta[:, 0:3], axis=-1).tolist(),
            "rotation_norm": np.linalg.norm(delta[:, 3:6], axis=-1).tolist(),
            "gripper_abs_diff": np.abs(delta[:, 6]).tolist(),
            "normalized_combined_norm": np.linalg.norm(delta_norm_space, axis=-1).tolist(),
            "reconstruction_max_abs_err": float(np.max(np.abs(recon - np.asarray(returned_actions)))),
        }

    def infer(self, obs, **kwargs):
        obs = dict(obs)
        meta = obs.pop("diag_meta", None) or {}
        self._records = []
        key_before = _sha16(jax.random.key_data(self._base._rng))
        start = time.monotonic()
        try:
            out = self._policy.infer(obs, **kwargs)
        finally:
            recs, self._records = self._records, None
        wall_ms = (time.monotonic() - start) * 1000

        names = ["cond"] if len(recs) == 1 else ["cond", "uncond"]
        assert len(recs) == len(names), f"unexpected number of sampler calls: {len(recs)}"
        branches, summaries = {}, {}
        raw_base = meta.get("save_raw_path")
        for name, rec in zip(names, recs):
            raw_path = None if raw_base is None else pathlib.Path(f"{raw_base}_{name}.npy")
            if raw_path is not None:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
            branches[name], summaries[name] = self._branch_diag(rec, raw_path)
        diag = {"branches": branches}
        if self._mode == "cag_tf":
            diag["guidance"] = self._guidance(recs, out["actions"])
        if self._attention:
            diag["attention"] = summaries
        out["server_info"] = {
            "call_index": self._calls,
            "rng_key_before": key_before,
            "mode": self._mode,
            "guidance_scale": self._guidance_scale,
            "policy_wall_ms": wall_ms,
            "total_wall_ms": (time.monotonic() - start) * 1000,
            "peak_gpu_bytes": _peak_gpu_bytes(),
        }
        out["diag"] = diag
        self._calls += 1
        return out


def _peak_gpu_bytes():
    try:
        return int(jax.devices()[0].memory_stats().get("peak_bytes_in_use"))
    except Exception:  # noqa: BLE001
        return None
