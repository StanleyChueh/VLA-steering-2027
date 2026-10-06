"""S2d oracle counterfactual-reference guidance for pi0.5 (ORACLE DIAGNOSTIC, not a method). Runs in envs/openpi-cf.

Mirrors LIBERO-CF's CounterfactualActionGuidedPolicy.infer (serve_policy_TF.py) line for line for the faithful and
language-masked passes (same input transform, same token masking, same rng split, ONE shared noise tensor), and adds
one pass with a reference instruction l_r supplied per request (obs["s2d_ref_prompt"]: the oracle source/training
instruction for SOURCE-*, the frozen non-source alternative for OTHER). With A_f, A_u, A_r the pre-guidance chunks
(model quantile space, unbatched [10, 32]):

  mode "ref_norm" (SOURCE-NORM / OTHER):
      s_mask = || 0.5 (A_f - A_u)[:, :7] ||_F                    (MASK's intervention norm at this call)
      d      = R / (||R||_F + 1e-6),  R = (A_f - A_r)[:, :7]
      A      = A_f;  A[:, :7] += s_mask * d                       (dims 7..31 are padding, dropped by LiberoOutputs)
  mode "ref_raw" (SOURCE-RAW):
      A      = A_f + 0.5 (A_f - A_r)                              (CAG's formula with l_r in place of the masked branch)

So ||A - A_f|| over the 7 action dims equals MASK's ||0.5 (A_f - A_u)|| at every call in ref_norm mode: only the
reference DIRECTION differs from CAG-TF (omega = 1.5). The action is then passed through the policy's own output
transform. Per-call diagnostics are returned under "diag" (noise fingerprints per pass, chunks, norms, cosines).
"""

import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model

EPS = 1e-6


def _sha16(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()[:16]


class ReferenceGuidedPolicy:
    def __init__(self, base_policy, *, mode: str, omega: float = 1.5):
        assert mode in ("ref_norm", "ref_raw"), mode
        self._base = base_policy
        self._mode = mode
        self._omega = omega
        self.metadata = dict(base_policy.metadata)
        self.metadata.update({"s2d_mode": mode, "s2d_omega": omega})

    # identical to CounterfactualActionGuidedPolicy helpers
    def _prepare_inputs(self, obs, prompt=None):
        inputs = jax.tree.map(lambda x: x, obs)
        if prompt is not None:
            inputs = dict(inputs)
            inputs["prompt"] = prompt
        return self._base._input_transform(inputs)

    @staticmethod
    def _mask_language_tokens(inputs):
        masked = dict(inputs)
        if masked.get("tokenized_prompt") is not None:
            masked["tokenized_prompt"] = jnp.zeros_like(masked["tokenized_prompt"])
        if masked.get("tokenized_prompt_mask") is not None:
            masked["tokenized_prompt_mask"] = jnp.zeros_like(masked["tokenized_prompt_mask"])
        return masked

    @staticmethod
    def _convert(inputs):
        return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

    def infer(self, obs, *, noise=None):
        obs = dict(obs)
        obs.pop("diag_meta", None)
        ref_prompt = obs.pop("s2d_ref_prompt", None)
        if not ref_prompt:
            raise ValueError("S2d: request has no s2d_ref_prompt")
        if noise is None:
            raise ValueError("S2d: noise must be supplied (CRN)")
        base = self._base
        model = base._model
        inputs = self._prepare_inputs(obs)
        masked = self._mask_language_tokens(inputs)
        ref_inputs = self._prepare_inputs(obs, prompt=ref_prompt)
        sample_kwargs = dict(base._sample_kwargs)
        inputs_b, masked_b, ref_b = self._convert(inputs), self._convert(masked), self._convert(ref_inputs)
        base._rng, noise_rng, cond_rng, uncond_rng = jax.random.split(base._rng, 4)
        noise_t = jnp.asarray(noise)
        if noise_t.ndim == 2:
            noise_t = noise_t[None, ...]
        assert noise_t.shape == (1, model.action_horizon, model.action_dim)
        sample_kwargs["noise"] = noise_t
        t0 = time.monotonic()
        a_f = np.asarray(base._sample_actions(cond_rng, _model.Observation.from_dict(inputs_b), **sample_kwargs)[0])
        a_r = np.asarray(base._sample_actions(cond_rng, _model.Observation.from_dict(ref_b), **sample_kwargs)[0])
        a_u = None
        if self._mode == "ref_norm":
            a_u = np.asarray(base._sample_actions(uncond_rng, _model.Observation.from_dict(masked_b), **sample_kwargs)[0])
        infer_ms = (time.monotonic() - t0) * 1000

        r = (a_f - a_r)[:, :7]
        r_norm = float(np.linalg.norm(r))
        out_a = a_f.copy()
        diag = {"r_ref_norm": r_norm, "noise_sha": _sha16(noise_t)}
        if self._mode == "ref_norm":
            r_cag = (a_f - a_u)[:, :7]
            s_mask = float(np.linalg.norm((self._omega - 1.0) * r_cag))
            out_a[:, :7] = a_f[:, :7] + s_mask * (r / (r_norm + EPS))
            diag.update({"s_mask": s_mask,
                         "cos_rcag_rref": float(r_cag.ravel() @ r.ravel() / max(np.linalg.norm(r_cag) * r_norm, 1e-12)),
                         "a_unconditioned_model": a_u[:, :7].tolist()})
        else:
            out_a = a_f + (self._omega - 1.0) * (a_f - a_r)
        diag["intervention_norm"] = float(np.linalg.norm((out_a - a_f)[:, :7]))
        state = np.asarray(inputs_b["state"][0])
        outputs = base._output_transform({"state": state, "actions": out_a})
        zeros = np.zeros(8, np.float32)
        diag["a_conditioned"] = np.asarray(base._output_transform({"state": zeros, "actions": a_f.astype(np.float32).copy()})["actions"]).tolist()
        diag["a_reference"] = np.asarray(base._output_transform({"state": zeros, "actions": a_r.astype(np.float32).copy()})["actions"]).tolist()
        diag["a_conditioned_model"] = a_f[:, :7].tolist()
        diag["a_reference_model"] = a_r[:, :7].tolist()
        diag["ref_prompt"] = ref_prompt
        branches = {"cond": {"noise_sha": diag["noise_sha"]}, "ref": {"noise_sha": diag["noise_sha"]}}
        if a_u is not None:
            branches["uncond"] = {"noise_sha": diag["noise_sha"]}
        outputs["policy_timing"] = {"infer_ms": infer_ms}
        outputs["diag"] = {"branches": branches, "s2d": diag}
        return outputs
