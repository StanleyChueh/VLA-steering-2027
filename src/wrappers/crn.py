"""Common-random-number (CRN) control of pi0.5 policy noise for S2b (runs in envs/openpi-cf).

The only stochastic quantity in `Pi0.sample_actions` (openpi-cf pi0.py L224-288) is the flow-matching noise tensor:
when `noise` is None it is drawn as `jax.random.normal(rng, (b, action_horizon, action_dim))`, and the Euler ODE
that follows is deterministic. Both public entry points accept that tensor directly:
  * openpi `Policy.infer(obs, noise=...)`                              (condition B)
  * LIBERO-CF `CounterfactualActionGuidedPolicy.infer(obs, noise=...)` (condition S), which feeds the SAME tensor
    to its conditioned and language-masked passes (serve_policy_TF.py: `sample_kwargs["noise"] = noise_tensor`
    for both `_sample_actions` calls).
CRN therefore needs no change to policy code: the noise for one policy call is a pure function of

    (master_seed, task_id, initial_state_id, policy_call_index)

and is passed through the public `noise=` argument. The marginal distribution is the native one (i.i.d. standard
normal float32 of shape (1, 10, 32), drawn by the same `jax.random.normal` with the default threefry key), and the
public CAG relationship (one shared tensor for both branches) is preserved because CAG receives it unchanged.
Episode lengths cannot shift later episodes: no state is carried between calls.
"""

import hashlib

import jax
import numpy as np

# Domain-separation tag so S2b streams can never coincide with openpi's native jax.random.key(0) stream.
STREAM_TAG = 0x53324220  # "S2B "
NOISE_SHAPE = (1, 10, 32)  # (batch, action_horizon, action_dim) of pi05_libero
KEY_FIELDS = ("master_seed", "task_id", "init_id", "call_index")


def crn_key(master_seed: int, task_id: int, init_id: int, call_index: int):
    key = jax.random.key(STREAM_TAG)
    for v in (master_seed, task_id, init_id, call_index):
        if not (0 <= int(v) < 2**31):
            raise ValueError(f"CRN key field out of range: {v}")
        key = jax.random.fold_in(key, int(v))
    return key


def crn_noise(master_seed: int, task_id: int, init_id: int, call_index: int, shape=NOISE_SHAPE) -> np.ndarray:
    """Standard-normal float32 noise, identical for identical tuples, independent across tuples."""
    return np.asarray(jax.random.normal(crn_key(master_seed, task_id, init_id, call_index), shape), dtype=np.float32)


def fingerprint(x) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()[:16]


class CRNPolicy:
    """Outermost server wrapper: pops `obs["crn"]`, derives the noise, calls `inner.infer(obs, noise=noise)`.

    Strict: every call must carry a CRN tuple whose master_seed matches the server's, so a run can never fall back
    to the native RNG stream silently. Adds `out["crn"] = {tuple..., "noise_sha"}` to the response.
    """

    def __init__(self, inner, master_seed: int):
        self._inner = inner
        self._master_seed = int(master_seed)
        self.metadata = dict(inner.metadata)
        self.metadata.update({"crn_master_seed": self._master_seed, "crn_stream_tag": STREAM_TAG,
                              "crn_noise_shape": list(NOISE_SHAPE)})

    def infer(self, obs, **kwargs):
        obs = dict(obs)
        tup = obs.pop("crn", None)
        if tup is None:
            raise ValueError("CRN server: observation has no 'crn' tuple")
        tup = {k: int(tup[k]) for k in KEY_FIELDS}
        if tup["master_seed"] != self._master_seed:
            raise ValueError(f"CRN master_seed mismatch: client {tup['master_seed']} != server {self._master_seed}")
        if "noise" in kwargs:
            raise ValueError("CRN server: caller-supplied noise is not allowed")
        noise = crn_noise(**tup)
        out = self._inner.infer(obs, noise=noise, **kwargs)
        out["crn"] = dict(tup, noise_sha=fingerprint(noise))
        return out
