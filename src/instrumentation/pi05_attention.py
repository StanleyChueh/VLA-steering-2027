"""Action-expert attention capture for openpi pi0/pi0.5 (JAX), without modifying external/ source.

Mechanism
---------
`openpi.models.gemma.Block` instantiates the module-global `Attention` at trace time, and `gemma.Module` runs the
blocks under `nn.scan`, which stacks each block's returned `kv_cache` along a leading layer axis. We install
`CapturingAttention` as `gemma.Attention`:

`CapturingAttention` is a verbatim copy of `gemma.Attention.__call__` whose only change is the return statement
(exact diff: patches/gemma_attention_capture.diff):
  * capture OFF (default, every control-path trace): returns `(out, (k, v))` exactly like the original, so the
    traced computation is unchanged.
  * capture ON and `kv_cache is not None` (i.e. the action-expert suffix pass during denoising): also returns the
    softmax `probs` tensor that is multiplied into V, as a third element of the kv-cache tuple, so `nn.scan`
    yields probs with shape [L, B, K, G, T, S].

The action that is executed always comes from the untouched jitted `model.sample_actions`. Attention is read from a
separate *diagnostic* re-run (`sample_actions_with_attention`) that mirrors `Pi0.sample_actions`, keeping its
`lax.while_loop`, with each Euler step's probs written into an extra loop-carry buffer. Its final x_0 is compared
against the executed action (identity test, scripts/audit/attention_identity_test.py): bit-identical on the S0C
test observation. (A Python-unrolled variant differs by ~1 bf16 ULP because XLA fuses it differently.)

Token layout for pi05_libero (prefix S_p = 968, suffix T = action_horizon = 10):
  [0,256) base_0_rgb | [256,512) left_wrist_0_rgb | [512,768) right_wrist_0_rgb (image_mask False for LIBERO)
  | [768,968) language (valid where tokenized_prompt_mask) | [968,978) action tokens (keys of the suffix).
"""

from collections.abc import Sequence
import contextlib

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import gemma as _gemma
from openpi.models import lora
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.shared import array_typing as at

_ORIGINAL_ATTENTION = _gemma.Attention
_CAPTURE = {"on": False}


class CapturingAttention(nn.Module):
    """Verbatim copy of openpi-cf gemma.Attention (gemma.py L158-250) + one optional extra output.

    Same dataclass field, same submodule names -> same parameter tree. The only change (see
    patches/gemma_attention_capture.diff) is the final return: when capture is ON and a KV cache is supplied,
    `probs` (the exact tensor multiplied into V) is appended to the returned cache tuple.
    """

    configs: Sequence[_gemma.Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_gemma._name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_gemma._name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_gemma._name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _gemma._apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _gemma._apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_gemma._name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        # --- instrumentation (the only deviation from the original) ---
        if _CAPTURE["on"] and kv_cache is not None:
            return out, (k, v, probs)
        return out, (k, v)


def install():
    """Replace gemma.Attention by CapturingAttention (capture OFF). Idempotent."""
    _gemma.Attention = CapturingAttention


def uninstall():
    _gemma.Attention = _ORIGINAL_ATTENTION


@contextlib.contextmanager
def capture():
    """Trace-time switch. Only affects functions *traced* inside this context."""
    _CAPTURE["on"] = True
    try:
        yield
    finally:
        _CAPTURE["on"] = False


def sample_actions_with_attention(model, rng, observation, *, num_steps=10, noise=None):
    """Mirror of openpi Pi0.sample_actions (pi0.py L224-288), keeping its `lax.while_loop` structure; the loop
    carry is extended with a step counter and a buffer that receives each step's attention probs.

    Returns (x_0, probs, prefix_mask) with probs: [num_steps, L, B, K, G, T_suffix, S_prefix + T_suffix] (bf16).
    Must be traced inside `capture()` (see `jit_diagnostic`).
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(
        observation, mask_language_tokens=model.mask_language_tokens, mask_vision_tokens=model.mask_vision_tokens
    )
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step_with_probs(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), kv_out = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        return x_t + dt * v_t, time + dt, kv_out[2]

    probs_shape = jax.eval_shape(step_with_probs, noise, 1.0)[2]
    buf0 = jnp.zeros((num_steps, *probs_shape.shape), probs_shape.dtype)

    def step(carry):
        x_t, time, i, buf = carry
        x_next, time_next, probs = step_with_probs(x_t, time)
        return x_next, time_next, i + 1, jax.lax.dynamic_update_index_in_dim(buf, probs, i, axis=0)

    def cond(carry):
        x_t, time, _, _ = carry
        return time >= -dt / 2

    x_0, _, _, probs = jax.lax.while_loop(cond, step, (noise, 1.0, 0, buf0))
    return x_0, probs, prefix_mask


def sample_actions_with_attention_unrolled(model, rng, observation, *, num_steps=10, noise=None):
    """Same as `sample_actions_with_attention` but with the Euler loop unrolled in Python (kept for comparison)."""
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(
        observation, mask_language_tokens=model.mask_language_tokens, mask_vision_tokens=model.mask_vision_tokens
    )
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    x_t, time = noise, 1.0
    all_probs = []
    for _ in range(num_steps):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_s = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_s, suffix_attn_mask], axis=-1)
        positions_s = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), kv_out = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions_s,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        all_probs.append(kv_out[2])
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        x_t, time = x_t + dt * v_t, time + dt
    return x_t, jnp.stack(all_probs), prefix_mask


def jit_diagnostic(model):
    """jax.jit the diagnostic sampler with the module state frozen (same pattern as openpi nnx_utils.module_jit)."""
    import flax.nnx as nnx  # noqa: PLC0415

    graphdef, state = nnx.split(model)

    def fun(state, rng, observation, noise):
        m = nnx.merge(graphdef, state)
        return sample_actions_with_attention(m, rng, observation, noise=noise)

    jitted = jax.jit(fun)

    def call(rng, observation, noise):
        # gemma.Module.__call__ declares a 2-tuple kv-cache return; the diagnostic trace returns a 3-tuple, so the
        # runtime jaxtyping check is disabled for this trace only (openpi's own switch).
        with capture(), at.disable_typechecking():
            return jitted(state, rng, observation, noise)

    return call


# ---------------------------------------------------------------------------------------------------------------
# Entropy summaries
# ---------------------------------------------------------------------------------------------------------------
IMAGE_TOKENS = 256


def token_layout(prefix_mask, num_images=3, image_tokens=IMAGE_TOKENS):
    """Index sets for a single example's prefix (prefix_mask: [S_p] bool)."""
    prefix_mask = np.asarray(prefix_mask, dtype=bool)
    s_p = prefix_mask.shape[0]
    img_end = num_images * image_tokens
    idx = np.arange(s_p)
    return {
        "image_valid": idx[(idx < img_end) & prefix_mask],
        "language_valid": idx[(idx >= img_end) & prefix_mask],
        "vlm_all": idx,  # KWTS N: all prefix columns (968), including masked ones (which receive 0 weight)
        "s_prefix": s_p,
    }


def normalized_entropy(p, axis=-1, eps=0.0):
    """H(p)/log(N) with p renormalised along `axis`; N = size of that axis."""
    p = np.asarray(p, dtype=np.float64)
    p = p / np.clip(p.sum(axis=axis, keepdims=True), 1e-300, None)
    plogp = np.where(p > eps, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return -plogp.sum(axis=axis) / np.log(p.shape[axis])


def summarize(probs, prefix_mask):
    """probs: [steps, L, K, G, T, S] for one example (batch dim removed). Returns a dict of numpy arrays.

    * H_obs[step, layer, head, k]   : user definition - action->observation (valid image tokens) normalised entropy.
    * H_vlm[step, layer, head, k]   : action->all valid VLM tokens (images + language), normalised.
    * kwts_E[step, k], kwts_Enorm   : "Knowing When to Stop" Eq. 1-2: head-average, sum over layers, normalise over
                                      the N = S_prefix VLM columns, natural-log entropy; Enorm = E / ln N.
    * obs_mass[step, layer, head, k]: fraction of the query's attention mass that lands on image tokens.
    """
    probs = np.asarray(probs, dtype=np.float32)
    steps, L, K, G, T, S = probs.shape
    probs = probs.reshape(steps, L, K * G, T, S)
    lay = token_layout(prefix_mask)
    img, vlm_valid = lay["image_valid"], np.concatenate([lay["image_valid"], lay["language_valid"]])
    a_img = probs[..., img]
    a_vlm = probs[..., : lay["s_prefix"]]
    out = {
        "H_obs": normalized_entropy(a_img).astype(np.float32),
        "H_vlm": normalized_entropy(probs[..., vlm_valid]).astype(np.float32),
        "obs_mass": a_img.sum(-1).astype(np.float32),
        "N_obs": len(img),
        "N_vlm_valid": len(vlm_valid),
        "N_kwts": lay["s_prefix"],
    }
    w = a_vlm.mean(axis=2).sum(axis=1)  # [steps, T, S_p]: head-average, then sum over layers
    w = w / w.sum(-1, keepdims=True)
    e = -(np.where(w > 0, w * np.log(np.where(w > 0, w, 1.0)), 0.0)).sum(-1)
    out["kwts_E"] = e.astype(np.float32)
    out["kwts_Enorm"] = (e / np.log(lay["s_prefix"])).astype(np.float32)
    return out
