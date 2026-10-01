# S3 feasibility — exposing π0.5 action-expert → observation attention

Label: **FEASIBILITY AUDIT.** This is not S3 itself: no reliability statistics and no adaptive chunking yet.
Date: 2026-10-01.

## Verdict: GO

Action-expert attention can be read out during π0.5 inference with the produced action **bit-identical** to the
uninstrumented policy, at no measurable latency cost, using only code in our repository.

## Where the attention lives (openpi as bundled in LIBERO-CF)

| Element | Location / value |
|---|---|
| Sampler | `openpi-cf/src/openpi/models/pi0.py:224-288` `Pi0.sample_actions`: prefix pass fills the KV cache, then `lax.while_loop` over **10 Euler steps** (t: 1 → 0, dt = −0.1) |
| Transformer | `gemma.py:340-411` `Module`: `nn.scan` over **18** `Block`s (`nn.remat`) with two expert weight sets (PaliGemma 2B width 2048; action expert gemma_300m width 1024, adaRMS time conditioning in π0.5) |
| Attention | `gemma.py:158-250` `Attention`: shared self-attention across experts, **8 query heads / 1 KV head**, head_dim 256, RoPE, explicit `einsum` → `jnp.where(mask, logits, −2.38e38)` → `softmax` → `einsum` with V |
| Kernel | **no fused/flash kernel**: probs are an ordinary XLA intermediate (fused away, never returned) |
| Action-expert queries | suffix pass: queries = 10 action tokens; keys = 968 cached prefix + 10 suffix |
| Prefix key layout | `[0,256)` base RGB · `[256,512)` left-wrist RGB · `[512,768)` right-wrist (zeros, `image_mask=False` → never attended) · `[768,968)` language (BOS + prompt; padding masked; 13 valid for task 0) |
| Suffix key layout | `[968,978)` action tokens (bidirectional inside the chunk) |
| Tensor per layer | `[B, K=1, G=8, T=10, S=978]` |
| Tensor per policy query | `[denoise_step=10, layer=18, B, 1, head=8, action_k=10, key=978]`, bf16 = **28.2 MB** |
| Horizon | action_horizon Hp = 10; LIBERO-CF executes `replan_steps = 5` |

## Instrumentation (ours)

`src/instrumentation/pi05_attention.py`. The exact diff against upstream `Attention` is in
`patches/gemma_attention_capture.diff`; it touches only the module-qualified helper names and the return statement.

1. `install()` replaces `openpi.models.gemma.Attention` with `CapturingAttention`, a verbatim copy whose return is
   `(out, (k, v, probs))` iff a trace-time flag is on **and** a KV cache is supplied (action-expert passes only).
   Otherwise it returns `(out, (k, v))`, exactly as upstream.
2. `nn.scan` stacks the extra cache element along the layer axis.
3. `jit_diagnostic(model)` jits a mirror of `sample_actions` that keeps `lax.while_loop` and writes each step's probs
   into a loop-carry buffer. openpi's runtime jaxtyping check (the declared 2-tuple cache) is disabled for this
   trace only.
4. `summarize(probs, prefix_mask)` returns, without averaging over any dimension prematurely:
   * `H_obs[step, layer, head, k]`: user definition. Attention renormalised over the **512 valid image tokens**,
     `H/ln N`.
   * `H_vlm[step, layer, head, k]`: over all valid VLM tokens (images + valid language).
   * `obs_mass[step, layer, head, k]`: fraction of each query's attention mass that lands on image tokens.
   * `kwts_E[step, k]`, `kwts_Enorm`: KWTS Eq. 1–2 (head mean, sum over layers, normalise over N = 968 prefix
     columns, natural-log entropy; `Enorm = E/ln 968`).

## Identity test

`scripts/audit/attention_identity_test.py` on a real LIBERO-CF observation (CF-Spatial task 0, init 0, after the
10 wait steps; `results/S0C/obs_cf_spatial_t0_i0.npz`). Fixed noise `N(0,I)` from key 0, same checkpoint.

| Comparison | max\|A−B\| | bit-identical |
|---|---|---|
| OFF vs OFF (repeat) | 0 | yes |
| OFF (stock) vs patched class, capture OFF, fresh trace | 0 | yes |
| `Policy.infer` stock vs patched (same RNG key, noise drawn internally) | 0 | yes |
| **OFF vs ON (diagnostic capture, `while_loop`)** | **0** | **yes** |
| OFF vs ON with the Euler loop unrolled in Python (rejected design) | 1.95e-3 | no (XLA fusion differs, ~1 bf16 ULP) |

Sanity checks: per-query attention rows sum to 1 within ±0.003 (bf16-rounded probs); masked columns (right wrist,
language padding) carry exactly 0 mass.

## Cost

| | value |
|---|---|
| Latency, stock sampler (warm) | 70–73 ms |
| Latency, diagnostic sampler (warm, returns probs) | 68–70 ms |
| One-time compile of diagnostic | 4.4 s |
| Device memory for probs | 28.2 MB per query (bf16) |
| Host storage, full raw probs | ~28 MB/query → ~0.6–1.2 GB/episode at 20–44 queries. Too large to keep for every episode. |
| Host storage, `H_obs`+`H_vlm`+`obs_mass` (fp16) | 10×18×8×10×3×2 B ≈ 86 KB/query → ~4 MB/episode |

Recommended logging policy for S3: per-query entropy/mass tensors for every query; raw final-denoise-step probs
(`[18,8,10,978]` fp16 = 2.8 MB) for a preregistered subset (e.g. the first query of every episode and every 5th
query thereafter).

Because the diagnostic sampler returns the bit-identical action, S3 can use it **as the control path** (one pass;
no duplicate computation), with the identity test re-run as a unit test on every S3 launch.

## Definitional issue to resolve before S3/S4 (paper vs pipeline)

KWTS (arXiv 2609.00908, §III-B/C) normalises over N = 968 VLM tokens and triggers truncation when the smoothed
entropy stays above `η ln N`, with η = 0.95 (π0.5 LIBERO: k = 3, τ = 0.01). In the public `pi05_libero` pipeline,
at most ~525 prefix columns are unmasked (512 image + ~13 language), so `E ≤ ln 525 = 0.91 · ln 968 < 0.95 · ln 968`.
**As written, the KWTS rule can never fire on this checkpoint**; it would reduce to Fixed-10. The S0C observation gives
`kwts_Enorm` = 0.44 → 0.50 across k = 1..10 at the final denoising step. This is either an unstated difference in
masking/N/aggregation in the authors' implementation (none is public) or a paper error. S3 will report both the
paper definition and the user-specified image-only normalised entropy, and will **not** adjust η.

## Preliminary observation (single observation, not a result)

At the final denoising step on the S0C observation, `kwts_Enorm` increases monotonically with horizon index k
(0.438 → 0.501), qualitatively matching the paper's "rises with k". Image-only per-head entropy is flat (0.67–0.69),
and the image share of attention mass rises slightly with k (0.257 → 0.282). One observation cannot test anything.
These numbers only confirm the pipeline produces sensible, non-degenerate values.
