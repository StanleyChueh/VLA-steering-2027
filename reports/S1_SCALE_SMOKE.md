# S1 smoke — OpenVLA greedy vs SCALE on LIBERO-10 (task 0)

Label: **SMOKE TEST (n=2 per condition; no inferential content).** Date: 2026-10-01.

## Setup

| Item | Value |
|---|---|
| Suite / task | `libero_10`, task 0: "put both the alphabet soup and the tomato sauce in the basket" |
| Initial states | 0, 1 (LIBERO default init states, same for both conditions) |
| Checkpoint | `openvla/openvla-7b-finetuned-libero-10` @ `80970322773f81baa2e22fe495d0487b93a05cfa`, bf16, FlashAttention-2 |
| Code | SCALE `b4ad2a69` functions, driven by `scripts/run_scale/run_scale_eval.py` (mirrors `run_libero_eval.py`; logging only) |
| SCALE hyperparameters | from `external/scale/configs/scale.yaml`: T0=1.0, κ=2.0, α=0.8, ε=1e-12, num_logits=256, attn_sensitivity=0.3 |
| Seeds | `set_seed_everywhere(7)` (official default), `env.seed(0)` (official) |
| Max steps / wait | 520 / 10 (official) |
| LIBERO | upstream `8f1084e3` |
| GPU | RTX 4090 #0 |
| Commands | `CUDA_VISIBLE_DEVICES=0 envs/scale/bin/python scripts/run_scale/run_scale_eval.py --task-suite libero_10 --task-ids 0 --episodes 2 --decoding-mode {greedy,scale} --out results/S1_smoke/{greedy,scale}` |
| Raw records | `results/S1_smoke/{greedy,scale}/episodes.jsonl` (per-step u_k, τ_k, γ, latency, actions) + `manifest.json` |

## Results

| Condition | Success | Episode lengths (control steps) | Median latency / step | Peak torch CUDA memory |
|---|---|---|---|---|
| Greedy | **1/2** | 520 (timeout, fail), 254 | 162.6 ms | 14.42 GiB |
| SCALE | **2/2** | 276, 274 | 164.3 ms | 14.41 GiB |

Paper reference (Table 11, 50 trials × 3 seeds): this task is 62.7% greedy → 66.0% SCALE. Two episodes say
nothing about that difference.

## Mechanism checks (SCALE episodes)

| Quantity | ep 0 | ep 1 |
|---|---|---|
| token uncertainty u_k: min / median / max | −5.55 / −5.00 / 17.58 | −5.55 / −4.83 / 17.17 |
| fraction of tokens with u_k > 0 (p_max ≲ 0.80) | 0.30 | 0.29 |
| decoding temperature τ_k: median / p90 / max | 0.0067 / 1.00 / 1.00 | 0.0079 / 1.00 / 1.00 |
| fraction of tokens sampled with τ_k > 0.1 | 0.37 | 0.36 |
| visual temperature γ: min / median / max | 0.518 / 0.983 / 1.976 | 0.554 / 0.986 / 1.915 |
| fraction of steps with \|γ − 1\| > 0.05 | 0.90 | 0.91 |
| γ at steps 1, 2, 3 | 1.0, 1.0, 0.80 | 1.0, 1.0, 0.67 |

Interpretation of the checks (sanity only):

* u_k hits its analytic floor −ln 256 = −5.545 exactly, as expected from the code's q_low/q_high construction
  (S0 A2), so the uncertainty implementation behaves as audited.
* τ is effectively **bimodal**: σ(u) saturates on both sides of u's [−5.5, 22] range, so tokens are either decoded
  near-greedily (τ ≈ 0.004–0.01) or sampled at nearly the full T0 = 1. That is a property of the method as
  implemented, worth keeping in mind when interpreting "adaptive" temperature.
* γ = 1 on the first two control steps, then modulation starts. This matches Alg. 1 timing (S0). Despite the
  undocumented s = 0.3 (S0 A1), γ covers nearly the full (1/κ, κ) = (0.5, 2) range, because Δu varies over tens of
  units.
* SCALE adds ~1.7 ms/step (+1%) over greedy, consistent with the paper's "single pass" claim (Table 9).

## Practical notes

* **TensorFlow / libdevice.** OpenVLA's preprocessing (JPEG round-trip, lanczos resize, center crop) runs in
  TF 2.15 on the GPU. Without CUDA's `libdevice`, every episode died at the first step (`libdevice not found`,
  caught by the loop's `except` and logged as `UnknownError()`). The upstream script would silently record such
  episodes as failures. The README's conda `cuda-compiler` provides libdevice; we install the pip equivalent pinned
  by `tensorflow[and-cuda]==2.15.0` (`nvidia-cuda-nvcc-cu12==12.2.140`). **Any SCALE/OpenVLA reproduction should
  assert zero `error` episodes.** Our runner logs `error` per episode.
* **GPU memory.** Torch peak is 14.4 GiB, but TF pre-allocates most of the remaining GPU memory (nvidia-smi showed
  23.0 GB used). Run SCALE on a dedicated GPU, or set `TF_FORCE_GPU_ALLOW_GROWTH=true`, before co-locating it with a
  π0.5 server. That was not tested here.
* Throughput: ~5.4 s per 100 control steps of policy compute; ~50 s for a successful LIBERO-10 episode,
  ~90 s for a 520-step timeout.

## Status

SCALE runs as released with the paper-default configuration on one 4090 in bf16 (no quantization needed). All S1
per-step fields are logged: success, uncertainty history, decoding temperature, visual attention temperature,
latency, and peak VRAM.
