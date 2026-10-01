# S1 minimal — released-code SCALE reproduction (OpenVLA, LIBERO-10)

Label: **PILOT** (10 episodes / task / condition). **Released-code SCALE reproduction**: the public implementation
is run as released, with no attempt to fix the paper/code discrepancies from S0 (A1 `attn_sensitivity=0.3`,
A2 q_low ε parameterisation). Date: 2026-10-01.

## Setup

| Item | Value |
|---|---|
| Suite / tasks | `libero_10`, task IDs 0, 4, 9 (first/middle/last, fixed before any run) |
| Instructions | 0: "put both the alphabet soup and the tomato sauce in the basket"; 4: "put the white mug on the left plate and put the yellow and white mug on the right plate"; 9: "put the yellow and white mug in the microwave and close it" |
| Episodes | 10 per task per condition, initial states 0–9 in the same order for both conditions |
| Conditions | `greedy` vs `scale` (`configs/scale.yaml` defaults: T0=1.0, κ=2.0, α=0.8, ε=1e-12, num_logits=256, attn_sensitivity=0.3) |
| Checkpoint | `openvla/openvla-7b-finetuned-libero-10` @ `80970322`, bf16 + FlashAttention-2 |
| Seeds | `set_seed_everywhere(7)` (released default); env `env.seed(0)` (hard-coded in SCALE's `get_libero_env`) |
| Max steps / wait | 520 / 10 (released) |
| Hardware | RTX 4090 #1 (dedicated; TF pre-allocates the rest of the GPU) |
| Command | `bash scripts/run_scale/run_greedy_vs_scale.sh 1 libero_10 0,4,9 10 results/S1_minimal` |
| Records | `results/S1_minimal/{greedy,scale}/episodes.jsonl` + `manifest.json`; summary `results/S1_minimal/summary.json` (`src/analysis/summarize_scale.py`) |

Provenance note: S1 was launched minutes before commit `58b5667`, so its manifests record `our_repo = a9ba17b,
dirty=true`. The runner and script that ran are byte-identical to the versions committed in `58b5667`.

## Results (success, Wilson 95% CI)

| Task | greedy | SCALE | paper (OpenVLA greedy* / SCALE, Table 11, 3×50 eps) |
|---|---|---|---|
| 0 (soup + sauce → basket) | 7/10 = 0.70 [0.40, 0.89] | 6/10 = 0.60 [0.31, 0.83] | 62.7 / 66.0 |
| 4 (two mugs → plates) | 4/10 = 0.40 [0.17, 0.69] | 4/10 = 0.40 [0.17, 0.69] | 49.3 / 51.3 |
| 9 (mug → microwave, close) | 6/10 = 0.60 [0.31, 0.83] | 6/10 = 0.60 [0.31, 0.83] | 40.0 / 54.0 |
| **Pooled** | **17/30 = 0.57 [0.39, 0.73]** | **16/30 = 0.53 [0.36, 0.70]** | 50.7 / 57.1 (mean of these 3) |

* Exceptions: **0 / 30 greedy, 0 / 30 SCALE.** Every failure is a behavioural failure (520-step timeout).
* State-matched transitions (descriptive, **not noise-controlled**): greedy→SCALE F→S 5, S→F 6, S→S 11, F→F 8.
* Geometry: static scene fingerprint identical for 30/30 matched pairs, and initial simulator state identical for
  30/30. Both conditions saw the same scenes.

Both arms are within sampling error of the paper's per-task numbers. The pilot **cannot distinguish** the paper's
+6.4 pp mean gain on these tasks from zero: the SE of a 30-vs-30 difference at p≈0.55 is ≈13 pp.

## Mechanism (SCALE arm, 12,013 control steps)

| Quantity | Observed | Expected from S0 audit |
|---|---|---|
| Visual-attention temperature γ applied | γ=1 at the first two steps of every episode, then ≠1 at 99.5% of steps; median 0.98, p05–p95 0.61–1.71, range 0.51–1.98 | γ = κ^tanh(0.3·Δu) ∈ (0.5, 2) with κ=2; γ=1 for the first two steps (Alg. 1 timing) |
| Token uncertainty u_k | step mean −1.79; p05 −5.01, p95 1.96 | u ∈ [−5.55, 22.1]; u=0 at p_max≈0.80 |
| Decoding temperature τ_k = σ(u_k) | median 0.0064; 66% of tokens τ<0.1. Per-dimension median: x 0.081, y 0.032, z 0.054, rotations 0.004–0.009, gripper 0.0039 | near-greedy when confident, up to τ→1 when uncertain |
| EMA ū | mean −1.81 | α=0.8 smoothing of u |

The mechanism is active, adaptive and within its designed ranges. Most tokens are decoded almost greedily, and
sampling heat goes to the translation dimensions. Mean u is −1.58 in successful and −1.92 in failed episodes. The
mean u over the first 20 steps does not separate outcomes (AUROC for failure 0.54). This is exploratory only.

## Cost

| | greedy | SCALE |
|---|---|---|
| Policy latency per step, median (p95) | 170.5 ms (175.1) | 171.4 ms (174.1) |
| Peak torch memory | 14.43 GiB | 14.42 GiB |
| nvidia-smi on the GPU | ≈23 GB (TF 2.15 pre-allocation) | same |
| Mean episode length (steps) | 383 | 400 |

SCALE is single-pass and costs nothing measurable over greedy. It fits one 4090 in bf16, so no quantisation was
used or needed.

## Verdict

* **Operational and qualitatively sane:** yes. No exceptions, the mechanism behaves as the equations predict, and
  scenes are matched.
* **Effect:** not detected (−1/30 pooled). This is consistent with both the paper's gain and no gain. Nothing is
  claimed either way.

## Defined, not launched: full released-code reproduction

To test the paper's LIBERO-Long claim (52.7 → 63.3 %), run all 10 tasks × 50 initial states × {greedy, SCALE}
(500 episodes per arm, about 11 h per arm on one 4090). The SE of the difference is then ≈3.1 pp, so the claimed
+10.6 pp would be detectable. Use 3 seeds if the paper's ± values are to be checked.

```bash
bash scripts/run_scale/run_greedy_vs_scale.sh 1 libero_10 0,1,2,3,4,5,6,7,8,9 50 results/S1_full_seed7
```
