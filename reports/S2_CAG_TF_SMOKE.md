# S2 smoke — π0.5 vs π0.5 + CAG-TF on LIBERO-CF (CF-Spatial task 0)

Label: **SMOKE TEST (n=2 per condition; no inferential content).** Date: 2026-10-01.

## Setup

| Item | Value |
|---|---|
| Suite / task | `libero_cf_spatial`, task 0 = `01-pick_up_the_cookie_box_and_place_it_on_the_plate` |
| Prompt sent | "pick up the cookie box and place it on the plate" (file-name derived) |
| Faithful / biased | `on cookies_1 plate_1` / `on akita_black_bowl_1 plate_1` (`src/metrics/libero_cf_labels.py`) |
| Initial states | indices 0, 1 (same for both conditions) |
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` (CRC32C-verified) |
| B | vanilla π0.5 (`create_trained_policy`, as `serve_policy.py`) |
| S | CAG-TF, LIBERO-CF's own `CounterfactualActionGuidedPolicy`, **ω = 1.5** (paper App. B; README says 2.0, see S0 C1), `uncond_empty_prompt=False` |
| Client | `scripts/run_cag/run_cf_eval.py` (mirrors `main_cf.py`): seed 7 (np + env), replan 5 of 10, 220 max steps, 10 wait steps |
| Policy RNG | fresh server per condition → stream starts at `jax.random.key(0)`; per-call key fingerprints logged |
| GPU | RTX 4090 #0, `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| Command | `bash scripts/run_cag/run_b_vs_s.sh 0 1.5 results/S2_smoke` (script then named `run_smoke.sh`; identical apart from the task/episode parameters) |
| Raw records | `results/S2_smoke/{B,S}/episodes.jsonl` (+ `manifest.json`, `server.log`, `client.log`) |

## Results

| Condition | faithful touch | biased touch | faithful success | biased success | mean length (steps) | policy calls / ep |
|---|---|---|---|---|---|---|
| B (vanilla) | 2/2 | 0/2 | **2/2** | 0/2 | 103 | 21 |
| S (CAG-TF ω=1.5) | 2/2 | 0/2 | **1/2** | 0/2 | 167 | 33.5 |

Per episode (init state → outcome):

| init | B | S |
|---|---|---|
| 0 | faithful success, 105 steps | faithful success, 114 steps |
| 1 | faithful success, 101 steps | faithful touch, **no success**, timed out at 220 steps |

Paired (n=2): rescue 0/0, harm 1/2. With n=2 this is not evidence of anything. For reference, LIBERO-CF's README
example for this task (π0.5, 50 rollouts) reports faithful success 25/50 and biased-bowl success 15/50, so task 0
is not saturated at the 50-episode scale.

## Cost

| | B | S |
|---|---|---|
| server inference, median (ms) | 57.3 | 133.1 (two full 10-step ODE passes) |
| client round-trip, median (ms) | 75.9 | 146.6 |
| first call incl. JIT compile (s) | 9.9 | 10.8 |
| peak GPU memory (GiB, JAX) | 6.68 | 6.84 |

## Checks

* Both conditions completed without exceptions (`error: null` for all episodes).
* Both servers started from the same RNG key, and the first episode's per-call key fingerprints are identical in B
  and S (`af5570f5…`, `3b658cb4…`, `c0750de6…`). With JAX 0.5.3, vanilla `split(k,2)` and CAG `split(k,4)` give
  bit-identical noise at equal call index (verified). Episode 1 started at different stream positions because
  episode 0 used 21 vs 23 calls, so **noise is paired only until the call counts diverge**.
* Touch/success labels resolved for all four conditions. No success without a recorded touch.

## Status

The CAG-TF pipeline runs end to end with the public checkpoint and produces all S2 per-episode fields: suite,
task, instruction, init-state index, env seed, faithful/biased touch/success, length, policy calls, latency, GPU
memory, executed actions, full predicted chunks, and policy-noise key fingerprint. Not yet logged: per-call guidance
magnitude ‖a_cond − a_uncond‖. That requires returning both branches from the server (a wrapper-only change),
and it is needed for S5 mechanistic outcome 11.
