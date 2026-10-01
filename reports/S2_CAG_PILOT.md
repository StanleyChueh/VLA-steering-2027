# S2 pilot — π0.5 vs π0.5 + CAG-TF (ω = 1.5) on LIBERO-CF Spatial

Label: **PILOT** (20 episodes / task / condition). RNG protocol: **matched initial states under native policy
stochasticity.** These results are not noise-paired. Date: 2026-10-01.

## Preregistered changes made before the run (commit `58b5667`, before any outcome existed)

1. **ω frozen.** ω = 1.5 is PRIMARY / PREREGISTERED (paper App. B / §V: "ω = 1.5 for both TF and VA"). ω = 2.0
   (README example) is FUTURE SENSITIVITY ANALYSIS ONLY and was not run.
2. **Clean task manifest:** `configs/cag/libero_cf_spatial_task_validity.csv` (+ `.details.json`), produced by
   `scripts/audit/audit_cf_task_validity.py`. Every task was checked textually (prompt actually sent vs BDDL
   `:language`; labels from `src/metrics/libero_cf_labels.py`) and physically on all 50 stored initial states, in
   the simulator built exactly as `main_cf.py` builds it (seed 7). The physical checks:
   * faithful and biased objects resolve to bodies;
   * no listed condition holds at the initial state;
   * for qualifier prompts ("black bowl on the cookie box / next to the ramekin / next to the plate / on the wooden
     cabinet"), the labelled faithful instance is the one that physically satisfies the qualifier: 50/50 initial
     states for tasks 6, 9, 11, 12.

   **Tasks 0–12: valid, label confidence high. Tasks 13 and 14: excluded.** Their BDDL `:language` disagrees with
   the prompt, and in both cases the BDDL language would make the *biased* object (akita_black_bowl_1) the
   instructed one, so the counterfactual is undefined. Rule "first, median, last valid task" → **0, 6, 12.**
   Labels are frozen and asserted at run time against the committed CSV (`tests/test_libero_cf_labels.py`).
3. **Guidance-magnitude logging** and 4. **action-expert attention logging** go through
   `src/instrumentation/pi05_diagnostics.DiagnosticPolicy`. It intercepts the policy's `_sample_actions` calls,
   returns their outputs untouched, and computes diagnostics on the side. Identity/regression test
   (`scripts/audit/diagnostics_identity_test.py`, `results/S2_identity/`): stock path (instrumentation never
   loaded) vs production S2 path, 3 real observations fed twice:

   | mode | actions bit-identical | max\|Δ\| | RNG state after sequence | diagnostic re-run x0 bit-identical |
   |---|---|---|---|---|
   | vanilla | 6/6 calls | 0 | identical | 6/6 |
   | CAG-TF | 6/6 calls | 0 | identical | 12/12 |

   In the pilot itself, every diagnostic re-run reproduced its branch output bit-for-bit: **1414/1414 (B) and
   3054/3054 (S)**. CAG reconstruction `uncond + ω(cond − uncond) = returned action` holds to ≤3.0e-7.
5. **Geometry logging:** `src/wrappers/geometry.py`. A finding during setup: LIBERO's `hard_reset` rebuilds the
   model at every `env.reset()`, and **fixture placement (cabinet, stove) is re-sampled at every reset**. All 50
   initial states of a task have different static geometry. The placement is a deterministic function of (env
   seed, reset count) and is unaffected by stepping, so matched episodes share geometry only if both runs reset in
   the same order. This is verified below.
6. **RNG:** fresh server per condition, native `jax.random.key(0)` stream, no per-episode reset.

## Setup

| Item | Value |
|---|---|
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` (CRC32C-verified in S0) |
| B | vanilla π0.5 (`create_trained_policy`) |
| S | LIBERO-CF `CounterfactualActionGuidedPolicy`, ω=1.5, language-token-masked unconditional branch |
| Tasks / episodes | 0, 6, 12 × initial states 0–19, same order in both conditions; env seed 7; replan 5 of 10; 220 max steps |
| Command | `TASKS=0,6,12 EPISODES=20 RAW_EPISODES=2 PORT=8765 bash scripts/run_cag/run_b_vs_s.sh 0 1.5 results/S2_pilot` |
| Analysis | `envs/openpi-cf/bin/python src/analysis/analyze_s2.py results/S2_pilot` → `results/S2_pilot/summary.json` |

## Matching checks

| Check | Result |
|---|---|
| Static geometry identical, B vs S, per (task, init) | **60/60** |
| Initial simulator state identical | **60/60** |
| First-call RNG key / flow noise identical, B vs S | yes / yes (same stream start; streams diverge once call counts differ) |
| Exception episodes | **0 (B), 0 (S)** |

## Primary outcomes (Wilson 95% CI)

| Task | Cond | faithful touch | biased touch | faithful success | biased success |
|---|---|---|---|---|---|
| 0 cookie box | B | 16/20 = 0.80 [0.58, 0.92] | 4/20 = 0.20 [0.08, 0.42] | 11/20 = 0.55 [0.34, 0.74] | 4/20 = 0.20 [0.08, 0.42] |
| | S | 18/20 = 0.90 [0.70, 0.97] | 5/20 = 0.25 [0.11, 0.47] | 8/20 = 0.40 [0.22, 0.61] | 2/20 = 0.10 [0.03, 0.30] |
| 6 bowl on cookie box | B | 19/20 = 0.95 [0.76, 0.99] | 1/20 = 0.05 [0.01, 0.24] | 18/20 = 0.90 [0.70, 0.97] | 1/20 = 0.05 [0.01, 0.24] |
| | S | 20/20 = 1.00 [0.84, 1.00] | 0/20 = 0.00 [0.00, 0.16] | 20/20 = 1.00 [0.84, 1.00] | 0/20 = 0.00 [0.00, 0.16] |
| 12 bowl on cabinet | B | 19/20 = 0.95 [0.76, 0.99] | 1/20 = 0.05 [0.01, 0.24] | 19/20 = 0.95 [0.76, 0.99] | 1/20 = 0.05 [0.01, 0.24] |
| | S | 20/20 = 1.00 [0.84, 1.00] | 1/20 = 0.05 [0.01, 0.24] | 19/20 = 0.95 [0.76, 0.99] | 1/20 = 0.05 [0.01, 0.24] |
| **Pooled** | **B** | 54/60 = 0.90 [0.80, 0.95] | 6/60 = 0.10 [0.05, 0.20] | **48/60 = 0.80 [0.68, 0.88]** | **6/60 = 0.10 [0.05, 0.20]** |
| | **S** | 58/60 = 0.97 [0.89, 0.99] | 6/60 = 0.10 [0.05, 0.20] | **47/60 = 0.78 [0.66, 0.87]** | **3/60 = 0.05 [0.02, 0.14]** |

Secondary: mean episode length B 116 vs S 126 steps; policy calls 23.6 vs 25.4. In S, 4 episodes touched both
objects (B: 0). No episode touched neither object.

**No statistical improvement is claimed.** Every interval overlaps. The only direction consistent with the paper is
biased success 6 → 3 / 60 (paper, π0.5 CF-Spatial: 56.9 % → 33.3 %).

### State-matched transitions (descriptive; NOT noise-controlled)

| | count |
|---|---|
| faithful success B→S: 1→1 / 1→0 / 0→1 / 0→0 | 41 / 7 / 6 / 6 |
| "rescue" (S succeeds \| B fails) | 6/12 |
| "harm" (S fails \| B succeeds) | 7/48 |
| task 0 only: 1→1 / 1→0 / 0→1 / 0→0 | 5 / 6 / 3 / 6 |
| biased touch B→S: 0→1 / 1→0 / 1→1 | 3 / 3 / 3 |
| outcome class: B-success→F-success / B-success→F-touch-no-success / F-success→B-success | 2 / 2 / 1 |

Pairs share the initial state and scene, but **not** the policy noise after the first divergence of call counts.
No B-vs-B replicate exists, so the share of these flips that native stochasticity alone would produce is
**unknown**. McNemar is not reported as a test for that reason.

## Why the pilot is weakly informative about semantic steering

Vanilla π0.5 makes few counterfactual errors on the selected tasks. Tasks 6 and 12 are at ceiling (faithful
success 18/20, 19/20; biased success 1/20 each). Only task 0 shows the vision-over-language failure CAG targets
(biased success 4/20). The paper's π0.5 CF-Spatial averages (15 tasks: faithful grounding 39.3 %, biased
grounding 61.3 %, faithful success 24.4 %, biased success 56.9 %) imply that strong counterfactual failures are
concentrated in other tasks. Our task 0 agrees with LIBERO-CF's README example for that task (25/50 faithful,
15/50 biased success). Two other points:
* The paper's 15-task average includes tasks 13 and 14, which we exclude as defective. In the S0 pipeline check
  on task 13, vanilla π0.5 completed the *biased* goal.
* The deterministic first/median/last rule did what it was meant to (no outcome-based selection). Its consequence
  is that 2 of 3 tasks have almost no semantic failure for CAG to fix.

## Mechanistic observations (S condition)

| Quantity (env action space unless noted; per call, horizon k = 0..9) | Value |
|---|---|
| ‖a_cond − a_uncond‖ translation, mean over executed k=0..4 | 0.256 (executed B action translation norm ≈ 0.79) |
| rotation | 0.041 |
| \|gripper diff\| | mean 0.086, median 0.007 (rare large disagreements) |
| normalised combined (quantile space, dims 0–6) | mean 0.47, median 0.35, p95 1.66 |
| executed CAG action = a_cond + (ω−1)(a_cond − a_uncond) | i.e. a shift of 0.5× the above per step |
| replanning disagreement D_combined (S3 definition) | B 0.35 → S 0.52 |

Masking the language changes the predicted chunk substantially. CAG is therefore not a no-op: it moves executed
actions by roughly 15 % of a typical translation command, and it makes successive chunks disagree more. In task 0
the effect is mostly on execution: S ends more often as "faithful touch, no success" (timeouts after touching the
right object).

Exploratory (n=2, post hoc): in task 0, the two S episodes where the robot went *only* for the biased object had
among the lowest early guidance magnitudes of the task: mean over the first 5 calls 0.149 (lowest of 20) and 0.194
(3rd lowest). The other 18 ranged 0.151–0.498, and the 0.151 episode also touched the biased object. This fits "vision overrides language": when the language-masked and conditioned branches agree,
CAG has nothing to amplify. It is a lead for a semantic-reliability signal, not a result.

## Cost

| | B | S |
|---|---|---|
| Policy inference, median (the stock computation) | 53.9 ms | 133.2 ms (two full 10-step ODE solves) |
| Server wall time incl. diagnostics, median | 139 ms | 282 ms |
| Peak GPU memory incl. diagnostics (JAX) | 6.64 GiB | 6.61 GiB |
| Storage: attention summaries / raw probs (first 2 eps/task) | 344 MB / 3.2 GB | 735 MB / 9.0 GB |

## Verdict

CAG-TF runs as published, at the paper's ω, with byte-exact provenance. It **measurably changes actions** (large
cond/uncond gap, higher chunk disagreement). On the four semantic outcomes it shows **no detectable effect** in
this pilot. The pilot is underpowered for semantic effects because the selected tasks rarely produce semantic
failures under vanilla π0.5. Excluded tasks 13/14 remain available for EXPLORATORY evaluation only.
