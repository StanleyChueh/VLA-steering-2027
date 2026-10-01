# Pilot decision (after S1 minimal, S2 pilot, S3 attention pilot)

Date: 2026-10-01. All results are **PILOT**: S1 n=10/task/condition, S2 n=20/task/condition. Nothing here is
confirmatory. Details: `S1_SCALE_MINIMAL.md`, `S2_CAG_PILOT.md`, `S3_ATTENTION_PILOT.md`. The preregistration
(`configs/cag/S2_pilot_prereg.yaml`) and frozen task manifest were committed in `58b5667` before any outcome.

## Headline numbers

| | Baseline | Method |
|---|---|---|
| S1 LIBERO-10 tasks 0/4/9, success | greedy 17/30 = 0.57 [0.39, 0.73] | SCALE 16/30 = 0.53 [0.36, 0.70] |
| S2 CF-Spatial tasks 0/6/12, faithful success | B 48/60 = 0.80 [0.68, 0.88] | CAG-TF 47/60 = 0.78 [0.66, 0.87] |
| S2 biased success | B 6/60 = 0.10 [0.05, 0.20] | CAG-TF 3/60 = 0.05 [0.02, 0.14] |
| S2 faithful touch / biased touch | B 54/60 / 6/60 | CAG-TF 58/60 / 6/60 |
| Exception episodes | 0 in all 4 arms (180 episodes) | |
| Peak VRAM | SCALE 14.42 GiB torch (≈23 GB nvidia-smi, TF pre-allocation) | π0.5 6.64 GiB (B) / 6.61 GiB (CAG-TF), incl. diagnostics |

## 1. Is released-code SCALE operational and qualitatively sane?

**Yes.** It ran 30/30 episodes without exceptions on matched scenes (30/30 identical geometry and initial state),
at the same latency as greedy (171 vs 170 ms/step). The mechanism behaves as the audited equations predict: γ
adapts within (0.5, 2) at 99.5 % of steps, and τ_k is near-greedy for confident tokens with heat on the
translation dimensions. **No benefit was detected** (−1/30). With n=30 per arm this is consistent both with the
paper's +6.4 pp on these tasks and with zero. The full 500-episode/arm definition is in the S1 report and was not
launched.

## 2. Does CAG-TF measurably alter semantic / grounding behaviour?

**Actions: yes. Semantic outcomes: not demonstrated.**
* Language masking changes the predicted chunk substantially: normalised ‖a_cond − a_uncond‖ is 0.47 per step,
  and the executed shift is 0.5× that.
* CAG raises replanning disagreement from 0.35 to 0.52.
* None of the four semantic outcomes moved beyond pilot noise. Biased success 6 → 3/60 is the only movement in the
  paper's direction. On task 0, the only selected task with real counterfactual failures, CAG produced more
  "touched the right object, timed out" episodes (faithful success 11 → 8/20).

The pilot is weakly informative: vanilla π0.5 is near ceiling on tasks 6 and 12 (biased success 1/20 each), so
CAG has almost nothing to rescue there. Without noise pairing or a B-vs-B replicate, the 13/60 state-matched flips
cannot be separated from native stochasticity.

## 3. Does action-expert attention entropy correlate with horizon or unreliability?

* **Horizon: weakly, and it depends on the definition.** Image-only normalised entropy is an inverted U over k
  with ≈2 % range. Only the KWTS aggregate (image + language, layer-summed, N = 968) rises monotonically.
* **Unreliability: no, for the preregistered measure.**
  * Image-only entropy vs replanning disagreement: ρ = −0.05 [−0.11, 0.02] (B), +0.04 [−0.02, 0.10] (S).
  * Mean, max, late-horizon and first-call entropy do not predict failure or biased interaction (all AUROC CIs
    include 0.5).
  * The entropy slope (AUROC ≈ 0.75) is a symptom of failing trajectories, not a predictor: restricted to the
    first 5 calls it falls to 0.50 in B.
  * The published 0.95·ln 968 threshold is unreachable: the maximum observed is 0.58.
* Exploratory lead: image+language entropy correlates with replanning disagreement (ρ ≈ 0.2) and survives
  within-episode demeaning.

**S3 gate: FAIL.** S4 (paper-derived adaptive chunking) is not justified.

## 4. Are there benchmark/data defects that invalidate any task?

| Defect | Affects | Status |
|---|---|---|
| Prompt ≠ BDDL `:language`, and the BDDL language would make the biased object the instructed one | CF-Spatial tasks **13, 14** | **Invalid for primary analysis.** Excluded; EXPLORATORY only. The paper's 15-task CF-Spatial averages likely include them. |
| LIBERO-CF does not encode faithful/biased | all LIBERO-CF metrics | Reconstructed labels; physically verified on all 50 initial states of all 15 tasks |
| Fixture placement re-sampled at every `env.reset()` (hard_reset), dependent on env seed and reset order | LIBERO and LIBERO-CF, every task | Not invalidating, but "initial-state index" does not fix the scene. Matched comparisons need identical reset order; verified here (60/60 S2, 30/30 S1). |
| Vanilla π0.5 at ceiling on CF tasks 6 and 12 | this pilot's task selection | Not a defect. The deterministic selection rule picked tasks with no semantic failures to fix. |

No defect found in tasks 0–12 or in the LIBERO-10 tasks used.

## 5. Which direction?

**A. Semantic-only research, as a conditional GO.** Not B, not C.

* **B (action-side) is closed by the S3 gate.** Image-attention entropy shows no reliability signal on π0.5.
* **C (complementarity) needs both a working S and a working A.** Neither is established.
* **A is not yet supported or refuted.** The CAG pilot could not test semantic steering because the selected tasks
  have few semantic failures, and the cond/uncond gap is a live candidate for a semantic reliability signal. Two
  early-guidance observations (n=2) and the vlm-entropy correlation point in that direction, but they are leads,
  not results.
* **Fallback to D:** if a properly targeted semantic test (below) also shows no CAG effect on biased behaviour, the
  evidence matches CASE 5. Then pivot to external generate-and-verify methods (VERITAS / VLS) rather than building
  an internal steering method.

SCALE (OpenVLA) is context for the landscape, not part of the π0.5 programme. Its full 500-episode/arm reproduction
is optional and can use the idle second GPU.

## Recommended next experiment (preregister before launching)

**S2b: CAG-TF on tasks that actually produce counterfactual failures.**

1. **Baseline screen (B only).** All 13 valid CF-Spatial tasks, initial states 20–29 (disjoint from the pilot's
   0–19), 130 episodes, about 30–40 min on one 4090. Freeze the inclusion rule *before* the screen, e.g. biased
   touch ≥ 3/10. The screen measures baseline failure only, never a CAG effect.
2. **Evaluation.** B vs CAG-TF (ω=1.5) on the selected tasks, using **fresh** initial states 30–49 to avoid
   regression to the mean from screen selection, 20 episodes/task/condition.
3. **Native-noise flip rate.** Add a B replicate with `POLICY_SEED=1`, to measure how many state-matched outcomes
   flip with no intervention.
4. **Preregistered secondary analysis.** Does early guidance magnitude ‖a_cond − a_uncond‖ predict biased
   interaction (the semantic-reliability lead)?

## Exact command to run next

```bash
cd ~/Stanley_ws/VLA-steering-2027
# S2b step 1 - baseline-only failure screen (freeze the inclusion rule in configs/cag/ before launching)
CONDS=B TASKS=0,1,2,3,4,5,6,7,8,9,10,11,12 EPISODES=10 INIT_START=20 RAW_EPISODES=0 PORT=8765 \
  bash scripts/run_cag/run_b_vs_s.sh 0 1.5 results/S2b_screen
```

Not launched. Per instructions, this phase stops here: no adaptive chunking, no router, no 50-episode runs.
