# S2b research decision

Date: 2026-10-05. Label: **DISCOVERY** (initial states 20–29). The preregistration
(`configs/cag/S2b_prereg.yaml`) and the hashes of all run code were frozen in `results/S2b/PREREG_FROZEN.txt`
before launch.

No method was implemented, ω was not tuned (1.5), h was not changed (5 of 10), and states 30–49 were neither
inspected nor executed (client guard `--forbid-init-from 30`). S2c was not launched.

Details:
* `S2B_RNG_VALIDATION.md`
* `S2B_CAUSAL_STEERING_EFFECT.md`
* `S2B_COUNTERFACTUAL_GEOMETRY.md`
* `S2B_TRUST_HORIZON_ANALYSIS.md`

## 1. Paired episodes and validity

* **520 episodes = 260 B/S pairs.** Design: 2 master seeds × 13 tasks (0–12) × states 20–29.
  * 0 exceptions; 260/260 pairs valid.
  * Run time: 50 min on 2× RTX 4090.
* Exact common random numbers. Noise = f(master seed, task, state, call index), passed through the public
  `noise=` argument. CAG's shared cond/uncond noise is preserved.
  * Validated before rollout: repeat bit-identity; seed sensitivity; preserved marginal (KS p = 0.53 vs native);
    instrumentation identity 18/18; cross-process bit-identity 36/36.
  * This required disabling XLA autotuning. Otherwise identical noise gave actions up to 5.9e-3 apart across
    processes.
* Per pair, all 260/260:
  * identical geometry and initial state;
  * identical noise fingerprint at every common call;
  * identical call-0 observation;
  * B's call-0 chunk **bit-identical** to S's pre-CAG conditioned chunk.
* Seeds 0 and 1 share scenes (130/130) and differ only in policy noise.

## 2–4. Rescue, harm, semantic correction

| | seed 0 | seed 1 | pooled |
|---|---|---|---|
| faithful success B → S | 36 → 53 | 36 → 50 | **72 → 103 / 260** (+11.9 pp) |
| **rescue** rate | 19/130 | 18/130 | **37/260 = 14.2 %** (19.7 % of B failures) |
| **harm** rate | 2/130 | 4/130 | **6/260 = 2.3 %** (8.3 % of B successes) |
| **semantic correction** (biased-first → faithful-first) | 16/130 | 14/130 | **30/260 = 11.5 %** (20.5 % of B biased-first) |

* Exact McNemar on success: p = 2.2e-4 (seed 0), 4.3e-3 (seed 1). Pooled state-cluster permutation: p = 2e-4.
* Biased success falls from 146 to 110 (p < 1e-4).
* 27 of 37 rescues are semantic corrections. All 6 harms are faithful → faithful execution failures; 5 of them
  touched the right object and timed out.

## 5. Per-seed treatment-effect consistency (130 states)

| | observed | expected if seeds were independent |
|---|---|---|
| stable positive | **14** | 8.7 |
| stable negative | **0** | 0.0 |
| neutral | 102 | 96.6 |
| inconsistent | 14 | 24.7 |

* Every harm occurs in a single seed; no state is harmed twice.
* Native-noise floor (B seed 0 vs B seed 1): 6/130 success flips. B vs S within a seed: 43/260.
* **Rescues are largely robust intervention effects, concentrated by task.** Task 9 gives 16/20 rescues and task 7
  gives 8/20, while tasks 1, 4, 5, 10 and 11 give 0 of 98 B failures.
* **Harms look like stochastic interactions.**

## 6. Best H1 predictors (held-out)

Primary window W0 = S call 0 (pre-treatment):

| target | feature | AUROC [95 % CI] | LOTO | seed 0→1 / 1→0 | within-task |
|---|---|---|---|---|---|
| rescue | C_exec (prefix coherence) | 0.748 [0.621, 0.847] | 0.748 | 0.682 / 0.813 | 0.614 |
| rescue | L_early (early leverage) | 0.715 [0.602, 0.809] | 0.715 | 0.743 / 0.690 | 0.520 |
| semantic correction | C_exec | 0.746 [0.601, 0.868] | 0.746 | 0.698 / 0.796 | 0.625 |
| semantic correction | C_mean | 0.713 [0.564, 0.839] | 0.713 | 0.692 / 0.742 | 0.681 |

Secondary window W5 (calls 0–4, partially post-treatment): C_mean reaches 0.835 for rescue and 0.821 for
semantic correction, with every holdout ≥ 0.80.

Limits on these numbers:
* **The task-ID baseline beats every feature:** 0.875–0.939 in seed transfer.
* Multivariate models transfer across seeds but **fail leave-one-task-out:** logistic regression 0.47 for rescue,
  0.45 for semantic correction.
* Harm (6 positives): G @W0 0.83 and d5–d9 0.72–0.76. With 6 events these are anecdotal.

## 7. Negative control

| signal | H1 rescue AUROC (W0 / W5) | H2 ρ with future disagreement |
|---|---|---|
| image-only action→observation entropy (failed S3 signal) | **0.50 / 0.51** | **0.007 [−0.034, 0.048]** |
| generic action magnitude | 0.53 / 0.61 | 0.11 |
| vlm entropy (S3 lead) | 0.55 / 0.61 | 0.23 (replicates) |
| observation (prefix) attention mass | 0.69 / **0.79** | — |

The negative control is null again. Prefix attention mass predicts rescue about as well as residual coherence.

## 8. H1: **NO-GO** (FAIL)

The preregistered gate requires rescue **and** harm to be studyable. Harm occurred 6 times (floor 15). Residual
features do predict rescue and semantic correction above 0.65 under task and seed holdout (12 and 6 features
respectively). But:
1. With 2.3 % harm, "steer always" is already near-optimal, so an adaptive steer/abstain or strength-reduction rule
   has almost nothing to gain.
2. The rescue signal is dominated by task identity, and its learned combinations do not transfer to new tasks.

The coherence signal is a real semantic-reliability lead: it separates corrected from uncorrected biased states
within task (0.69–0.72, EXPLORATORY). It does not justify H1's method direction.

## 9. Best H2 correlations

Primary unit: call level, 5,849 calls. Target: future replanning disagreement D (S3 definition).

| feature | ρ [episode CI] | seeds | tasks same sign | within-episode | partial \| L_mean | vs D_cond |
|---|---|---|---|---|---|---|
| **J** (residual instability) | **0.493 [0.456, 0.529]** | 0.49 / 0.50 | 13/13 | 0.43 | 0.28 | 0.38 |
| J_exec | 0.360 [0.321, 0.398] | 0.36 / 0.36 | 12/13 | 0.27 | 0.12 | 0.27 |
| d[k], within-k | 0.358 [0.321, 0.396] | 0.35 / 0.37 | 13/13 | 0.33 | ≈ 0 (pure magnitude) | 0.23 |
| G, C_mean, C_exec | 0.12, −0.07, 0.00 | | | | | |

Comparators: L_mean 0.43; call index 0.38.

**EXPLORATORY:** with residual magnitude, call index and the conditioned chunk's own roughness controlled jointly,
J's unique partial ρ is **0.07**.

## 10. H2: **GO** (PASS)

J and J_exec meet every preregistered criterion:
* |ρ| ≥ 0.20 with the CI excluding 0;
* same sign in both seeds and in ≥ 10 tasks.

They are neither magnitude-driven (partial | L = 0.28) nor mechanical (ρ vs D_cond = 0.38). d[k] passes
numerically but is redundant with residual magnitude.

Caveats:
* Disagreement is not failure.
* No horizon was varied, so nothing here is causal about chunk length.
* Most of J's predictive content is shared with generic signals.

## 11. Do guidance effects differ before and after object contact?

**Yes.** Within episodes (n = 242; both seeds agree):

| quantity | change after contact |
|---|---|
| residual magnitude | +28 % [+21, +34] |
| residual instability J | +26 % (+106 % in the short contact-before-close window, n = 28) |
| executed CAG shift | +15 % |
| coherence | flat |
| language attention mass | ≤ 3 % change, B and S alike |

CAG pushes harder and less coherently during manipulation. The 6 harm episodes have the highest post-contact
residual (0.46 vs 0.39) and instability (0.18 vs 0.13), a lead consistent with "semantic guidance interferes with
execution". The preregistered contact-pre-close phase is nearly empty (31 calls), because π0.5 closes the gripper
before contact registers in 157 of 242 episodes.

## 12. Recommendation: **C, temporal-commitment-only research, reconsider novelty before building anything**

By the preregistered matrix, H1 FAIL + H2 PASS gives **C**. A (joint) and B (semantic-only adaptive steering) are
ruled out: at ω = 1.5, CAG is beneficial (+11.9 pp) and almost never harmful, so adaptive semantic steering has
nothing to avoid.

The novelty reconsideration that C requires is not encouraging so far:
* The H2 signal is robust (ρ ≈ 0.5, 13/13 tasks, both seeds), but **its unique, counterfactual-specific part is
  small** (partial ρ 0.07 after controlling magnitude, phase and chunk roughness).
* "Internal signal → adaptive chunk length" is already the core claim of Knowing When to Stop and GeoAAC.
* A J-based trust horizon is only worth building if it beats those generic signals at choosing h. That is an
  interventional question S2b did not test.

So C is conditional:
* The next experiment should be a cheap, preregistered **falsification** on reserved states 30–39 (not launched
  here). Does executing fewer steps when J is high reduce failures or disagreement more than the same rule driven
  by call index, residual magnitude or chunk roughness alone?
* If it does not, the outcome is **D (stop / pivot)**.

Not part of the matrix, and not a decision: S2b also exposed a different open problem. Five CF-Spatial tasks (all
four "ramekin" prompts and "black bowl next to the plate") fail on ≈ 100 % of states under π0.5, and fixed-ω CAG
rescues none of them (0/98). If the programme pivots, the evidence points at **failures that fixed-strength
language amplification cannot fix**, not at gating an intervention that rarely hurts.

---

S2b stops here. No method implemented, no ω tuning, no change to h, states 30–49 untouched, S2c not launched.
