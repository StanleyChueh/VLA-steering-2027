# S2c-0 research decision: semantic controllability mechanism

Date: 2026-10-05. Label: **DISCOVERY / MECHANISTIC.** No rollouts, CAG unchanged, ω not tuned, h unchanged, no new
method, states 30–49 untouched. Preregistration `configs/cag/S2c0_prereg.yaml`, frozen in
`results/S2c0/PREREG_FROZEN.txt` before any language query.

Details:
* `S2C0_SEMANTIC_CONTROLLABILITY.md`
* `S2C0_CAG_DIRECTION_ANALYSIS.md`
* `S2C0_LANGUAGE_ACTION_MANIFOLD.md`

## Decision: **D, no convincing controllability mechanism under the preregistered rules → STOP the semantic-controllability hypothesis before consuming states 30–39. S2c-1 (ω response surface): NO-GO.**

## What was measured

* **Inputs.** The 130 S2b call-0 policy inputs (13 tasks × states 20–29), regenerated and matched to S2b's
  geometry and observation hashes in 130/130 scenes.
* **Queries.** Each scene was queried with its four benchmark instructions plus CAG's masked branch under
  identical noise (2 CRN seeds + 8 secondary).
* **Reproduction.** The faithful and masked chunks reproduce S2b's executed inputs bit-for-bit (260/260).
* **Groups (from S2b):** G0 zero-rescue = {1, 4, 5, 10, 11}; G1 rescued = {0, 2, 3, 7, 8, 9, 12}.

| metric (task medians) | G0 | G1 | Cliff's δ | median diff 95 % CI | preregistered deficiency |
|---|---|---|---|---|---|
| SCNR (faithful-vs-biased / noise) | 2.09 | 5.56 | −0.60 | [−4.95, +0.24] | no (and not "comparable") |
| S_fb (raw contrast) | 1.04 | 1.73 | −0.49 | [−2.70, +0.13] | no |
| N_f (policy noise) | 0.56 | 0.46 | +0.37 | [−0.15, +0.27] | — |
| **ALIGN** cos(R_cag, R_fb) | **0.12** | **0.82** | −0.71 | [−0.98, −0.12] | **yes** |
| **P_cov** (gap covered by CAG at ω = 1.5) | **0.06** | **0.30** | −0.71 | [−0.43, −0.15] | **yes** |
| **faithful separation / N_f** | **2.80** | **4.93** | −0.77 | [−3.58, −0.62] | **yes** |
| effective rank / spread SNR | 1.79 / 3.45 | 1.59 / 4.70 | +0.31 / −0.43 | include 0 | no |
| **generic action magnitude ‖A_f‖** | 1.290 | 1.361 | **−1.00** | [−0.088, −0.024] | (control) |

![rescued vs zero-rescue](figures/S2c0_rescued_vs_zero_rescue.png)

## Preregistered rule evaluation

| rule | result |
|---|---|
| collapse: deficiency(SCNR or faithful separation/N_f) + generic control | deficiency on separation only; **generic control fails** |
| misorientation: SCNR comparable (δ > −0.33) + deficiency(ALIGN) + generic control | SCNR δ = −0.60 is not comparable; **generic control fails** |
| amplitude: SCNR and ALIGN comparable + deficiency(P_cov) | **no** (neither is comparable) |
| mixed: ≥ 2 G0 tasks below the G1 minimum on SCNR, and ≥ 2 others on ALIGN/P_cov | **no**: rescued task 7 is the G1 minimum on all three, so no G0 task lies below it |
| generic control: \|δ(metric)\| > \|δ(‖A_f‖)\| and > \|δ(N_f)\| | **fails for every metric**: ‖A_f‖ separates the groups perfectly (δ = −1.00) |

S2c-1 gate:
1. coherent relationship with rescue — **yes** (three deficiencies; task ρ 0.82 for faithful separation/N_f);
2. reproducible G0 deficiency consistent with amplitude- or direction-limited control — **no** (decision D);
3. not explained by generic magnitude or noise — **no**.

**GO = false.**

## Answers to the phase questions

* **A (low semantic separability / low SCNR)?** **No.**
  * Zero-rescue tasks have SCNR ≈ 2: language moves the action about twice as far as policy noise does.
  * Their four-instruction manifolds are as spread as the rescued tasks' (δ = −0.09).
  * They are not semantically collapsed.
* **B (adequate separability, poor CAG alignment)?** **Descriptively yes, but not certified.**
  * CAG's direction is nearly orthogonal to the faithful-vs-biased contrast in G0 (ALIGN 0.12 vs 0.82).
  * CAG covers 6 % of the gap vs 30 %.
  * The faithful chunk sits closest to the training-bowl chunk.
  * The preregistered rules do not license this conclusion: generic magnitude separates the groups as well, and
    SCNR is not "comparable".
* **C (neither)?** This is the formal outcome.

## Why the outcome is D, and how strong the evidence against a mechanism is

**The decision follows the rules as frozen.** These caveats are reported and do not change it:

1. **The generic control is rank-based and could not be passed here.**
   * With 5 vs 7 tasks, any task-level quantity aligned with the groups reaches |δ| = 1.
   * ‖A_f‖ does, although its difference is only about 5 % (1.29 vs 1.36). It is mostly translation magnitude
     at call 0 (δ = −0.94); the gripper term is constant.
   * The control therefore cannot tell "confounded by magnitude" apart from "magnitude is another symptom". In
     zero-rescue tasks the faithful instruction also produces a slightly weaker first motion, which is what
     weak grounding would look like.
   * Lesson for future preregistrations: use a magnitude-aware control (partial correlation, or matched effect
     size), not a rank comparison against a perfect separator.
2. **Exploratory checks (post hoc; they cannot change the decision):**
   * At task level (n = 12), the relationship to rescue rate survives controlling for ‖A_f‖ for translation
     ALIGN (partial ρ 0.57), faithful separation/N_f (0.56) and P_cov (0.48). It does not survive for SCNR
     (0.10) or full ALIGN (0.27).
   * In the three scene-matched task pairs (same scene, training instruction and instruction set), the
     zero-rescue member is lower on ALIGN, P_cov, SCNR and leverage in 3/3. It is also lower on ‖A_f‖ in 3/3.
3. **Task 7 is a counter-example to any call-0 geometry mechanism.** It is rescued in 40 % of B failures despite
   the most misaligned CAG direction (ALIGN −0.11; exec prefix −0.58), negative P_cov and the lowest SCNR (1.32).
4. **Call-0 geometry is a single snapshot.** The faithful-vs-biased contrast may develop later in the trajectory.
   This phase deliberately looked only at the pre-treatment call.

## Implications

* **Do not run the S2c-1 ω response surface.** Under the gate it is NO-GO. Descriptively it is also poorly
  motivated: where CAG fails, its direction is nearly orthogonal to the semantic contrast (ALIGN ≈ 0.1,
  P_cov ≈ 0.06). Raising ω mostly lengthens a vector that does not point away from the biased behaviour. That
  is a geometric inference, not a measurement; no other ω was run.
* **Do not consume states 30–39** for the semantic-controllability hypothesis.
* The most informative within-task signal found so far is **faithful separation relative to noise**, with
  within-task AUROC 0.71 for rescue and 0.75 for semantic correction. It is still below the task-ID baseline
  (0.87–0.91), and it is a predictor, not a mechanism.
* Not a decision, and not permitted in this phase: if the programme continues on semantic steering, the
  descriptive pattern points at the **reference** CAG uses (language-masked), not at its strength. A different
  reference, for example the training instruction itself, would be a new intervention with its own novelty
  check (contrastive / negative-prompt guidance is prior art) and its own preregistration. It would consume
  states 30–39.

---

S2c-0 stops here. No states 30–49 executed, ω not tuned, no method implemented.
