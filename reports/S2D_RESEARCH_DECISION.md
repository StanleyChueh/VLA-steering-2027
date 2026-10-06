# S2d research decision: oracle counterfactual reference

Date: 2026-10-05. Label: **ORACLE DIAGNOSTIC.**
* Preregistration `S2D_ORACLE_REFERENCE_PREREG.md`, frozen before rollout (`results/S2d/PREREG_FROZEN.txt`).
* States 30–49 not used. ω not tuned. Nothing trained. VLA not modified. No selector built.

Details:
* `S2D_ORACLE_REFERENCE_RESULTS.md`
* `S2D_REFERENCE_SPECIFICITY.md`

## Decision: **C. FAIL — the oracle source reference does not solve the zero-rescue regime. Stop semantic counterfactual steering and pivot.**

## Evidence

* **Validity.**
  * 780 new episodes (SOURCE-NORM, OTHER, SOURCE-RAW × 2 seeds × 13 tasks × states 20–29), all valid.
  * S2b B/MASK reused after an exact whole-trajectory replay (4/4 bit-identical).
  * In every S2d episode, the call-0 faithful chunk is bit-identical to S2b's B chunk, and the noise is identical
    to S2b's at every common call.
  * SN/OT match MASK's intervention norm at every call (median error 4.8e-7).
* **Gate A fails.** In the S2b zero-rescue tasks (1, 4, 5, 10, 11; 100 pairs), SOURCE-NORM vs MASK:
  * faithful success 6 % vs 2 % (4/0, p = 0.125);
  * faithful first-contact 8 % vs 2 % (6/0, p = 0.126).

  The bootstrap CIs exclude 0, but neither outcome reaches the preregistered p < 0.025. The absolute effect is 4–6
  pairs out of 100. Task 1 never moves under any condition.
* **Gate C fails.** The norm-matched OTHER reference gives the same faithful outcomes: 7 % / 7 %; SN − OT −0.01 /
  +0.01, p = 1.0. The true source behaviour is not a privileged reference for producing the instructed behaviour.
* D (norm, clipping, calls) and E (no significant control regression; −5.6 pp, p = 0.24) hold, but they cannot
  rescue A and C.

## What S2d does establish

1. **A source-specific suppression effect.**
   * Steering away from the actual source trajectory lowers biased success in G0 from 76 % (MASK) to 59 %
     (p = 0.007). OTHER stays at 75 % (SN vs OT p = 0.001).
   * The released behaviour mostly becomes timeouts (16 % → 29 %), not instructed-object behaviour.
   * Knowing the competing mode lets steering suppress it, but there is nothing for the policy to fall back to.
2. **Magnitude is not the missing ingredient.** The unnormalised SOURCE-RAW (26 % larger median push) is not
   reliably better than SOURCE-NORM in G0, and it is the worst arm for control-task execution:
   * faithful success 0.51 vs MASK 0.63, p = 0.026;
   * post-contact failure 0.31.

   This agrees with S2c-0: the zero-rescue regime is not amplitude-limited.
3. **On the control tasks, a better reference trades grounding for execution.** Source-referenced guidance reaches
   the faithful object a little more often than MASK (+6.9 pp, p = 0.079). It then fails after contact twice as
   often (25 % vs 12 %), so task success is not better. This is the S2b execution-interference pattern, made
   larger by the source reference.

Combined reading across S2b → S2c-0 → S2d:
* The masked-language reference is descriptively misaligned with the faithful-vs-source contrast (S2c-0).
* Replacing it with the oracle source reference does not unlock the zero-rescue tasks; it only suppresses the
  retrieved trajectory (S2d).
* In the zero-rescue tasks, the bottleneck is therefore not *which reference* action-space steering contrasts
  against. Under these prompts, the instructed-object behaviour is not available to be amplified at action level.
  That is an interpretation consistent with an instruction-action binding failure upstream of the action expert;
  it is not tested here.

## What this does NOT claim

* It does not show reference choice is irrelevant everywhere. On controls it moves grounding (and execution) by a
  few points.
* With 100 G0 pairs, an oracle effect of about 4 pp is not distinguishable from 0 at the preregistered level. The
  gate asked whether the oracle *unlocks* the regime, and 6 % faithful success does not.
* No automatic method was designed or tested. Per the preregistration, there is no post-hoc search for better
  reference formulas.

## Recommendation

* **Stop semantic counterfactual steering** (masked, source-referenced or alternative-referenced action
  contrasts) as the programme's main direction.
* Do not build the automatic reference/source-mode selector: its oracle upper bound failed.
* Keep states 30–39 and 40–49 unconsumed.
* What the evidence points toward, for the next decision rather than for this phase: the zero-rescue failures look
  like a **binding / retrieval** problem, where the instruction retrieves the trained trajectory family and the
  instructed object has no accessible action mode. Candidate pivots should act before or outside the action
  expert, not by contrasting action chunks. Examples:
  * representation-level analysis of how the instruction tokens bind to objects;
  * external generate-and-verify approaches (the earlier VERITAS/VLS fallback).

  Each would need its own novelty review and preregistration.
* Novelty context: the brief lists CAG, Stable Language Guidance / RSS, CofactVLA, ReSteer, Mostly Harmless VLA
  Steering, CoVer-VLA, ReGuide and "When Instructions Retrieve Trajectories" / ECT. **None was re-read in this
  phase.** The comparison was required only before designing an automatic method, which a FAIL rules out.
  "When Instructions Retrieve Trajectories" is the most relevant to the pivot above and should be read first.

---

S2d stops here. No states 30–49 executed, ω not tuned, no selector built, no method claimed.
