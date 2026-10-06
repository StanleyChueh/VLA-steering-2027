# S2d preregistration: oracle counterfactual-reference test

Status: **frozen before any S2d rollout.** SHA-256 of this file, the reference-prompt manifest and the run code are
in `results/S2d/PREREG_FROZEN.txt`. Label: **ORACLE DIAGNOSTIC**, not a method.

Constraints:
* The source instruction is known only because LIBERO-CF is built from LIBERO-Spatial training scenes.
* Nothing is trained, ω = 1.5 is not tuned, the VLA is not modified, and no selector is built.
* States 30–49 are not used.

Written after S2b and S2c-0 outcomes were known. The task groups come from S2b and are not redefined.

## Question

Does counterfactual steering fail in the S2b zero-rescue tasks because its fixed language-masked reference does not
represent the actual competing (source/training) behaviour?

## Data and fixed policy

| item | value |
|---|---|
| tasks / states / seeds | CF-Spatial 0–12, initial states 20–29, CRN master seeds 0 and 1 (260 pairs per condition) |
| policy | frozen `pi05_libero`; H = 10, h = 5; 220 max steps; env seed 7; S2b task/state/reset order |
| RNG | S2b CRN: noise = f(master_seed, task, init, call index), one tensor shared by every pass of a call; deterministic XLA |
| B, MASK | **reused from S2b** (`results/S2b/seed{0,1}/{B,S}`). Justified by an exact replay under the current code: 4 S2b episodes (B and MASK, task 0, states 20–21, seed 0) reproduced bit-for-bit over the whole trajectory |
| new conditions | SOURCE-NORM (SN), OTHER (OT), SOURCE-RAW (SR): 3 × 260 = 780 new episodes |

## Conditions

All chunks are pre-guidance, in model quantile space; norms are Frobenius over the 10 × 7 action dims, ε = 1e-6.
* A_f: faithful prompt.
* A_u: CAG's language-masked pass.
* A_b: oracle source/training instruction.
* A_o: frozen non-source alternative.

Every pass of a call uses the same CRN noise.

| condition | action | role |
|---|---|---|
| B | A_f | S2b |
| MASK | A_f + 0.5 (A_f − A_u) (CAG-TF, ω = 1.5) | S2b |
| **SOURCE-NORM** | A_f + s_mask · (A_f − A_b) / (‖A_f − A_b‖ + ε), with s_mask = ‖0.5 (A_f − A_u)‖ | primary |
| **OTHER** | A_f + s_mask · (A_f − A_o) / (‖A_f − A_o‖ + ε) | specificity control |
| SOURCE-RAW | A_f + 0.5 (A_f − A_b) | secondary (norm can differ) |

* SN and OT have exactly MASK's intervention norm at every call: verified error ≤ 6.6e-7.
* Only the reference direction differs from MASK.
* Implementation: `src/wrappers/reference_guidance.py`. It mirrors LIBERO-CF CAG's code path for the faithful and
  masked passes. Server modes are `ref_norm` / `ref_raw`, and the client sends the frozen reference prompt per task.

**Reference prompts** (`configs/cag/S2d_reference_prompts.json`, from `scripts/audit/s2d_reference_prompts.py`).
Benchmark strings only.
* source = the instruction of the identical LIBERO-Spatial training scene (S2c-0 l_b).
* other = the lexicographically first remaining valid benchmark instruction (neither l_f nor l_b).

| task | group | source (oracle) | other |
|---|---|---|---|
| 1 | **G0** | bowl between the plate and the ramekin | bowl next to the ramekin |
| 4 | **G0** | bowl on the cookie box | bowl on the wooden cabinet |
| 5 | **G0** | bowl in the top drawer of the wooden cabinet | bowl on the wooden cabinet |
| 10 | **G0** | bowl on the wooden cabinet | bowl on the stove |
| 11 | **G0** | bowl from table center | cookie box |
| 0, 2, 3, 6, 7, 8, 9, 12 | control | see manifest | see manifest |

## Validation before rollout (`scripts/audit/s2d_reference_validation.py`; `results/S2d_validation/`)

| check | result |
|---|---|
| deterministic repetition (fresh policy objects) | 24/24 bit-identical |
| identical CRN noise in every pass (fingerprint) | 24/24 |
| faithful pass == `Policy.infer(l_f)` == S2b B call-0 chunk | 24/24 bit-identical |
| masked pass == CAG's masked pass; reference pass == `Policy.infer(l_r)` | 16/16; 24/24 bit-identical |
| SN/OT intervention norm == MASK's ‖0.5 (A_f − A_u)‖ | max error 6.6e-7 |
| returned action == output_transform(formula) | max error 2.0e-7 |
| full server stack (CRN → provenance → reference policy) == direct | 24/24 bit-identical |
| cross-process (GPU 1 vs GPU 0) | 24/24 bit-identical |
| S2b B/MASK replay under current code | 4/4 episodes bit-identical whole trajectory |
| noise marginal | unchanged: same `src/wrappers/crn.py` (committed `40b263f`; S2b KS p = 0.53 vs native) |

In-run checks for every pair:
* static geometry and initial sim state equal to S2b;
* call-0 observation hash equal to S2b;
* SN/OT/SR call-0 faithful chunk == S2b B call-0 chunk, bitwise;
* CRN tuple and noise fingerprint at every call;
* no exceptions.

Invalid pairs are excluded from all analyses and listed.

## Outcomes (S2b definitions; S2b labels reused for B and MASK)

* **Primary:**
  * faithful first-object contact (grounding class = faithful-first);
  * faithful success;
  * biased first-object contact;
  * biased success.
* **Secondary:**
  * episode length;
  * timeout (220 steps with no listed condition met);
  * post-contact failure (faithful touch without faithful success);
  * policy calls;
  * server inference latency;
  * action clipping: fraction of executed components in dims 0–5 with |a| > 1;
  * trajectory roughness: mean ‖a_{t+1} − a_t‖ over executed actions.
* Rescue and harm are reported relative to B and to MASK.

## Groups (from S2b, not redefined)

* **G0 (primary diagnostic) = tasks 1, 4, 5, 10, 11:** 100 pairs. In S2b this was 98 B failures and 0 MASK rescues.
* **Control = tasks 0, 2, 3, 6, 7, 8, 9, 12:** 160 pairs.

## Statistics

* **Per seed:** exact McNemar on CRN pairs.
* **Pooled:** state-cluster sign-flip permutation test (10,000; both seeds of a state flipped together) on the sum
  of paired differences.
* **CIs:** state-cluster bootstrap 95 % CI (2,000) of the paired rate difference.
* **Breakdowns:** per-task net effects; seed consistency (states with a same-sign change in both seeds).
* The task structure is never pooled away: G0 and control are always reported separately, and per task.

## Hard gate

Primary outcomes for A–C: faithful success **or** faithful first-contact. Bonferroni over the two: p < 0.025.

| criterion | operational rule |
|---|---|
| **A** SN improves over MASK in G0 | for at least one of the two outcomes: paired difference SN − MASK > 0 in G0, state-cluster permutation p < 0.025, and bootstrap CI lower bound > 0 |
| **B** not concentrated in one task | for an outcome passing A: net gain (SN − MASK) ≥ +2 pairs (of 20) in ≥ 2 of the 5 G0 tasks |
| **C** source is privileged | same outcome, G0: SN − OT > 0 with permutation p < 0.05 and bootstrap CI lower bound > 0 |
| **D** not norm / clipping / extra calls | (i) per-call SN intervention norm == s_mask (logged; median \|error\| < 1e-4), and the median per-call SN intervention norm in G0 within ±10 % of MASK's (S2b); (ii) G0 clipping rate SN − MASK ≤ +1 percentage point; (iii) the replanning schedule is unchanged by construction (h = 5); calls per episode reported |
| **E** no large regression on controls | control tasks, faithful success: it is NOT the case that SN − MASK ≤ −5 pp with permutation p < 0.05 |

Decision, exactly one:
* **A. PASS:** A ∧ B ∧ C ∧ D ∧ E.
* **B. PARTIAL:** A holds and at least one of B–E fails.
* **C. FAIL:** A fails. Stop semantic counterfactual steering and pivot.

No post-hoc search for better reference formulas. SOURCE-RAW and every analysis beyond the table are secondary and
cannot change the decision.

## Novelty context (for the decision report only)

The phase brief lists CAG, Stable Language Guidance / RSS, CofactVLA, ReSteer, Mostly Harmless VLA Steering,
CoVer-VLA, ReGuide and "When Instructions Retrieve Trajectories" / ECT. **None of these was re-read in this
phase.** If the result is PASS, a literature comparison is a precondition for designing any automatic
reference/source-mode method. The candidate contribution would have to be training-free identification of the
retrieved competing mode plus reference-aware correction. It must not be another masked branch, larger guidance,
projection, rephrasing, uncertainty gating or geometric target guidance.
