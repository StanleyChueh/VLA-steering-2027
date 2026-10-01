# Next-experiment decision (end of S0 + smoke tests)

Date: 2026-10-01. All numbers below are **SMOKE** (n=2 per condition) and carry no inferential weight.
Details: `S0_PROVENANCE_AND_FEASIBILITY.md`, `S1_SCALE_SMOKE.md`, `S2_CAG_TF_SMOKE.md`, `S3_ATTENTION_FEASIBILITY.md`.

## Go / no-go

**SCALE: GO**
OpenVLA+LIBERO path is COMPLETE and runs as released at paper-default hyperparameters (bf16, one 4090).
Report two code↔paper discrepancies alongside any result: the undocumented `attn_sensitivity=0.3` inside γ's tanh,
and the ε parameterisation of q_low. Other backbones are not public (paper: PARTIAL).

**CAG-TF: CONDITIONAL GO**
π0.5 + CAG-TF is RECONSTRUCTIBLE: the method path is complete and runs with the public checkpoint. Conditions:
1. Decide ω before the pilot: paper App. B gives 1.5 for π0/π0.5 TF, the README command gives 2.0. Smoke used 1.5.
2. Faithful/Biased metrics rely on our reconstructed labels (`src/metrics/libero_cf_labels.py`, tested). LIBERO-CF
   does not ship them.
3. Pilot task 14 has a prompt ≠ BDDL `:language` inconsistency (labels follow the prompt the policy receives).
   It stays in the preregistered set but must be reported separately as well as pooled.

**π0.5 action-expert attention instrumentation: GO**
Capture-ON actions are bit-identical to the stock policy (max\|Δ\| = 0), with no latency cost.
Caveat for S3/S4: as written, the KWTS plateau threshold (η=0.95 of ln 968) is **unreachable** on `pi05_libero`
because masked tokens cap entropy at ≈0.91 ln 968.

## Smoke results

**SCALE smoke** (LIBERO-10 task 0, init states 0–1, seed 7):

| | success |
|---|---|
| greedy (baseline) | **1 / 2** |
| SCALE | **2 / 2** |

**CAG smoke** (CF-Spatial task 0, init states 0–1, ω=1.5), faithful success (all faithful touch 2/2, all biased 0/2):

| | faithful success |
|---|---|
| baseline (vanilla π0.5) | **2 / 2** |
| CAG-TF | **1 / 2** |

## Peak VRAM

| | memory | notes |
|---|---|---|
| SCALE | **14.41 GiB** torch (greedy 14.42 GiB) | TF pre-allocates the rest of the GPU unless `TF_FORCE_GPU_ALLOW_GROWTH=true` |
| π0.5 | **6.68 GiB** vanilla, **6.84 GiB** CAG-TF | JAX peak, `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| π0.5 attention capture | +28 MB/query | |

## Critical reproducibility issues

1. **CAG ω ambiguity** (README 2.0 vs paper 1.5). The OpenVLA-OFT ω also differs (README 2.0 vs paper 3.0).
2. **LIBERO-CF does not encode Faithful vs Biased**; all CF-Spatial tasks list the same 4 conditions. The paper's
   headline metrics cannot be recomputed from the released code without a reconstruction like ours.
3. **LIBERO-CF data inconsistencies**: tasks 13 and 14 have BDDL `:language` that disagrees with the
   filename-derived prompt actually sent. For task 13 the BDDL language would make the faithful object equal to the
   biased one.
4. `set_custom_conditions` is commented out upstream and its call is silently swallowed (harmless, but it shows the
   harness was not run as written). Episodes terminate on **any** condition (OR), not the BDDL goal.
5. The bundled `openpi-cf` is **not installable as shipped** (missing README.md) and has no git history, so its
   upstream openpi base commit is unknown.
6. CAG-TF guidance is applied to **final action chunks after two full ODE solves**, not per-step velocity CFG. The
   paper does not specify this. Shared noise between branches makes it a deterministic action-space extrapolation.
7. **Policy RNG = one stream per server process.** Results depend on server restarts and episode order. Vanilla and
   CAG draw identical noise at equal call index (verified), but pairing breaks after the first episode whose call
   count differs. A per-episode reset keyed on (master_seed, task, init_state) would give exact pairing with
   unchanged marginals (relevant to S5; not used yet, to preserve native semantics).
8. **SCALE: undocumented `attn_sensitivity=0.3`**; q_low ε parameterisation differs from the paper; LIBERO unpinned
   in README; paper's 3 seeds unlisted; the source of greedy's reported variance is unexplained.
9. **OpenVLA/SCALE silently fail without CUDA libdevice for TF** (every episode becomes an "exception → failure").
   Any OpenVLA-based number should be checked for zero exception episodes.
10. **KWTS definition vs public π0.5 pipeline**: the η=0.95·ln 968 plateau threshold is unreachable (see above).
    The paper's LIBERO gain over Fixed-10 cannot come from the rule as written.
11. π0.5-LIBERO receives **no proprioceptive state** (`discrete_state_input=False`). This matters when attributing
    "action-side" failures.

## Recommended next experiment

Run the two preregistered pilots that establish A and B, in parallel on the two GPUs (≈1–1.5 h each):

* **S2 pilot (primary, GPU 0):** B vs S (CAG-TF, ω=1.5) on `libero_cf_spatial` tasks 0, 7, 14 × 20 init states
  (0–19), fresh server per condition. Outcomes: faithful/biased touch/success, paired rescue/harm.
* **S1 minimal (GPU 1):** greedy vs SCALE on `libero_10` tasks 0, 4, 9 × 10 init states, seed 7.

Before launching S2, two one-line decisions are yours:
(a) confirm ω=1.5 (paper) vs 2.0 (README). Do not pick it from pilot outcomes.
(b) whether to add per-call guidance-magnitude logging (‖a_cond − a_uncond‖) now. It is a wrapper-only change and
    is needed later for S5 outcome 11.

S3 (entropy diagnostics) should follow the S2 pilot on the same episodes, with the diagnostic sampler as control
path. Its definitional issue (KWTS N/masking) must be settled in the S3 preregistration first.

## Exact command to run next

```bash
cd ~/Stanley_ws/VLA-steering-2027
# S2 pilot (GPU 0): 3 tasks x 20 episodes x {B, S}
TASKS=0,7,14 EPISODES=20 PORT=8765 bash scripts/run_cag/run_b_vs_s.sh 0 1.5 results/S2_pilot
python3 src/analysis/summarize_cf.py results/S2_pilot/B results/S2_pilot/S

# S1 minimal (GPU 1, can run concurrently): 3 tasks x 10 episodes x {greedy, SCALE}
bash scripts/run_scale/run_greedy_vs_scale.sh 1 libero_10 0,4,9 10 results/S1_minimal
```

Not launched. Per instructions, this phase stops here.
