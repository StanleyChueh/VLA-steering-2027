# S2b RNG validation: exact common-random-number control for B vs CAG-TF

Date: 2026-10-05. Written from tests run **before** any S2b rollout (initial states 20–29). The tests used only
already-used states (state 0 of tasks 0, 6 and 12) and excluded task 13. Preregistration:
`configs/cag/S2b_prereg.yaml`. Its SHA-256, and the hashes of all run code, were recorded in
`results/S2b/PREREG_FROZEN.txt` at 2026-10-05T10:45:51+08:00, before launch.

**Verdict: exact CRN is implementable without changing policy semantics. All checks PASS**, with one required
change. B and S run in different server processes, so XLA autotuning must be turned off. Otherwise identical noise
gives actions that differ by up to 5.9e-3 between processes (details below).

## Why exact CRN is possible

* The only stochastic quantity in `Pi0.sample_actions` (openpi-cf `pi0.py` L224–288) is the flow-matching noise. If
  `noise is None` it is drawn as `jax.random.normal(rng, (1, 10, 32))`. The 10-step Euler ODE after it is
  deterministic, and `preprocess_observation(None, …, train=False)` uses no randomness.
* Both public entry points take the noise tensor directly: `Policy.infer(obs, noise=…)` (B) and LIBERO-CF
  `CounterfactualActionGuidedPolicy.infer(obs, noise=…)` (S). CAG passes the **same** tensor to its conditioned and
  language-masked passes. This is the public CAG stochastic process, and CRN keeps it unchanged.
* So CRN needs no policy-code change. `src/wrappers/crn.py` derives the noise and passes it through `noise=`:

  ```
  key   = fold_in(fold_in(fold_in(fold_in(jax.random.key(0x53324220), master_seed), task_id), init_id), call_index)
  noise = jax.random.normal(key, (1, 10, 32))     # float32, same sampler/shape/dtype as the native draw
  ```
  The client sends `(master_seed, task_id, init_id, call_index)` with every request (`call_index` = policy call
  within the episode). `CRNPolicy` is the outermost server wrapper and is strict:
  * a request without a tuple is refused;
  * a master seed different from the server's is refused;
  * caller-supplied noise is refused.

  It returns the tuple and the noise SHA-256 prefix with every response. The client logs these per call, together
  with a hash of the observation (both images + state). The derivation is stateless, so an episode's length cannot
  shift any later episode's noise.

## Requirement → evidence

| Requirement | Test | Result |
|---|---|---|
| 1. B and S conditioned branches get identical noise for the same tuple | P3 (stock path, 9 obs×call tuples): S cond/uncond noise == CRN noise == B noise, bitwise | **PASS 9/9** |
| | B sampler output == S conditioned-branch sampler output, bitwise; B env action == output_transform(S cond), bitwise | **PASS 9/9** |
| 2. Native marginal noise distribution preserved | N3: 416,000 CRN values vs the same count of native openpi draws (`split(_rng)` → `normal`) | mean −0.0016 vs +0.0020; std 1.0010 vs 0.9991; skew −0.009 vs −0.003; excess kurtosis 0.004 vs −0.005; **KS two-sample p = 0.53**; KS vs N(0,1) p = 0.54 |
| | Independence: correlation of noise at neighbouring call indices (50 pairs) / master seed 0 vs 1 (13 pairs) | mean −0.012 / +0.031; max \|r\| 0.155 / 0.127 (sampling SD for 320 values: 0.056) |
| 3. Public CAG cond/uncond relationship preserved | P3: one shared tensor for both passes; guided = uncond + 1.5(cond − uncond) | **PASS**, reconstruction error ≤ 1.4e-7 |
| 4. Divergent episode lengths cannot shift later episodes | N4: episode (task 5, init 21) noise stream after a 3-call vs a 40-call previous episode | **identical** (stateless by construction) |
| | P4: perturbing the policy's own `_rng` stream once noise is supplied | B and S actions **bit-identical 9/9** (`_rng` still advances but is unused) |
| 5. Noise fingerprint per call | `crn.noise_sha` in every response; `diag.branches.*.noise_sha` must equal it | logged; P5 equality **18/18** |
| Deterministic repeat → bit-identical actions | P1: same process, same (obs, tuple) twice | **PASS 9/9** (B and S) |
| | Cross-process, deterministic XLA (see below): fresh process on GPU 0 and on GPU 1 vs a reference process | **36/36 bit-identical** on each GPU |
| Changing master_seed changes the noise | N2: changing any key field gives a different tensor with 0 shared elements; P2: seed 0 vs 1 actions differ | **PASS** (min max\|ΔB\| = 0.049) |
| Instrumentation changes no action | P5: S2b server stack (`pi05_attention.install()` + `DiagnosticPolicy` + `CRNPolicy`, tuple sent via `obs["crn"]`) vs stock path with no instrumentation installed, same tuples | **18/18 bit-identical** (vanilla and CAG-TF); diagnostic re-runs bit-identical 18/18; logged `a_conditioned` == stock B action bitwise 9/9 |

Files:
* `scripts/audit/s2b_crn_validation.py` → `results/S2b_rng_validation/det_full_gpu1/crn_validation.json` (full run)
  and `det_ref_gpu0/`, `det_rep_gpu{0,1}/` (cross-process).
* The autotuned runs that exposed the problem are kept: `gpu0_full/`, `gpu{0,1}_stock_repeat/`.

## Finding: XLA autotuning breaks cross-process bit-identity

Under the S2 pilot's default XLA settings, two processes given identical observations and identical noise produced
**0/36 bit-identical** action chunks:
* max |Δ| = 5.8e-3 env units, median 2.9e-3;
* the noise tensors were identical (same fingerprints);
* within a process, repeats were bit-identical.

The cause is per-process kernel autotuning. bf16 differences are then amplified over the 10 Euler steps. A 3e-3
difference is about 1 % of a typical translation command, and it would have broken the "B call 0 == S conditioned
call 0" identity, because B and S run in separate servers.

Fix: every CRN server runs with `XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"`.
* `run_b_vs_s.sh` exports the flags whenever `CRN_SEED` is set.
* `serve_cf_policy.py` refuses to start with `--crn-master-seed` if they are missing.
* The flags are recorded in the server metadata.

Result: 36/36 bit-identical across processes and across both GPUs.

This changes kernel selection only, not the model or the sampler. Deterministic-mode actions differ from autotuned
actions by the same ULP-amplified amount (max 5.9e-3, median 3.5e-3). Cost: policy latency rises from about 54 to
about 107 ms for B and from about 133 to about 219 ms for S, with 3 servers sharing one GPU.

## End-to-end check over the real server/client stack (excluded task 13, initial state 0)

Two B episodes and one S episode, CRN seed 0, all three servers concurrently on GPU 0
(`results/S2b_rng_validation/e2e/`, `src/analysis/s2b_pair_checks.py`):

| Check | Result |
|---|---|
| B run twice: executed actions over the whole episode (116 steps, 24 calls) | **bit-identical**; observation-hash sequence identical; outcome identical |
| B vs S: geometry and initial sim state | identical |
| B vs S: call-0 observation hash | identical |
| B call-0 action chunk vs S call-0 conditioned (pre-CAG) chunk | **bit-identical** |
| B vs S noise fingerprint at every common call index (24) | identical |
| First call index where B and S observations differ | 1 (expected: S executes the CAG chunk from call 0) |
| CRN tuple correct on every call; diagnostic noise == CRN noise; diagnostic re-runs bit-identical | yes / yes / yes |

The guard `--forbid-init-from 30` was also exercised. A request for states 25–34 was refused before any
environment was built.

## What CRN does and does not control in S2b

* At every call index c, B and S use the same noise. At call 0 they also see the same observation, so S's
  conditioned chunk equals B's chunk exactly. All of S's call-0 difference is therefore the CAG residual.
* From call 1 on, the observations differ because S executed a different chunk. The noise is still matched by call
  index, which is the standard CRN coupling for sequential decisions. Outcome differences are then the effect of
  CAG plus the chaotic amplification of that perturbation through the closed loop. Two master seeds, which share
  the same scenes because env seed 7 and reset order are fixed, are used to separate robust effects from such
  stochastic interactions (`S2B_CAUSAL_STEERING_EFFECT.md`).
* The environment is deterministic given actions: B replicated bit-for-bit end-to-end.
