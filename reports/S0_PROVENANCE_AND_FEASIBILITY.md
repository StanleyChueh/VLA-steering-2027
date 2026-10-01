# S0 — Provenance, Reproducibility Audit, and Feasibility

Date: 2026-10-01 · Workspace: `~/Stanley_ws/VLA-steering-2027` · Status: **AUDIT (no experimental claims)**

All external code is under `external/` (gitignored, read-only; verified `git status --porcelain` empty for every
repo after all installs). Everything we wrote lives in `src/`, `scripts/`, `configs/`, `tests/`, `patches/`.
Environments live in `envs/` (gitignored) and are rebuilt by `scripts/audit/setup_envs.sh`.

---

## 1. Hardware gate

| Item | Value |
|---|---|
| GPUs | 2× RTX 4090 24 GB (GPU0 idle; GPU1 drives the display, ~0.45–0.6 GB used) |
| Driver / CUDA (driver) | 580.178.04 / 13.0 |
| CPU / RAM | i7-13700 (24 threads) / 62 GB (≈30 GB available while desktop apps run) |
| Disk | 3.2 TB free on `/` |
| Network (observed) | very variable: 0.05–8 MB/s; GitHub clones stalled on HTTP/2 resets (retried over HTTP/1.1) |

Single-GPU fit (measured in this phase, not estimated):

| Model | Precision | Peak GPU memory | Fits one 4090? |
|---|---|---|---|
| π0.5 `pi05_libero` (vanilla) | bf16 (openpi default) | **6.68 GiB** (JAX `peak_bytes_in_use`, smoke run) | yes |
| π0.5 + CAG-TF (2 passes, same weights) | bf16 | **6.84 GiB** | yes |
| π0.5 + attention capture (diagnostic) | bf16 | +28 MB per query for full probs (10 steps × 18 layers × 8 heads × 10 × 978, bf16) | yes |
| OpenVLA-7B greedy / SCALE | bf16 + FlashAttention-2 | **14.42 / 14.41 GiB** (torch `max_memory_allocated`); TF 2.15 additionally pre-allocates most remaining memory (nvidia-smi 23.0 GB) | yes, but needs a dedicated GPU unless `TF_FORCE_GPU_ALLOW_GROWTH=true` |

No quantization was used anywhere. The RTX PRO 6000 was not needed.

---

## 2. Provenance

### 2.1 External repositories (cloned 2026-10-01)

| Path | Remote | Branch | HEAD SHA | Latest commit | Commits | Tags | Submodules |
|---|---|---|---|---|---|---|---|
| `external/scale` | github.com/snumprlab/scale | master | `b4ad2a69d14f91712704711e810cf9830e2b7121` | 2026-06-29 "Update README.md" (Hyeonbeom Choi) | 22 | none | none |
| `external/libero-cf` | github.com/yuffish/LIBERO-CF | main | `8460457bfca6e0ef2e856bc104e2c60b023ef2a7` | 2026-04-23 "first commit" (fangyu) | **1** | none | none |
| `external/mechanistic-steering-vlas` | github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas | main | `559c0f25a3cc5a20fc8b804774a88415404e0d22` | 2026-01-30 "Update: README" | 5 | none | none |
| `external/vla-explain` | github.com/robot-future/vla-explain | main | `202d2a9a00fb4b99083559525d54d8f2a7eb4d3f` | 2026-06-10 "version 2.3 update author profiles and code links" | 24 | v2.2, v2.3 | none |
| `external/deps/LIBERO` (dependency of SCALE) | github.com/Lifelong-Robot-Learning/LIBERO | master | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | 2025-03-15 "Add support for dataset download from huggingface" | – | – | – |

Paper PDFs used for the audit: `external/papers/{2602.04208 (SCALE), 2602.17659 (LIBERO-CF/CAG), 2609.00908 (KWTS)}.pdf`.

### 2.2 Checkpoints

| Use | Checkpoint | Pin / verification | Size |
|---|---|---|---|
| SCALE, `libero_10` | HF `openvla/openvla-7b-finetuned-libero-10` | revision `80970322773f81baa2e22fe495d0487b93a05cfa` (HF main at audit time) | 15 GB |
| π0.5 vanilla + CAG-TF | `gs://openpi-assets/checkpoints/pi05_libero` | 16/16 files CRC32C-verified vs GCS (10/10 also MD5); newest object 2025-09-08 | 12.44 GB |
| (other SCALE suites, not downloaded) | `openvla/openvla-7b-finetuned-libero-{spatial,object,goal}` | – | ~15 GB each |
| (CAG-VA, not downloaded, not in scope) | HF `yuffish/pi05_libero_unconditioned` | – | – |

Model references in the secondary repos (not installed): mechanistic-steering-vlas → `openvla/openvla-7b-finetuned-libero-10`,
openpi `pi0_fast*`, `pi0_libero`; vla-explain → HF `HanxinZhang/pi05-xicm` (π0.5 checkpoint for their ISS/NMR@k demo).

### 2.3 Environments (all isolated; nothing installed into `external/`)

| env | Python | Key versions | Deviations from upstream instructions |
|---|---|---|---|
| `envs/openpi-cf` | 3.11 | jax 0.5.3 (cuda12), flax 0.10.2, orbax 0.11.13, transformers 4.53.2 (from LIBERO-CF `uv.lock`, `--frozen`) | **openpi-cf cannot be built as shipped**: its `pyproject.toml` references a `README.md` that LIBERO-CF does not include. Installed the locked deps with `--no-install-project` and exposed `openpi-cf/src` through a `.pth` file. |
| `envs/libero-cf-client` | 3.8 | robosuite 1.4.1, mujoco 3.2.3, numpy 1.22.4, torch 1.11.0 | torch installed as `+cpu` instead of `+cu113`: the client only calls `torch.load` on init-state files. LIBERO-CF itself exposed via `.pth`, not `pip install -e .` (which would write into `external/`). |
| `envs/scale` | 3.10 | torch 2.2.0 (cu121), transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, flash-attn 2.5.5 (official prebuilt wheel), tensorflow 2.15.0, numpy 1.26.4, mujoco 3.3.2, robosuite 1.4.1 | pip/uv instead of conda for torch (same pinned version). LIBERO pinned to `8f1084e3` (SCALE README clones unpinned HEAD). |

LIBERO paths come from project-local configs (`configs/libero/{upstream,cf}/config.yaml` via `LIBERO_CONFIG_PATH`),
so `~/.libero` and the COAST LIBERO config are never touched.

---

## 3. S0A — SCALE audit

**Paper ↔ repo mapping.** Paper: arXiv 2602.04208 (ICML 2026). The repo implements **only the OpenVLA + LIBERO
instantiation** (README: "This repository provides the OpenVLA on LIBERO instantiation"). π0-FAST, SpatialVLA,
SIMPLER-WidowX, LIBERO-PRO and the real-robot experiments are not released.

| Question | Answer (with source) |
|---|---|
| Checkpoint per suite | `openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}`, auto-selected from `--task_suite` (`run_libero_eval.py:163-165`, `run_libero_eval.sh:23-29`) |
| Precision | bf16, `attn_implementation="flash_attention_2"` (`openvla_utils.py:56-64`); 8-/4-bit flags exist but default False |
| LIBERO version | upstream LIBERO HEAD, **unpinned** in README; we pinned `8f1084e3` |
| Seeds | code default `seed=7` (`run_libero_eval.py:123`), `set_seed_everywhere` (`robot_utils.py:247-255`), env `env.seed(0)` (`libero_utils.py:376`). Paper reports mean ± std over **three seeds** (App. K, Table 12) but does not list them. |
| Trials / task | 50 (`run_libero_eval.py:108`); max steps 220/280/300/520 (spatial/object/goal/10) (`:224-230`); 10 wait steps |
| Decoding modes | `greedy`, `temp`, `topk`, `topp`, `scale` (`modeling_prismatic.py:633-841`) |
| Hyperparameters (`configs/scale.yaml`) | T0=1.0, ε=1e-12, num_logits=256, κ=2.0, α=0.8, **attn_sensitivity=0.3**; "tuned on LIBERO-Long, applied unchanged" |

**Equations as implemented** (`external/scale/prismatic/extern/hf/modeling_prismatic.py`):

* Self-uncertainty (`_compute_self_uncertainty`, L598-631): mask to the 256 action-bin logits (ids 31744–31999),
  softmax → p; `q_low = [1-(K-1)ε, ε, …]` on top-1, `q_high = 1/K`; `u_k = Σ p log(q_high/q_low)`.
  With K=256, ε=1e-12: `u_k = p_max·(−5.545) + (1−p_max)·22.08` ∈ [−5.55, 22.1]; u=0 at p_max≈0.80.
* Adaptive action decoding (L708-748): `τ_k = T0·σ(u_k)`; sample `softmax(ℓ_top/τ_k)` over the 256 action tokens
  (`torch.multinomial`, global torch RNG). Greedy fallback only if τ<1e-3, unreachable for T0=1 (τ_min=σ(−5.55)=0.0039).
* Step uncertainty and EMA (L688-697, L750-758): per-dimension EMA lists, combined by mean, which is equivalent by
  linearity to the paper's scalar `u_t = mean_k u_k`, `ū_t = αū_{t-1} + (1−α)u_t`. Timing matches Alg. 1:
  γ_t uses `Δu_{t-1} = u_{t-1} − ū_{t-2}`; γ=1 for the first two control steps.
* Adaptive visual attention (L66-137, L692-697): `γ = κ^{tanh(s·mean Δu)}` with s=`attn_sensitivity`; applied to
  every self-attention block of **both** DINOv2 and SigLIP (`scale_new = head_dim^{-1/2}/γ`), matching App. G.3.2.

**Trace.** observation (`get_libero_image`: 180° rotate, JPEG round-trip, lanczos 224) → `get_vla_action`
(center-crop 0.9, prompt `In: What action should the robot take to {task}?\nOut:`) → `predict_action(scale)`:
`apply_visual_attention_temperature(γ_t)` → `vision_backbone` (DINOv2+SigLIP with patched attention) → projector →
Llama-2 + KV-cache loop over 7 action tokens → per-token logits → `u_k` → `τ_k` → sampled token → de-tokenise
(bin centres, q01/q99 un-normalisation) → gripper normalise/invert → `env.step`. The new `[ema_u, last_u]` state is
returned and threaded into the next control step.

**Code ↔ paper discrepancies**

| # | Discrepancy | Impact |
|---|---|---|
| A1 | **Undocumented `attn_sensitivity` s=0.3 inside tanh.** Paper Eq. 8 is `γ_t = κ^{tanh(Δu_{t-1})}`, and App. G.3.3 lists only T0, κ, α. | γ responds ~3× more weakly to Δu than the paper equation. Paper numbers may come from either form. Must be reported as a code-defined hyperparameter. |
| A2 | `q_low` parameterisation: code puts ε on every non-top token (`1-(K-1)ε` on top-1); the paper uses `1-ε` and `ε/(|V|-1)`. | Code ε=1e-12 ≡ paper ε≈2.55e-10. Shifts the u=0 crossing from p_max≈0.83 (paper form) to ≈0.80. Paper Table 14 suggests ±1 pp sensitivity. |
| A3 | The vision encoder switches from timm fused SDPA to a manual attention path at the first SCALE call, even when γ=1. | SCALE is not bit-comparable to greedy even when γ=1, so "κ→1, T0→0" ablations will not reproduce greedy exactly. Minor. |
| A4 | Paper: 3 seeds. Code: one default seed, seeds unlisted. Greedy decoding with fixed init states and `env.seed(0)` should be deterministic, yet the paper reports greedy ± 0.9 pp. | The variance source for greedy is unexplained (GPU nondeterminism? TF image ops?). We record per-run fingerprints. |
| A5 | LIBERO unpinned. | Fixed by our pin. |
| A6 | Only OpenVLA/LIBERO released. | Other backbones are not reproducible from public code. |

**Classification: SCALE (OpenVLA + LIBERO) = COMPLETE** (runnable end to end with public checkpoints, all
hyperparameters in config), subject to A1/A2 being reported. The paper as a whole is **PARTIAL** (A6).

---

## 4. S0B — CAG / LIBERO-CF audit

**Bundled openpi.** `external/libero-cf/openpi-cf` is a vendored openpi fork with **no git history** (LIBERO-CF is a
single "first commit"), so the upstream base commit cannot be read off. Lock: jax 0.5.3, flax 0.10.2,
orbax-checkpoint 0.11.13, transformers 4.53.2, torch 2.7.1, lerobot @ `0cf86487`. Fork additions:
`Pi0Config.mask_language_tokens/mask_vision_tokens/empty_language_*`, `InjectDefaultPrompt(probability=…)`, the
`pi05_libero_unconditioned` training config, and `scripts/serve_policy_{TF,VA}.py`.

**π0.5 config used** (`pi05_libero`): `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`,
action_dim 32 (7 used), max_token_len 200, bf16, 10 Euler flow steps. Note: with `discrete_state_input=False` and
π0.5 having no continuous state token, **pi05_libero receives no proprioceptive state at all**.

**CAG-TF trace** (`openpi-cf/scripts/serve_policy_TF.py`):

1. `_prepare_inputs` (L124-132): the same input transforms as vanilla (`LiberoInputs` → normalise → tokenise
   `"<prompt>"` + images base / left-wrist / right-wrist (zeros, `image_mask=False`)).
2. Unconditioned branch (L113-119, default `uncond_empty_prompt=False`): `tokenized_prompt` and
   `tokenized_prompt_mask` are **zeroed**. All language positions (including BOS) are masked out of attention and
   positions, so the prefix is images only.
3. RNG / noise (L157-161): `policy._rng → split 4 → (new_rng, noise_rng, cond_rng, uncond_rng)`;
   `noise = N(0,I)` of shape (1,10,32) is **shared by both branches**. `cond_rng`/`uncond_rng` are unused by
   `sample_actions` once noise is provided.
4. Each branch runs the full 10-step Euler ODE (`Pi0.sample_actions`, deterministic given noise).
5. Combination (L172): `a = a_uncond + ω (a_cond − a_uncond)` on the **final normalised action chunks**,
   i.e. extrapolation in action space after integration, not CFG on the velocity field at each denoising step.
   The paper (Eq. 4) states it at the policy level and does not pin this down. Unnormalisation (affine quantile
   map) then `[:, :7]`.
6. The websocket server returns the (10,7) chunk; the client executes the first `replan_steps=5`.

Vanilla baseline: `serve_policy.py` → `Policy.infer` (`split 2`, noise drawn inside `sample_actions`).
**Verified:** with JAX 0.5.3 (`threefry_partitionable=True`), `split(k,2)[i] == split(k,4)[i]`, so vanilla and CAG
draw **bit-identical noise at the same server call index**. Episodes still desynchronise, because call counts
differ between conditions (smoke: B ep0 = 21 calls, S ep0 = 23).

**Guidance scale.** README TF command: `--guidance_scale 2.0` (π0 and π0.5). Paper App. B: "For CAG, we use a
guidance scale of ω = 1.5 for both TF and VA" (π0/π0.5), and ω = 3.0 for OpenVLA-OFT, whose README also says 2.0.
**The audit therefore indicates 1.5 for π0.5 TF**, which is what the smoke test used. README 2.0 is a recorded
discrepancy (C1). It is not tuned on outcomes.

**Client / benchmark** (`external/libero-cf/eval/main_cf.py`): `np.random.seed(7)`, `env.seed(7)` (OpenVLA/SCALE
uses `env.seed(0)`), 224 resize-with-pad, 10 wait steps, `replan_steps=5`, max 220 steps (CF-Spatial),
50 trials/task over init states 0..49 (`init_files/libero_cf_spatial/*.pruned_init`, 50×92 each),
prompt = **file-name-derived language** (`grab_language_from_filename`).
Metrics per condition: *touch* = any MuJoCo contact between a gripper body (name contains
gripper/finger/hand) and the object's root body, checked after every step; *success* = predicate true at
episode end (`evaluate_conditions`); episodes terminate when **any** listed condition holds
(`set_success_any_conditions`, `problems/libero_tabletop_manipulation.py:134-147`).

**Benchmark/code issues found**

| # | Issue | Handling |
|---|---|---|
| C1 | ω: README 2.0 vs paper 1.5 (π0.5 TF). | Use 1.5 (paper). Flag. A second confirmatory ω is the user's call, not chosen on outcomes. |
| C2 | Faithful/Biased labels are **not encoded anywhere**: every CF-Spatial task lists the same 4 conditions. | Reconstructed from the task files (`src/metrics/libero_cf_labels.py`): biased = BDDL `:goal` subject (the original LIBERO training goal, always `akita_black_bowl_1`); faithful = object named in the prompt the policy receives, with multi-instance bowls resolved by the prompt's spatial phrase against BDDL `:init`. Pinned by `tests/test_libero_cf_labels.py`. |
| C3 | **Prompt ≠ BDDL `:language` for task 13** ("black bowl *on* the wooden cabinet" → bowl_2 vs ":language … *in the top layer of* the wooden cabinet" → bowl_1, which is the biased object) **and task 14** (prompt "ramekin" vs ":language … akita black bowl next to the plate"). | Labels follow the prompt actually sent. **Task 14 is one of the preregistered pilot tasks (0, 7, 14)**; it is kept, but flagged. |
| C4 | `env.set_custom_conditions` is **commented out** in `ControlEnv` (`envs/env_wrapper.py:109`); `main_cf.py` calls it inside `try/except: pass`. | Upstream it is a silent no-op; conditions are passed explicitly to `evaluate_conditions`. Our runner mirrors that. |
| C5 | `openpi-cf` not buildable as shipped (missing README.md). | Workaround above (no source change). |
| C6 | `main_cf.py` cannot select tasks (only `starting_task_id`, then all tasks to the end). | Our runner adds `--task-ids`; the loop is otherwise line-for-line identical. |
| C7 | The policy RNG is a per-server-process stream starting at `jax.random.key(0)`. Results depend on server restarts and on episode order. | One fresh server per run, recorded per-call key fingerprints. |

**Classification: π0.5 + CAG-TF = RECONSTRUCTIBLE.** The method path is complete and runs with the public
checkpoint, but reproducing the paper's Faithful/Biased metrics requires the reconstructed labels (C2) and a
guidance-scale decision (C1).

---

## 5. S0C — π0.5 action-expert attention feasibility

**Architecture** (`openpi-cf/src/openpi/models/{pi0.py,gemma.py}`). One Gemma stack with two expert weight sets
(PaliGemma 2B: width 2048; action expert gemma_300m: width 1024), depth 18, **8 query heads, 1 KV head (MQA),
head_dim 256**, shared self-attention. π0.5 uses adaRMS time conditioning in the action expert.

**Token layout** (per query): prefix S_p = 968 = `[0,256)` base camera | `[256,512)` left wrist | `[512,768)` right
wrist (**masked**, `image_mask=False` for LIBERO π0.5) | `[768,968)` language (200 slots; 13 valid for CF task 0).
Suffix = 10 action tokens (bidirectional within the chunk, attend to all valid prefix tokens). Denoising: 10 Euler
steps (`lax.while_loop`); each step runs the suffix through all 18 layers against the cached prefix KV.

**Attention tensor.** `gemma.Attention` computes `probs = softmax(masked logits)` with explicit `einsum` (no fused
flash kernel), so the weights are materialised inside the XLA graph but not returned. Per suffix pass:
`[B, K=1, G=8, T=10, S=978]` per layer → per query `[steps=10, L=18, 1, 1, 8, 10, 978]` (bf16, 28.2 MB).

**Instrumentation** (`src/instrumentation/pi05_attention.py`, diff vs upstream in
`patches/gemma_attention_capture.diff`):

* `CapturingAttention` is a verbatim copy of `gemma.Attention.__call__`. Its only change is that, when a
  trace-time capture flag is on *and* a KV cache is present (action-expert passes only), it also returns `probs` as
  a third kv-cache element. `nn.scan` then stacks it per layer.
* The diagnostic sampler mirrors `Pi0.sample_actions`, keeping `lax.while_loop`, and stores each step's probs in a
  loop-carry buffer. openpi's runtime jaxtyping check is disabled only for this trace (`at.disable_typechecking`).

**Identity test** (`scripts/audit/attention_identity_test.py`; real CF-Spatial task-0 observation; fixed noise;
results in `results/S0C/attention_identity_result.json`):

| Comparison | max\|Δ\| | bit-identical |
|---|---|---|
| stock vs stock (repeat) | 0 | yes |
| stock vs patched class, capture OFF (fresh trace) | 0 | **yes** |
| stock `Policy.infer` vs patched `Policy.infer` (same policy RNG key) | 0 | **yes** |
| stock vs **diagnostic capture ON** (`while_loop`) | 0 | **yes** |
| stock vs diagnostic capture ON, Python-unrolled loop (first attempt) | 1.95e-3 (~1 bf16 ULP) | no (XLA fusion differs) |

Sanity: attention rows sum to 1 ± 0.003 (bf16 rounding of probs); exactly 0 mass on masked columns. Latency
70 ms for the stock sampler vs 68 ms for the diagnostic (warm, single query); diagnostic compile 4.4 s once.

**Classification: GO.** Attention can be exposed without changing the action (bit-identical).

**Paper-definition caveat found during S0C (for S3/S4).** KWTS (arXiv 2609.00908) defines `p_{j,i} = Norm(Σ_l
head-mean α_{l,j,i})` over N = 968 VLM tokens and detects a plateau at `Ē ≥ η ln N` with η = 0.95 (π0.5 LIBERO:
k=3, τ=0.01). In the public `pi05_libero` pipeline, 256 right-wrist tokens and ~187 language padding slots are
masked, so at most ~525 columns can carry mass and `E ≤ ln 525 ≈ 0.91 ln 968`. **Under the paper's stated N the
η=0.95 threshold is unreachable**, and the rule would always execute the full horizon (≡ Fixed-10). The paper's
LIBERO numbers (Ours 97.25 vs Fixed-10 94.75) imply a different N, masking, or aggregation than written. The
exact definition must be resolved before S4. On the S0C observation: KWTS `E/ln 968` = 0.44 → 0.50 across k=1..10
at the final step, and image-only per-head normalised entropy ≈ 0.67–0.69.
