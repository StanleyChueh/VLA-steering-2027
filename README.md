# VLA-steering-2027

Do semantic/perceptual failures and action-execution failures expose different internal reliability signals in a
frozen VLA, and can intervention be targeted at the right stage? Zero-retraining study on 2× RTX 4090.

Status and decisions: `reports/` (start with `reports/PILOT_DECISION.md`; S0 decisions in `reports/NEXT_EXPERIMENT_DECISION.md`).

## Layout

| Path | Contents |
|---|---|
| `external/` | upstream repos, read-only, gitignored; pinned SHAs in `reports/S0_PROVENANCE_AND_FEASIBILITY.md` |
| `envs/` | isolated venvs (gitignored), built by `scripts/audit/setup_envs.sh {openpi,client,scale}` |
| `configs/libero/{upstream,cf}` | project-local LIBERO path configs (`LIBERO_CONFIG_PATH`) |
| `configs/cag/` | frozen LIBERO-CF task-validity manifest and S2/S3 preregistration |
| `src/wrappers/` | run manifests; π0.5 / CAG-TF policy server with per-call provenance; scene-geometry fingerprint |
| `src/instrumentation/` | π0.5 action-expert attention capture; `DiagnosticPolicy` (CAG guidance magnitude + attention entropies, bit-identical actions) |
| `src/metrics/` | LIBERO-CF faithful/biased label reconstruction |
| `src/analysis/` | run summaries, Wilson intervals, S2 / S3 pilot analyses |
| `scripts/audit/` | downloads, env setup, identity test, observation dump |
| `scripts/run_scale/`, `scripts/run_cag/` | evaluation runners (mirror the upstream loops; logging only) |
| `patches/` | exact diffs of any upstream code we copied and changed |
| `results/` | run outputs (each with `manifest.json`); `attn/` and `raw/` tensors are gitignored |
| `tests/` | `uvx pytest tests/` |

## Quick start

```bash
bash scripts/audit/download_checkpoints.sh all     # OpenVLA libero-10 + pi05_libero
bash scripts/audit/setup_envs.sh all
bash scripts/run_cag/run_b_vs_s.sh 0 1.5 results/S2_smoke             # B vs CAG-TF smoke
CUDA_VISIBLE_DEVICES=0 envs/scale/bin/python scripts/run_scale/run_scale_eval.py \
    --task-suite libero_10 --task-ids 0 --episodes 2 --decoding-mode scale --out results/S1_smoke/scale
```
