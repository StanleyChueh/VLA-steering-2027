#!/usr/bin/env bash
# S2: condition B (vanilla pi0.5) then S (pi0.5 + CAG-TF) on libero_cf_spatial, same tasks / init states.
# A fresh server per condition so each run starts from the policy's native RNG state (jax.random.key(0)):
# matched initial states under native policy stochasticity (NOT noise-paired).
# DIAG=1 (default) serves through DiagnosticPolicy: guidance magnitude + action-expert attention logging with
# bit-identical actions (scripts/audit/diagnostics_identity_test.py).
# CRN_SEED=<m> (S2b): common random numbers, noise = f(m, task, init, call) (src/wrappers/crn.py).
# FORBID_INIT_FROM=<n>: the client refuses any initial-state index >= n.
# Usage: [TASKS=0] [EPISODES=2] [INIT_START=0] [CONDS="B S"] [POLICY_SEED=0] [PORT=8765] [DIAG=1] [RAW_EPISODES=2] bash scripts/run_cag/run_b_vs_s.sh <gpu> <guidance_scale> <out_dir>
#   smoke : DIAG=0 bash scripts/run_cag/run_b_vs_s.sh 0 1.5 results/S2_smoke
#   pilot : TASKS=0,6,12 EPISODES=20 bash scripts/run_cag/run_b_vs_s.sh 0 1.5 results/S2_pilot
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GPU=${1:-0}; W=${2:-1.5}; OUT=${3:-$ROOT/results/S2_smoke}
PORT=${PORT:-8765}; TASKS=${TASKS:-0}; EPISODES=${EPISODES:-2}; DIAG=${DIAG:-1}; RAW_EPISODES=${RAW_EPISODES:-2}
INIT_START=${INIT_START:-0}; CONDS=${CONDS:-"B S"}; POLICY_SEED=${POLICY_SEED:-0}
CRN_SEED=${CRN_SEED:-}; FORBID_INIT_FROM=${FORBID_INIT_FROM:-}
SERVER_FLAGS=(--policy-seed "$POLICY_SEED"); [ "$DIAG" = "1" ] && SERVER_FLAGS+=(--diagnostics)
CLIENT_FLAGS=()
[ -n "$CRN_SEED" ] && SERVER_FLAGS+=(--crn-master-seed "$CRN_SEED") && CLIENT_FLAGS+=(--crn-master-seed "$CRN_SEED")
# exact CRN needs cross-process bit-determinism of the B and S servers (reports/S2B_RNG_VALIDATION.md)
[ -n "$CRN_SEED" ] && export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
[ -n "$FORBID_INIT_FROM" ] && CLIENT_FLAGS+=(--forbid-init-from "$FORBID_INIT_FROM")
# S2d oracle reference conditions: SN (SOURCE-NORM), OT (OTHER, norm-matched), SR (SOURCE-RAW)
REF_PROMPTS=${REF_PROMPTS:-$ROOT/configs/cag/S2d_reference_prompts.json}
EXTRA_CLIENT=()
run_condition() {
  local cond=$1; shift
  mkdir -p "$OUT/$cond"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU "$ROOT/envs/openpi-cf/bin/python" \
    "$ROOT/src/wrappers/serve_cf_policy.py" --port $PORT "${SERVER_FLAGS[@]}" "$@" > "$OUT/$cond/server.log" 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true" EXIT
  until grep -q "server listening" "$OUT/$cond/server.log" 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 2; done
  kill -0 $pid || { echo "server for $cond died"; tail -20 "$OUT/$cond/server.log"; exit 1; }
  MUJOCO_GL=egl "$ROOT/envs/libero-cf-client/bin/python" "$ROOT/scripts/run_cag/run_cf_eval.py" \
    --suite libero_cf_spatial --task-ids "$TASKS" --episodes "$EPISODES" --condition "$cond" --port $PORT --out "$OUT/$cond" \
    --raw-attention-episodes "$RAW_EPISODES" --init-start "$INIT_START" "${CLIENT_FLAGS[@]}" "${EXTRA_CLIENT[@]}" \
    2>&1 | grep --line-buffered -v -E "EGL|Warning|^\s|^\)|Traceback|Exception ignored|OpenGL|cArguments|result =" | tee "$OUT/$cond/client.log"
  kill $pid; wait $pid 2>/dev/null || true
}
for c in $CONDS; do
  case $c in
    B) run_condition B --mode vanilla ;;
    S) run_condition S --mode cag_tf --guidance-scale "$W" ;;
    SN) EXTRA_CLIENT=(--ref-prompts "$REF_PROMPTS" --ref-kind source); run_condition SN --mode ref_norm --guidance-scale "$W" ;;
    OT) EXTRA_CLIENT=(--ref-prompts "$REF_PROMPTS" --ref-kind other); run_condition OT --mode ref_norm --guidance-scale "$W" ;;
    SR) EXTRA_CLIENT=(--ref-prompts "$REF_PROMPTS" --ref-kind source); run_condition SR --mode ref_raw --guidance-scale "$W" ;;
  esac
done
