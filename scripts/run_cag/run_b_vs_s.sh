#!/usr/bin/env bash
# S2: condition B (vanilla pi0.5) then S (pi0.5 + CAG-TF) on libero_cf_spatial, same tasks / init states.
# A fresh server per condition so each run starts from the policy's native RNG state (jax.random.key(0)):
# matched initial states under native policy stochasticity (NOT noise-paired).
# DIAG=1 (default) serves through DiagnosticPolicy: guidance magnitude + action-expert attention logging with
# bit-identical actions (scripts/audit/diagnostics_identity_test.py).
# Usage: [TASKS=0] [EPISODES=2] [PORT=8765] [DIAG=1] [RAW_EPISODES=2] bash scripts/run_cag/run_b_vs_s.sh <gpu> <guidance_scale> <out_dir>
#   smoke : DIAG=0 bash scripts/run_cag/run_b_vs_s.sh 0 1.5 results/S2_smoke
#   pilot : TASKS=0,6,12 EPISODES=20 bash scripts/run_cag/run_b_vs_s.sh 0 1.5 results/S2_pilot
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GPU=${1:-0}; W=${2:-1.5}; OUT=${3:-$ROOT/results/S2_smoke}
PORT=${PORT:-8765}; TASKS=${TASKS:-0}; EPISODES=${EPISODES:-2}; DIAG=${DIAG:-1}; RAW_EPISODES=${RAW_EPISODES:-2}
SERVER_FLAGS=(); [ "$DIAG" = "1" ] && SERVER_FLAGS+=(--diagnostics)
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
    --raw-attention-episodes "$RAW_EPISODES" \
    2>&1 | grep -v -E "EGL|Warning|^\s|^\)|Traceback|Exception ignored|OpenGL|cArguments|result =" | tee "$OUT/$cond/client.log"
  kill $pid; wait $pid 2>/dev/null || true
}
run_condition B --mode vanilla
run_condition S --mode cag_tf --guidance-scale "$W"
