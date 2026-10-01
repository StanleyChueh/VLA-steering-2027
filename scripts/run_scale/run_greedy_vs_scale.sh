#!/usr/bin/env bash
# S1: OpenVLA greedy vs SCALE (paper defaults) on standard LIBERO, same tasks / init states / seed.
# Usage: bash scripts/run_scale/run_greedy_vs_scale.sh <gpu> <suite> <task_ids> <episodes> <out_dir>
#   smoke  : bash scripts/run_scale/run_greedy_vs_scale.sh 0 libero_10 0 2 results/S1_smoke
#   minimal: bash scripts/run_scale/run_greedy_vs_scale.sh 1 libero_10 0,4,9 10 results/S1_minimal
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GPU=$1; SUITE=$2; TASKS=$3; EPISODES=$4; OUT=$5
mkdir -p "$OUT"
for mode in greedy scale; do
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl "$ROOT/envs/scale/bin/python" "$ROOT/scripts/run_scale/run_scale_eval.py" \
    --task-suite "$SUITE" --task-ids "$TASKS" --episodes "$EPISODES" --decoding-mode $mode --out "$OUT/$mode" \
    > "$OUT/$mode.log" 2>&1
  grep -E "^\[$mode\]" "$OUT/$mode.log"
  # Any episode that ended in an exception is an infrastructure failure, not a policy failure.
  if grep -q '"error": "' "$OUT/$mode/episodes.jsonl"; then echo "ERROR episodes in $mode - inspect $OUT/$mode.log"; exit 1; fi
done
