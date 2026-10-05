#!/usr/bin/env bash
# S2b causal-discovery run (configs/cag/S2b_prereg.yaml): B vs CAG-TF (omega 1.5, h 5 of H 10) under common random
# numbers, CF-Spatial tasks 0-12 x initial states 20-29, master seeds 0 and 1 -> 4 x 130 = 520 episodes.
# Seed m runs on GPU m; its B and S servers share that GPU and use identical task/state/reset order.
# Initial states >= 30 are refused by the client (reserved: 30-39 method development, 40-49 final confirmation).
# Usage: bash scripts/run_cag/run_s2b.sh [out_dir=results/S2b]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=${1:-$ROOT/results/S2b}
TASKS=0,1,2,3,4,5,6,7,8,9,10,11,12
mkdir -p "$OUT"
pids=()
for SEED in 0 1; do
  for COND in B S; do
    if [ "$COND" = S ]; then OFF=1; else OFF=0; fi; PORT=$((8770 + 2 * SEED + OFF))
    CRN_SEED=$SEED FORBID_INIT_FROM=30 CONDS=$COND TASKS=$TASKS EPISODES=10 INIT_START=20 RAW_EPISODES=0 PORT=$PORT \
      bash "$ROOT/scripts/run_cag/run_b_vs_s.sh" "$SEED" 1.5 "$OUT/seed$SEED" > "$OUT/seed$SEED.$COND.driver.log" 2>&1 &
    pids+=($!)
    sleep 20  # stagger checkpoint loading
  done
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
exit $status
