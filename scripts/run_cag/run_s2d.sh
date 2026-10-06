#!/usr/bin/env bash
# S2d oracle reference test (reports/S2D_ORACLE_REFERENCE_PREREG.md): SOURCE-NORM (SN), OTHER (OT), SOURCE-RAW (SR)
# under S2b CRN, CF-Spatial tasks 0-12 x states 20-29, master seeds 0 and 1 -> 6 x 130 = 780 episodes.
# B and MASK are reused from results/S2b (exact replay verified). Seed m on GPU m; three servers per GPU.
# States >= 30 are refused by the client.
# Usage: bash scripts/run_cag/run_s2d.sh [out_dir=results/S2d]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=${1:-$ROOT/results/S2d}
mkdir -p "$OUT"
pids=()
for SEED in 0 1; do
  i=0
  for COND in SN OT SR; do
    PORT=$((8780 + 3 * SEED + i)); i=$((i + 1))
    DIAG=0 CRN_SEED=$SEED FORBID_INIT_FROM=30 CONDS=$COND TASKS=0,1,2,3,4,5,6,7,8,9,10,11,12 EPISODES=10 INIT_START=20 \
      RAW_EPISODES=0 PORT=$PORT bash "$ROOT/scripts/run_cag/run_b_vs_s.sh" "$SEED" 1.5 "$OUT/seed$SEED" \
      > "$OUT/seed$SEED.$COND.driver.log" 2>&1 &
    pids+=($!)
    sleep 20
  done
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
exit $status
