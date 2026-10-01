#!/usr/bin/env bash
# Download the exact public checkpoints required by the S1/S2 smoke tests.
#   - OpenVLA libero-10 (SCALE, libero_10 suite)      -> HF cache (~/.cache/huggingface/hub)
#   - pi05_libero (CAG-TF on LIBERO-CF)                -> openpi cache (~/.cache/openpi/openpi-assets/...)
# The openpi checkpoint is fetched into a staging dir and renamed only after every file
# verifies against the GCS object size, so openpi never sees a partial checkpoint.
#
# Usage: bash scripts/audit/download_checkpoints.sh [openvla|pi05|all]
set -euo pipefail
WHAT=${1:-all}
LOG_DIR="$(cd "$(dirname "$0")/../.." && pwd)/results/downloads"
mkdir -p "$LOG_DIR"

download_openvla() {
  local repo="openvla/openvla-7b-finetuned-libero-10"
  # Revision pinned to the HF main SHA observed during the S0 audit (2026-10-01).
  local rev="80970322773f81baa2e22fe495d0487b93a05cfa"
  uvx --from "huggingface_hub[hf_xet]" hf download "$repo" --revision "$rev" 2>&1 | tail -5
  echo "openvla done: $repo@$rev"
}

download_pi05() {
  local prefix="checkpoints/pi05_libero"
  local cache="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}/openpi-assets"
  local final="$cache/$prefix"
  local stage="$cache/${prefix}.staging"
  if [[ -d "$final" ]]; then echo "pi05 already present at $final"; return; fi
  mkdir -p "$stage"
  curl -s "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=${prefix}/&fields=items(name,size,md5Hash)" \
    > "$LOG_DIR/pi05_libero_manifest.json"
  python3 - "$LOG_DIR/pi05_libero_manifest.json" "$prefix" > "$LOG_DIR/pi05_files.tsv" <<'EOF'
import json, sys
items = json.load(open(sys.argv[1]))["items"]
for it in items:
    print(f"{it['name']}\t{it['size']}\t{it['name'][len(sys.argv[2]) + 1:]}")
EOF
  # GCS throttles single streams, so fetch files in parallel; curl -C - resumes partial files.
  while IFS=$'\t' read -r name size rel; do
    mkdir -p "$stage/$(dirname "$rel")"
    local dst="$stage/$rel"
    if [[ -f "$dst" && $(stat -c %s "$dst") == "$size" ]]; then continue; fi
    curl -s --retry 10 -C - -o "$dst" "https://storage.googleapis.com/openpi-assets/$name" &
  done < "$LOG_DIR/pi05_files.tsv"
  wait
  while IFS=$'\t' read -r name size rel; do
    [[ $(stat -c %s "$stage/$rel") == "$size" ]] || { echo "SIZE MISMATCH $rel"; exit 1; }
    echo "ok $rel $size"
  done < "$LOG_DIR/pi05_files.tsv"
  mv "$stage" "$final"
  echo "pi05 done: $final"
}

case "$WHAT" in
  openvla) download_openvla ;;
  pi05) download_pi05 ;;
  all) download_openvla & download_pi05 & wait ;;
esac
