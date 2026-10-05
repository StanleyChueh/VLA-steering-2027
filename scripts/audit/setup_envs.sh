#!/usr/bin/env bash
# Build the three isolated environments used in Phase S0-S2. All venvs live under envs/ (gitignored);
# nothing is installed into external/ (upstream repos stay read-only; source trees are exposed via .pth).
#
#   envs/openpi-cf         : openpi as bundled in LIBERO-CF (policy server: vanilla pi0.5 + CAG-TF)
#   envs/libero-cf-client  : LIBERO-CF simulator/eval client (py3.8, per LIBERO-CF README)
#   envs/scale             : SCALE OpenVLA + upstream LIBERO (py3.10, per SCALE README)
#   envs/analysis          : offline analysis only (numpy/scipy/scikit-learn/matplotlib; S2b H1/H2 models)
#
# Usage: bash scripts/audit/setup_envs.sh [openpi|client|scale|analysis|all]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXT="$ROOT/external"
WHAT=${1:-all}

site_packages() { "$1/bin/python" -c "import site; print(site.getsitepackages()[0])"; }

setup_openpi() {
  # Deviation: openpi-cf/pyproject.toml references a README.md that LIBERO-CF does not ship, so the project
  # itself cannot be built. Install the locked dependency set only, then expose src/ via a .pth file.
  (cd "$EXT/libero-cf/openpi-cf" && GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$ROOT/envs/openpi-cf" \
    uv sync --frozen --no-install-project)
  echo "$EXT/libero-cf/openpi-cf/src" > "$(site_packages "$ROOT/envs/openpi-cf")/openpi_cf_src.pth"
}

setup_client() {
  uv venv --clear --python 3.8 "$ROOT/envs/libero-cf-client"
  local py="$ROOT/envs/libero-cf-client/bin/python"
  # Deviation: torch is only used client-side for torch.load() of init-state files
  # (libero/libero/benchmark/__init__.py), so the pinned torch 1.11.0 is installed as the CPU build
  # instead of +cu113 (saves ~1.7 GB; no GPU code runs in the client).
  # Resolved in one pass so that thop/robomimic cannot pull a newer CUDA torch.
  sed -E 's/^(torch==1\.11\.0|torchaudio==0\.11\.0|torchvision==0\.12\.0)\+cu113/\1+cpu/' \
    "$EXT/libero-cf/requirements.txt" > "$ROOT/envs/client_requirements.txt"
  uv pip install --python "$py" -r "$ROOT/envs/client_requirements.txt" -r "$EXT/libero-cf/requirements_libero.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
  uv pip install --python "$py" -e "$EXT/libero-cf/packages/openpi-client"
  # LIBERO-CF's `libero` package + eval/ helpers, without `pip install -e .` (which would write egg-info into external/).
  echo "$EXT/libero-cf" > "$(site_packages "$ROOT/envs/libero-cf-client")/libero_cf_src.pth"
}

setup_scale() {
  uv venv --clear --python 3.10 "$ROOT/envs/scale"
  local py="$ROOT/envs/scale/bin/python"
  # SCALE pyproject pins torch==2.2.0 / torchvision==0.17.0 / torchaudio==2.2.0 (PyPI default build: cu121).
  uv pip install --python "$py" torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
  uv pip install --python "$py" "accelerate>=0.25.0" draccus==0.8.0 einops huggingface_hub "imageio[ffmpeg]" \
    matplotlib numpy pyyaml rich sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 tqdm \
    transformers==4.40.1 wandb tensorflow==2.15.0 packaging ninja
  # flash-attn 2.5.5 (README step 4): official prebuilt wheel matching torch 2.2 / cp310 / cxx11abi=FALSE.
  uv pip install --python "$py" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
  # README steps 5-6: LIBERO + its runtime requirements, then the README's version pins.
  uv pip install --python "$py" -r "$EXT/scale/experiments/robot/libero/libero_requirements.txt"
  uv pip install --python "$py" numpy==1.26.4 mujoco==3.3.2
  # TF 2.15 runs OpenVLA's image preprocessing (JPEG/lanczos resize/crop) on the GPU and needs libdevice + ptxas.
  # The README gets them from conda `cuda-compiler`; here the pip component pinned by tensorflow[and-cuda]==2.15.0.
  uv pip install --python "$py" "nvidia-cuda-nvcc-cu12==12.2.140"
  # LIBERO (upstream, pinned clone in external/deps) and SCALE's `prismatic` package via .pth, not editable installs.
  printf '%s\n%s\n' "$EXT/deps/LIBERO" "$EXT/scale" > "$(site_packages "$ROOT/envs/scale")/scale_src.pth"
}

setup_analysis() {
  uv venv --clear --python 3.11 "$ROOT/envs/analysis"
  uv pip install --python "$ROOT/envs/analysis/bin/python" numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 \
    matplotlib==3.9.2 pyyaml
}

case "$WHAT" in
  openpi) setup_openpi ;;
  client) setup_client ;;
  scale) setup_scale ;;
  analysis) setup_analysis ;;
  all) setup_openpi; setup_client; setup_scale; setup_analysis ;;
esac
