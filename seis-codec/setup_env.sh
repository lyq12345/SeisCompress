#!/usr/bin/env bash
# Prepare Python environment for seis-codec training.
#
# Usage (from anywhere):
#   bash /path/to/SeisCompress/seis-codec/setup_env.sh
#
# CPU-only PyTorch:
#   USE_CPU=1 bash setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.seis-codec"

echo "==> Repo root: ${REPO_ROOT}"

if [[ "${USE_CPU:-0}" == "1" ]]; then
  echo "==> Installing PyTorch (CPU) + dependencies..."
  pip install --upgrade pip
  pip install -r "${SCRIPT_DIR}/requirements-cpu.txt"
else
  echo "==> Installing PyTorch (CUDA 12.4) + dependencies..."
  pip install --upgrade pip
  pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

echo "==> Installing descript-audio-codec (editable)..."
pip install -e "${REPO_ROOT}/descript-audio-codec"

# seisLM has no pinned deps in setup.py; expose via PYTHONPATH (avoids egg-info on small disks)
CACHE_ROOT="${SEISBENCH_CACHE_ROOT:-/tmp/seisbench-cache}"
mkdir -p "${CACHE_ROOT}"

cat > "${ENV_FILE}" << ENVEOF
# Source before training:  source seis-codec/.env.seis-codec
export SEISCOMPRESS_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/seisLM:${REPO_ROOT}/descript-audio-codec:\${PYTHONPATH:-}"
# ETHZ dataset is ~22GB; point cache to a disk with enough free space
export SEISBENCH_CACHE_ROOT="${CACHE_ROOT}"
ENVEOF

echo "==> Wrote ${ENV_FILE}"
echo ""
echo "Done. Activate paths with:"
echo "  source ${ENV_FILE}"
echo ""
echo "Then run training, e.g.:"
echo "  cd ${SCRIPT_DIR}"
echo "  python train.py --use_spectral_loss --log_name ethz_gan_spectral --log_version clip_normvq"
echo ""
echo "Quick smoke test:"
echo "  cd ${SCRIPT_DIR} && source ${ENV_FILE} && python train.py --test_run --use_spectral_loss"
