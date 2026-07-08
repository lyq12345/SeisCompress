#!/usr/bin/env bash
# One-click dataset preparation for seis-codec/train.py
#
# Usage:
#   bash scripts/prepare_datasets.sh
#   bash scripts/prepare_datasets.sh /path/to/data/root
#   SEISCOMPRESS_DATA_ROOT=/path/to/data bash scripts/prepare_datasets.sh
#
# Downloads (by default):
#   - ETHZ from SeisBench (~22 GB waveforms)
#   - Foreshock-aftershock NRCA validation data (~525 MB zip)
# Creates:
#   - $DATA_ROOT/seisbench/          SeisBench cache
#   - $DATA_ROOT/seislm/             task-specific data
#   - SeisCompress/data -> seislm/   symlink for train.py
#   - $DATA_ROOT/seislm_env.sh       env vars for training

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${SEISCOMPRESS_DATA_ROOT:-${1:-/srv/disk01/yuqiao-datasets/Seismic}}"

echo "SeisCompress dataset preparation"
echo "  repo:      ${REPO_ROOT}"
echo "  data root: ${DATA_ROOT}"
echo "  log:       ${DATA_ROOT}/seiscompress_prepare_datasets.log"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found." >&2
  exit 1
fi

export SEISCOMPRESS_DATA_ROOT="${DATA_ROOT}"
python3 "${SCRIPT_DIR}/prepare_datasets.py" --data-root "${DATA_ROOT}"
