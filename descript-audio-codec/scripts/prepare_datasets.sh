#!/usr/bin/env bash
# One-click dataset preparation for descript-audio-codec training.
#
# Usage:
#   bash scripts/prepare_datasets.sh
#   bash scripts/prepare_datasets.sh /path/to/data/root
#   DAC_DATA_ROOT=/data/audio bash scripts/prepare_datasets.sh --tier minimal
#   DAC_DATA_ROOT=/data/audio bash scripts/prepare_datasets.sh --tier full --include-audioset-unbalanced
#
# Tiers (see prepare_datasets.py --help for details):
#   minimal  ~40 GB   DAPS, MUSDB, VCTK, small DNS speech subsets
#   speech   ~100+ GB  minimal + more DNS speech + Common Voice
#   full     TB-scale  all DNS speech shards + Jamendo + AudioSet

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${DAC_DATA_ROOT:-${1:-/data/audio}}"

if [[ "${1:-}" == /* ]]; then
  shift || true
fi

echo "DAC dataset preparation"
echo "  repo:      ${REPO_ROOT}"
echo "  data root: ${DATA_ROOT}"
echo "  log:       ${DATA_ROOT}/dac_prepare_datasets.log"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found." >&2
  exit 1
fi

for tool in curl tar unzip; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Error: required tool '${tool}' not found." >&2
    exit 1
  fi
done

export DAC_DATA_ROOT="${DATA_ROOT}"
python3 "${SCRIPT_DIR}/prepare_datasets.py" --data-root "${DATA_ROOT}" "$@"
