#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env.seis-codec ]]; then
  # shellcheck disable=SC1091
  source .env.seis-codec
fi

python3 prepare_seisbench_datasets.py \
  --datasets seislm_pretrain \
  --cache_root "${SEISBENCH_CACHE_ROOT:-/data/seismic/seisbench}" \
  --sample_rate 100 \
  --component_order ZNE \
  --dimension_order NCW \
  --smoke_split dev \
  --smoke_samples 2
