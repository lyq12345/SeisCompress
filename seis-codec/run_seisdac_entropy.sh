#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env.seis-codec ]]; then
  # shellcheck disable=SC1091
  source .env.seis-codec
fi

CKPT="${CKPT:-/data/seismic/seis-codec-logs/ethz_nogan_spectral/seislm_enc_peaknorm_latreg0p003_nogan_200ep_freeze/checkpoints/best-137-24288.ckpt}"
DATA_NAME="${DATA_NAME:-ETHZ}"
SPLIT="${SPLIT:-dev}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/seismic/seis-codec-eval/seisdac_entropy_ethz}"
BASELINE_CSV="${BASELINE_CSV:-/data/seismic/seis-codec-eval/codec_baselines_ethz/codec_baseline_results.csv}"
SEISDAC_QUANTIZERS="${SEISDAC_QUANTIZERS:-1,2,3,4,5,6,7,8,9}"
ZSTD_LEVEL="${ZSTD_LEVEL:-9}"
DEVICE="${DEVICE:-$(python3 - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)}"
BATCH_SIZE="${BATCH_SIZE:-32}"

python3 evaluate_seisdac_entropy.py \
  --checkpoint "$CKPT" \
  --data_name "$DATA_NAME" \
  --split "$SPLIT" \
  --num_samples "$NUM_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --baseline_csv "$BASELINE_CSV" \
  --seisdac_quantizers "$SEISDAC_QUANTIZERS" \
  --zstd_level "$ZSTD_LEVEL" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE"
