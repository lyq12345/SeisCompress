#!/usr/bin/env bash
set -euo pipefail

CKPT="${CKPT:-/data/seismic/seis-codec-logs/ethz_nogan_spectral/seislm_enc_peaknorm_latreg0p003_nogan_200ep_freeze/checkpoints/best-137-24288.ckpt}"
DATA_NAME="${DATA_NAME:-ETHZ}"
SPLIT="${SPLIT:-dev}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/seismic/seis-codec-eval/codec_baselines_ethz}"
OURS_METRICS="${OURS_METRICS:-/data/seismic/seis-codec-eval/ethz_nogan_latreg0p003_best137/metrics.txt}"
SEISDAC_QUANTIZERS="${SEISDAC_QUANTIZERS:-1,2,3,4,5,6,7,8,9}"
DEVICE="${DEVICE:-$(python3 - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)}"
BATCH_SIZE="${BATCH_SIZE:-32}"

python3 evaluate_codec_baselines.py \
  --checkpoint "$CKPT" \
  --data_name "$DATA_NAME" \
  --split "$SPLIT" \
  --num_samples "$NUM_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --ours_metrics "$OURS_METRICS" \
  --seisdac_quantizers "$SEISDAC_QUANTIZERS" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE"
