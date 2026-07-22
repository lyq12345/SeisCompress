#!/usr/bin/env bash
set -euo pipefail

CKPT="${CKPT:-/data/seismic/seis-codec-logs/ethz_nogan_spectral/seislm_enc_peaknorm_latreg0p003_nogan_200ep_freeze/checkpoints/best-137-24288.ckpt}"
DATA_NAME="${DATA_NAME:-ETHZ}"
SPLIT="${SPLIT:-dev}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/seismic/seis-codec-eval/codec_baselines_ethz_picking}"
RECON_RESULTS="${RECON_RESULTS:-/data/seismic/seis-codec-eval/codec_baselines_ethz/codec_baseline_results.csv}"
SEISDAC_QUANTIZERS="${SEISDAC_QUANTIZERS:-1,2,3,4,5,6,7,8,9}"
INCLUDE_FAMILIES="${INCLUDE_FAMILIES:-}"
PICKER="${PICKER:-phasenet}"
PICKER_WEIGHTS="${PICKER_WEIGHTS:-ethz}"
DEVICE="${DEVICE:-$(python3 - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)}"
BATCH_SIZE="${BATCH_SIZE:-64}"

python3 evaluate_codec_baseline_picking.py \
  --checkpoint "$CKPT" \
  --data_name "$DATA_NAME" \
  --split "$SPLIT" \
  --num_samples "$NUM_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --reconstruction_results "$RECON_RESULTS" \
  --seisdac_quantizers "$SEISDAC_QUANTIZERS" \
  --include_families "$INCLUDE_FAMILIES" \
  --picker "$PICKER" \
  --picker_weights "$PICKER_WEIGHTS" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE"
