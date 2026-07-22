#!/usr/bin/env bash
set -euo pipefail

INIT_CKPT="${INIT_CKPT:-/data/seismic/seis-codec-logs/ethz_nogan_spectral/seislm_enc_peaknorm_latreg0p003_nogan_200ep_freeze/checkpoints/best-137-24288.ckpt}"
SEISLM_CKPT="${SEISLM_CKPT:-/data/seismic/seisLM/pretrained_models/pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt}"
RATE_LOSS_WEIGHT="${RATE_LOSS_WEIGHT:-0.1}"
RATE_LOSS_WARMUP_EPOCHS="${RATE_LOSS_WARMUP_EPOCHS:-5}"
ENTROPY_TEMPERATURE="${ENTROPY_TEMPERATURE:-0.1}"
ENTROPY_LEARNING_RATE="${ENTROPY_LEARNING_RATE:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICES="${DEVICES:--1}"
LOG_DIR="${LOG_DIR:-/data/seismic/seis-codec-logs}"
LOG_NAME="${LOG_NAME:-ethz_nogan_spectral}"
RATE_TAG="${RATE_LOSS_WEIGHT//./p}"
LOG_VERSION="${LOG_VERSION:-seislm_enc_peaknorm_latreg0p003_nogan_factorized_rate${RATE_TAG}_${MAX_EPOCHS}ep}"

python3 train.py \
  --init_checkpoint "$INIT_CKPT" \
  --no_gan \
  --use_spectral_loss \
  --use_seislm_encoder \
  --seislm_encoder_checkpoint "$SEISLM_CKPT" \
  --freeze_seislm_extractor \
  --amp_norm_type peak \
  --latent_reg_weight 0.003 \
  --latent_reg_threshold 1.0 \
  --latent_reg_target latents \
  --use_entropy_model \
  --rate_loss_weight "$RATE_LOSS_WEIGHT" \
  --rate_loss_warmup_epochs "$RATE_LOSS_WARMUP_EPOCHS" \
  --entropy_temperature "$ENTROPY_TEMPERATURE" \
  --entropy_learning_rate "$ENTROPY_LEARNING_RATE" \
  --entropy_cdf_precision 16 \
  --learning_rate 1e-4 \
  --gradient_clip_g 100 \
  --max_epochs "$MAX_EPOCHS" \
  --disable_early_stopping \
  --batch_size "$BATCH_SIZE" \
  --devices "$DEVICES" \
  --log_dir "$LOG_DIR" \
  --log_name "$LOG_NAME" \
  --log_version "$LOG_VERSION"
