# Seis-Codec: Seismic Data Compression

This directory implements the logic for seismic wave data compression, combining ideas from high-fidelity audio compression (Descript Audio Codec) with seismic wave analysis (seisLM / SeisBench).

## Overview

The primary goal is to compress seismic wave data and reduce its bitrate while maximally preserving the accuracy on downstream tasks (such as earthquake detection, phase picking, magnitude estimation, etc.).

We utilize the architecture of **DAC (Descript Audio Codec)** and apply it to seismic data:
- `model.py` contains `SeisDAC`, which extends the standard DAC model to support multi-channel inputs (e.g., Z, N, E components) and outputs instead of the default 1-channel audio.
- `train.py` provides the training loop utilizing PyTorch Lightning for ease of use, and `seisbench` data loaders derived from `seisLM`.

## Running the Code

Ensure your environment has the required dependencies (which include both `dac` and `seisbench`, as well as `lightning` and `ml_collections`). This can generally be set up by utilizing the environment from `seisLM` and additionally installing `descript-audio-codec`.

To start a test training run:
```bash
python3 train.py --test_run
```

To run a full training:
```bash
python3 train.py
```

### Train with a factorized entropy model

Warm-start the current best no-GAN checkpoint and jointly optimize waveform
distortion plus the estimated rate in kbps:

```bash
./run_factorized_entropy_train.sh
```

The default run uses `rate_loss_weight=0.1`, a five-epoch rate warm-up, 50
fine-tuning epochs, and all visible GPUs with batch size 8 per GPU. Settings can
be overridden without editing the script:

```bash
RATE_LOSS_WEIGHT=0.3 MAX_EPOCHS=50 BATCH_SIZE=8 \
  ./run_factorized_entropy_train.sh
```

### Measure entropy-coded SeisDAC bitrate

The theoretical SeisDAC bitrate assumes fixed-width RVQ indices. To measure
actual reversible streams with per-window headers, zstd, and factorized rANS
when the checkpoint contains an entropy model, run:

```bash
./run_seisdac_entropy.sh
```

By default this evaluates 512 ETHZ development windows and writes CSV, JSON,
LaTeX, and PDF/SVG rate-distortion outputs to
`/data/seismic/seis-codec-eval/seisdac_entropy_ethz`. Override settings with
environment variables, for example:

```bash
NUM_SAMPLES=64 DATA_NAME=STEAD OUTPUT_DIR=/data/seismic/seis-codec-eval/seisdac_entropy_stead \
  ./run_seisdac_entropy.sh
```

Evaluate a trained factorized checkpoint with its learned CDF and actual rANS
streams using:

```bash
CKPT=/path/to/factorized/checkpoints/best.ckpt \
OUTPUT_DIR=/data/seismic/seis-codec-eval/factorized_entropy_ethz \
  ./run_seisdac_entropy.sh
```

### Calibrate a factorized PMF without changing the codec

Fit one categorical probability table per RVQ codebook from the complete ETHZ
training split, attach it to the existing no-GAN checkpoint, and preserve all
encoder, quantizer, and decoder weights:

```bash
python3 calibrate_factorized_entropy.py
```

The calibrated checkpoint and its train-split count summary are written under
`/data/seismic/seis-codec-logs/ethz_nogan_spectral/seislm_enc_peaknorm_latreg0p003_nogan_200ep_freeze_entropy_calibrated/checkpoints`.
Measure its held-out development bitrate with:

```bash
CKPT=/data/seismic/seis-codec-logs/ethz_nogan_spectral/seislm_enc_peaknorm_latreg0p003_nogan_200ep_freeze_entropy_calibrated/checkpoints/calibrated_train_pmf.ckpt \
OUTPUT_DIR=/data/seismic/seis-codec-eval/seisdac_entropy_ethz_calibrated_train_pmf \
  ./run_seisdac_entropy.sh
```

### Task-Aware Loss (Feature-matching with SeisLM)
If you want the model to aggressively preserve features necessary for downstream tasks, you can enable the **task-aware loss**. This uses a frozen pre-trained `seisLM` model to extract intermediate representations and computes an L1 loss between the original and reconstructed waveforms. This forces the compression codec to preserve the critical features that `seisLM` expects.

To run with task-aware loss enabled:
```bash
python3 train.py --use_task_aware_loss --seis_lm_checkpoint /path/to/pretrained/seislm.ckpt
```

## Adaptation Details
1. **Multi-Channel Adaptation**: We modified the first and last `WNConv1d` layers in the `DAC` Encoder and Decoder to take an arbitrary number of `in_channels` (defaulting to 3 for ZNE components of seismic waveforms).
2. **Loss Formulation**: We retained the GAN (Discriminator/Generator) configuration, L1 loss, and VQ (Commitment/Codebook) losses. Spectral losses (e.g., STFT, MelSpectrogram) can be enabled if STFT parameters are appropriately adjusted for the very low sample rate of seismic waves (typically 100 Hz vs. audio's 44100 Hz).
3. **Data Pipeline**: The data loading logic is seamlessly integrated with the data handlers in `seisLM` which utilizes `SeisBench`.
