#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

if [[ -f .env.seis-codec ]]; then
  # shellcheck disable=SC1091
  source .env.seis-codec
fi

CKPT="${CKPT:-/data/seismic/seis-codec-logs/ethz_nogan_spectral/seislm_enc_peaknorm_latreg0p003_nogan_200ep_freeze/checkpoints/best-137-24288.ckpt}"
OUT_ROOT="${OUT_ROOT:-/data/seismic/seis-codec-eval}"
RECON_SAMPLES="${RECON_SAMPLES:-512}"
RECON_PLOTS="${RECON_PLOTS:-12}"
PICKING_SAMPLES="${PICKING_SAMPLES:-2048}"
PICKING_TOLERANCE_SEC="${PICKING_TOLERANCE_SEC:-0.2}"
PICKING_SEARCH_RADIUS_SEC="${PICKING_SEARCH_RADIUS_SEC:-1.0}"
FORCE="${FORCE:-0}"

RECON_DATASETS=(
  ETHZ
  STEAD
  GEOFON
  InstanceCountsCombined
  Iquique
  MLAAPDE
  PNW
  OBST2024
)

# Use only PhaseNet weights that are known to exist for these datasets.
# Format: SeisBenchDataset:PhaseNetWeight:output-prefix
PICKING_JOBS=(
  ETHZ:ethz:ethz
  STEAD:stead:stead
  GEOFON:geofon:geofon
  InstanceCountsCombined:instance:instance_counts_combined
  Iquique:iquique:iquique
)

safe_name() {
  case "$1" in
    ETHZ) echo "ethz" ;;
    STEAD) echo "stead" ;;
    GEOFON) echo "geofon" ;;
    InstanceCountsCombined) echo "instance_counts_combined" ;;
    Iquique) echo "iquique" ;;
    MLAAPDE) echo "mlaapde" ;;
    PNW) echo "pnw" ;;
    OBST2024) echo "obst2024" ;;
    *) echo "$1" | tr '[:upper:]' '[:lower:]' ;;
  esac
}

run_or_report() {
  local name="$1"
  shift
  echo
  echo "================================================================================"
  echo "$name"
  echo "================================================================================"
  "$@"
  local status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "FAILED: $name (exit=$status)" >&2
    return "$status"
  fi
  return 0
}

echo "Checkpoint: $CKPT"
echo "Output root: $OUT_ROOT"
echo "Reconstruction samples: $RECON_SAMPLES"
echo "Picking samples: $PICKING_SAMPLES"

mkdir -p "$OUT_ROOT"

failures=()

for data in "${RECON_DATASETS[@]}"; do
  safe="$(safe_name "$data")"
  out_dir="$OUT_ROOT/${safe}_nogan_latreg0p003_best137"
  metrics="$out_dir/metrics.txt"
  if [[ "$FORCE" != "1" && -f "$metrics" ]]; then
    echo "Skip reconstruction $data: $metrics exists"
    continue
  fi

  if ! run_or_report "Reconstruction evaluation: $data" \
    python3 evaluate.py \
      --checkpoint "$CKPT" \
      --data_name "$data" \
      --num_samples "$RECON_SAMPLES" \
      --num_plots "$RECON_PLOTS" \
      --output_dir "$out_dir"; then
    failures+=("reconstruction:$data")
  fi
done

for job in "${PICKING_JOBS[@]}"; do
  IFS=: read -r data weights prefix <<< "$job"
  out_dir="$OUT_ROOT/${prefix}_nogan_latreg0p003_picking"
  metrics="$out_dir/picking_metrics.txt"
  if [[ "$FORCE" != "1" && -f "$metrics" ]]; then
    echo "Skip picking $data: $metrics exists"
    continue
  fi

  if ! run_or_report "Phase-picking evaluation: $data with PhaseNet-$weights" \
    python3 evaluate_picking.py \
      --checkpoint "$CKPT" \
      --data_name "$data" \
      --split dev \
      --picker phasenet \
      --picker_weights "$weights" \
      --num_samples "$PICKING_SAMPLES" \
      --tolerance_sec "$PICKING_TOLERANCE_SEC" \
      --search_radius_sec "$PICKING_SEARCH_RADIUS_SEC" \
      --output_dir "$out_dir"; then
    failures+=("picking:$data")
  fi
done

echo
echo "================================================================================"
echo "Finished cross-dataset evaluation"
echo "================================================================================"

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "Failures:"
  printf '  %s\n' "${failures[@]}"
  exit 1
fi

echo "All requested evaluations completed or were already present."

if [[ -f summarize_cross_dataset_eval.py ]]; then
  python3 summarize_cross_dataset_eval.py \
    --out_root "$OUT_ROOT" \
    --output "$OUT_ROOT/cross_dataset_summary.md"
fi
