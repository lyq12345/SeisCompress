"""Calibrate causal first-order RVQ entropy tables from a training split."""

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from calibrate_factorized_entropy import (
    DEFAULT_CHECKPOINT,
    build_calibration_loader,
    seed_everything,
)
from factorized_entropy import FactorizedCategoricalEntropyModel
from first_order_entropy import FirstOrderCategoricalEntropyModel
from train import SeisDACLightning


DEFAULT_OUTPUT_CHECKPOINT = (
    "/data/seismic/seis-codec-logs/ethz_nogan_spectral/"
    "seislm_enc_peaknorm_latreg0p003_nogan_200ep_freeze_entropy_first_order/"
    "checkpoints/calibrated_train22626_first_order.ckpt"
)


def add_entropy_models(model: SeisDACLightning, cdf_precision: int) -> None:
    generator = model.generator
    generator.entropy_model = FactorizedCategoricalEntropyModel(
        n_codebooks=int(generator.n_codebooks),
        codebook_size=int(generator.codebook_size),
        cdf_precision=cdf_precision,
    )
    generator.first_order_entropy_model = FirstOrderCategoricalEntropyModel(
        n_codebooks=int(generator.n_codebooks),
        codebook_size=int(generator.codebook_size),
        cdf_precision=cdf_precision,
    )
    model.use_entropy_model = True
    model.config.model.use_entropy_model = True
    model.config.model.use_first_order_entropy_model = True
    model.config.model.entropy_cdf_precision = int(cdf_precision)
    model.config.model.entropy_temperature = float(
        model.config.model.get("entropy_temperature", 0.1)
    )
    model.config.training.rate_loss_weight = 0.0


@torch.inference_mode()
def collect_transition_counts(
    model: SeisDACLightning,
    loader,
    *,
    device: str,
    expected_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    generator = model.generator.eval()
    n_codebooks = int(generator.n_codebooks)
    codebook_size = int(generator.codebook_size)
    marginal_counts = torch.zeros(
        n_codebooks,
        codebook_size,
        dtype=torch.int64,
        device=device,
    )
    initial_counts = torch.zeros_like(marginal_counts)
    transition_counts = torch.zeros(
        n_codebooks,
        codebook_size * codebook_size,
        dtype=torch.int64,
        device=device,
    )
    processed = 0
    frames_per_window = None

    for batch_idx, batch in enumerate(loader):
        waveforms = batch["waveforms"].to(device=device, dtype=torch.float32, non_blocking=True)
        waveforms = torch.nan_to_num(waveforms, nan=0.0, posinf=0.0, neginf=0.0)
        prepared = generator.preprocess(waveforms, int(model.config.model.sample_rate))
        encoder_latent = generator.encoder(prepared)
        _, codes, _, _, _, _ = generator.quantizer(
            encoder_latent,
            n_quantizers=n_codebooks,
        )
        codes = codes.long()
        if frames_per_window is None:
            frames_per_window = int(codes.shape[-1])
        elif frames_per_window != int(codes.shape[-1]):
            raise RuntimeError("Calibration produced inconsistent latent frame counts")

        for codebook_idx in range(n_codebooks):
            values = codes[:, codebook_idx, :]
            marginal_counts[codebook_idx] += torch.bincount(
                values.reshape(-1),
                minlength=codebook_size,
            )
            initial_counts[codebook_idx] += torch.bincount(
                values[:, 0],
                minlength=codebook_size,
            )
            pair_ids = values[:, :-1] * codebook_size + values[:, 1:]
            transition_counts[codebook_idx].scatter_add_(
                0,
                pair_ids.reshape(-1),
                torch.ones_like(pair_ids.reshape(-1), dtype=torch.int64),
            )

        processed += int(codes.shape[0])
        if (batch_idx + 1) % 50 == 0 or processed == expected_samples:
            print(f"Collected {processed}/{expected_samples} transition windows")

    if processed != expected_samples or frames_per_window is None:
        raise RuntimeError(f"Expected {expected_samples} windows, processed {processed}")
    expected_symbols = processed * frames_per_window
    expected_transitions = processed * (frames_per_window - 1)
    if not torch.all(marginal_counts.sum(dim=1) == expected_symbols):
        raise RuntimeError("Marginal counts do not match encoded symbols")
    if not torch.all(initial_counts.sum(dim=1) == processed):
        raise RuntimeError("Initial counts do not match encoded windows")
    if not torch.all(transition_counts.sum(dim=1) == expected_transitions):
        raise RuntimeError("Transition counts do not match encoded transitions")

    return (
        marginal_counts.cpu(),
        initial_counts.cpu(),
        transition_counts.reshape(n_codebooks, codebook_size, codebook_size).cpu(),
        processed,
        frames_per_window,
    )


def entropy_from_counts(counts: np.ndarray, axis: int = -1) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=axis, keepdims=True)
    probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    log_probabilities = np.zeros_like(probabilities)
    np.log2(probabilities, out=log_probabilities, where=probabilities > 0)
    return -(probabilities * log_probabilities).sum(axis=axis)


def calibration_statistics(
    model: SeisDACLightning,
    marginal_counts: torch.Tensor,
    initial_counts: torch.Tensor,
    transition_counts: torch.Tensor,
    *,
    samples: int,
    frames_per_window: int,
    marginal_smoothing: float,
    backoff_concentration: float,
) -> Dict:
    marginal_np = marginal_counts.numpy().astype(np.float64, copy=False)
    initial_np = initial_counts.numpy().astype(np.float64, copy=False)
    transitions_np = transition_counts.numpy().astype(np.float64, copy=False)
    marginal_entropy = entropy_from_counts(marginal_np)
    conditional_entropy = entropy_from_counts(transitions_np)
    context_totals = transitions_np.sum(axis=-1)
    conditional_entropy_per_codebook = (
        conditional_entropy * context_totals
    ).sum(axis=-1) / context_totals.sum(axis=-1)

    first_order_model = model.generator.first_order_entropy_model
    marginal_cdf, conditional_cdf = first_order_model.quantized_cdfs()
    total = float(1 << first_order_model.cdf_precision)
    marginal_probabilities = np.diff(marginal_cdf.astype(np.int64), axis=-1) / total
    conditional_probabilities = np.diff(
        conditional_cdf.astype(np.int64),
        axis=-1,
    ) / total
    marginal_log_probabilities = -np.log2(marginal_probabilities)
    conditional_log_probabilities = -np.log2(conditional_probabilities)
    startup_bits = float((initial_np * marginal_log_probabilities).sum())
    transition_bits = float((transitions_np * conditional_log_probabilities).sum())

    duration_sec = int(model.config.data.window_length) / float(model.config.model.sample_rate)
    empirical_factorized_bps = float(
        marginal_entropy.sum() * frames_per_window / duration_sec
    )
    empirical_first_order_bps = float(
        (
            marginal_entropy.sum()
            + conditional_entropy_per_codebook.sum() * (frames_per_window - 1)
        )
        / duration_sec
    )
    cdf_first_order_bps = (startup_bits + transition_bits) / (samples * duration_sec)

    codebooks = []
    for idx in range(marginal_np.shape[0]):
        codebooks.append(
            {
                "codebook": idx + 1,
                "active_codes": int(np.count_nonzero(marginal_np[idx])),
                "active_contexts": int(np.count_nonzero(context_totals[idx])),
                "active_transitions": int(np.count_nonzero(transitions_np[idx])),
                "marginal_entropy_bits_per_symbol": float(marginal_entropy[idx]),
                "conditional_entropy_bits_per_symbol": float(
                    conditional_entropy_per_codebook[idx]
                ),
            }
        )

    return {
        "samples": samples,
        "frames_per_window": frames_per_window,
        "marginal_smoothing": marginal_smoothing,
        "backoff_concentration": backoff_concentration,
        "empirical_factorized_bps": empirical_factorized_bps,
        "empirical_first_order_bps": empirical_first_order_bps,
        "cdf_first_order_cross_entropy_bps": cdf_first_order_bps,
        "codebooks": codebooks,
    }


def save_calibrated_checkpoint(
    source_checkpoint: Path,
    output_checkpoint: Path,
    model: SeisDACLightning,
    metadata: Dict,
) -> None:
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = dict(checkpoint["state_dict"])
    checkpoint["state_dict"]["generator.entropy_model.logits"] = (
        model.generator.entropy_model.logits.detach().cpu().clone()
    )
    checkpoint["state_dict"]["generator.first_order_entropy_model.marginal_cdf"] = (
        model.generator.first_order_entropy_model.marginal_cdf.detach().cpu().clone()
    )
    checkpoint["state_dict"]["generator.first_order_entropy_model.conditional_cdf"] = (
        model.generator.first_order_entropy_model.conditional_cdf.detach().cpu().clone()
    )

    hyper_parameters = copy.deepcopy(checkpoint["hyper_parameters"])
    config = hyper_parameters["config"]
    config.model.use_entropy_model = True
    config.model.use_first_order_entropy_model = True
    config.model.entropy_cdf_precision = int(
        model.generator.first_order_entropy_model.cdf_precision
    )
    config.model.entropy_temperature = float(
        model.config.model.get("entropy_temperature", 0.1)
    )
    config.training.rate_loss_weight = 0.0
    hyper_parameters["config"] = config
    checkpoint["hyper_parameters"] = hyper_parameters
    checkpoint["first_order_entropy_calibration"] = metadata

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_checkpoint.with_suffix(output_checkpoint.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(output_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit causal same-codebook first-order PMFs from RVQ transition counts."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_checkpoint", default=DEFAULT_OUTPUT_CHECKPOINT)
    parser.add_argument("--data_name", default="ETHZ")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=22626,
        help="Maximum calibration windows; 0 uses the complete split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--marginal_smoothing", type=float, default=1.0)
    parser.add_argument("--backoff_concentration", type=float, default=32.0)
    parser.add_argument("--cdf_precision", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if not math.isfinite(args.marginal_smoothing) or args.marginal_smoothing <= 0:
        raise ValueError("marginal_smoothing must be finite and positive")
    if not math.isfinite(args.backoff_concentration) or args.backoff_concentration <= 0:
        raise ValueError("backoff_concentration must be finite and positive")

    seed_everything(args.seed)
    source_checkpoint = Path(args.checkpoint).resolve()
    output_checkpoint = Path(args.output_checkpoint).resolve()
    print(f"Loading source codec: {source_checkpoint}")
    model = SeisDACLightning.load_from_checkpoint(
        source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model.eval()
    add_entropy_models(model, args.cdf_precision)

    loader, sample_count = build_calibration_loader(
        model,
        data_name=args.data_name,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"Calibrating first-order PMFs on {sample_count} {args.data_name} "
        f"{args.split} windows"
    )
    model.generator.to(args.device)
    (
        marginal_counts,
        initial_counts,
        transition_counts,
        processed,
        frames_per_window,
    ) = collect_transition_counts(
        model,
        loader,
        device=args.device,
        expected_samples=sample_count,
    )
    model.generator.to("cpu")
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    model.generator.entropy_model.calibrate_from_counts(
        marginal_counts,
        smoothing=args.marginal_smoothing,
    )
    model.generator.first_order_entropy_model.calibrate_from_counts(
        marginal_counts,
        transition_counts,
        marginal_smoothing=args.marginal_smoothing,
        backoff_concentration=args.backoff_concentration,
    )
    statistics = calibration_statistics(
        model,
        marginal_counts,
        initial_counts,
        transition_counts,
        samples=processed,
        frames_per_window=frames_per_window,
        marginal_smoothing=args.marginal_smoothing,
        backoff_concentration=args.backoff_concentration,
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_checkpoint),
        "data_name": args.data_name,
        "split": args.split,
        "seed": args.seed,
        "augmentation_pipeline": "source checkpoint validation augmentations",
        "cdf_precision": args.cdf_precision,
        **statistics,
    }
    save_calibrated_checkpoint(
        source_checkpoint,
        output_checkpoint,
        model,
        metadata,
    )

    summary_path = output_checkpoint.parent / "first_order_calibration_summary.json"
    counts_path = output_checkpoint.parent / "first_order_calibration_counts.npz"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        counts_path,
        marginal_counts=marginal_counts.numpy(),
        initial_counts=initial_counts.numpy(),
        transition_counts=transition_counts.numpy(),
    )
    print(
        "Train empirical rates: "
        f"factorized={statistics['empirical_factorized_bps']:.2f} bps, "
        f"first_order={statistics['empirical_first_order_bps']:.2f} bps"
    )
    print(
        "Train first-order CDF cross-entropy: "
        f"{statistics['cdf_first_order_cross_entropy_bps']:.2f} bps"
    )
    print(f"Saved checkpoint: {output_checkpoint}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved counts: {counts_path}")


if __name__ == "__main__":
    main()
