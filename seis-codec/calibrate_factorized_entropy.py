"""Calibrate a factorized RVQ entropy model from training-split code counts."""

import argparse
import copy
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
import torch
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader, Subset

from evaluate import ensure_dataset_split
from factorized_entropy import FactorizedCategoricalEntropyModel
from train import SeisDACLightning, waveform_collator


DEFAULT_CHECKPOINT = (
    "/data/seismic/seis-codec-logs/ethz_nogan_spectral/"
    "seislm_enc_peaknorm_latreg0p003_nogan_200ep_freeze/"
    "checkpoints/best-137-24288.ckpt"
)
DEFAULT_OUTPUT_CHECKPOINT = (
    "/data/seismic/seis-codec-logs/ethz_nogan_spectral/"
    "seislm_enc_peaknorm_latreg0p003_nogan_200ep_freeze_entropy_calibrated/"
    "checkpoints/calibrated_train_pmf.ckpt"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_entropy_model(model: SeisDACLightning, cdf_precision: int) -> None:
    generator = model.generator
    if generator.entropy_model is None:
        generator.entropy_model = FactorizedCategoricalEntropyModel(
            n_codebooks=int(generator.n_codebooks),
            codebook_size=int(generator.codebook_size),
            cdf_precision=cdf_precision,
        )
    elif generator.entropy_model.cdf_precision != cdf_precision:
        generator.entropy_model.cdf_precision = int(cdf_precision)

    model.use_entropy_model = True
    model.config.model.use_entropy_model = True
    model.config.model.entropy_cdf_precision = int(cdf_precision)
    model.config.model.entropy_temperature = float(
        model.config.model.get("entropy_temperature", 0.1)
    )
    model.config.training.rate_loss_weight = 0.0


def build_calibration_loader(
    model: SeisDACLightning,
    *,
    data_name: str,
    split: str,
    max_samples: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, int]:
    dataset_cls = getattr(sbd, data_name)
    dataset = dataset_cls(
        sampling_rate=int(model.config.model.sample_rate),
        component_order="ZNE",
        dimension_order="NCW",
        cache=model.config.data.get("cache_dataset", None),
    )
    ensure_dataset_split(dataset)
    split_dataset = dataset.get_split(split)
    generator = sbg.GenericGenerator(split_dataset)
    # Deployment evaluation uses this exact windowing and normalization path.
    generator.add_augmentations(model.get_val_augmentations())

    sample_count = len(generator)
    if max_samples > 0 and max_samples < sample_count:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(sample_count, size=max_samples, replace=False))
        calibration_dataset = Subset(generator, indices.tolist())
        sample_count = int(max_samples)
    else:
        calibration_dataset = generator

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "collate_fn": waveform_collator,
    }
    if num_workers > 0:
        loader_kwargs["worker_init_fn"] = worker_seeding
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(calibration_dataset, **loader_kwargs)
    return loader, sample_count


@torch.inference_mode()
def collect_code_counts(
    model: SeisDACLightning,
    loader: DataLoader,
    *,
    device: str,
    expected_samples: int,
) -> Tuple[torch.Tensor, int, int]:
    generator = model.generator.eval().to(device)
    n_codebooks = int(generator.n_codebooks)
    codebook_size = int(generator.codebook_size)
    counts = torch.zeros(n_codebooks, codebook_size, dtype=torch.int64)
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
            counts[codebook_idx] += torch.bincount(
                codes[:, codebook_idx, :].reshape(-1).cpu(),
                minlength=codebook_size,
            )
        processed += int(codes.shape[0])
        if (batch_idx + 1) % 50 == 0 or processed == expected_samples:
            print(f"Collected {processed}/{expected_samples} calibration windows")

    if processed != expected_samples or frames_per_window is None:
        raise RuntimeError(f"Expected {expected_samples} windows, processed {processed}")
    expected_symbols = processed * frames_per_window
    if not torch.all(counts.sum(dim=1) == expected_symbols):
        raise RuntimeError("Code counts do not match the number of encoded symbols")
    generator.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return counts, processed, frames_per_window


def entropy_bits(probabilities: torch.Tensor) -> torch.Tensor:
    terms = torch.where(
        probabilities > 0,
        -probabilities * torch.log2(probabilities),
        torch.zeros_like(probabilities),
    )
    return terms.sum(dim=-1)


def calibration_statistics(
    model: SeisDACLightning,
    counts: torch.Tensor,
    *,
    samples: int,
    frames_per_window: int,
    smoothing: float,
) -> Dict:
    counts_f64 = counts.double()
    empirical_probabilities = counts_f64 / counts_f64.sum(dim=-1, keepdim=True)
    model_probabilities = model.generator.entropy_model.probabilities().detach().double().cpu()
    empirical_entropy = entropy_bits(empirical_probabilities)
    cross_entropy = -(
        empirical_probabilities * torch.log2(model_probabilities)
    ).sum(dim=-1)
    cdf = model.generator.entropy_model.quantized_cdf()
    cdf_probabilities = torch.from_numpy(np.diff(cdf.astype(np.int64), axis=1)).double()
    cdf_probabilities /= float(1 << model.generator.entropy_model.cdf_precision)
    cdf_cross_entropy = -(
        empirical_probabilities * torch.log2(cdf_probabilities)
    ).sum(dim=-1)

    window_length = int(model.config.data.window_length)
    sample_rate = int(model.config.model.sample_rate)
    duration_sec = window_length / float(sample_rate)
    frames_per_second = frames_per_window / duration_sec
    codebooks = []
    for idx in range(counts.shape[0]):
        codebooks.append(
            {
                "codebook": idx + 1,
                "symbols": int(counts[idx].sum()),
                "active_codes": int((counts[idx] > 0).sum()),
                "empirical_entropy_bits_per_symbol": float(empirical_entropy[idx]),
                "calibrated_cross_entropy_bits_per_symbol": float(cross_entropy[idx]),
                "cdf_cross_entropy_bits_per_symbol": float(cdf_cross_entropy[idx]),
                "max_symbol_probability": float(empirical_probabilities[idx].max()),
            }
        )

    return {
        "samples": samples,
        "frames_per_window": frames_per_window,
        "symbols_per_codebook": int(samples * frames_per_window),
        "smoothing": smoothing,
        "empirical_entropy_bits_per_symbol_sum": float(empirical_entropy.sum()),
        "calibrated_cross_entropy_bits_per_symbol_sum": float(cross_entropy.sum()),
        "cdf_cross_entropy_bits_per_symbol_sum": float(cdf_cross_entropy.sum()),
        "empirical_entropy_bps": float(empirical_entropy.sum() * frames_per_second),
        "calibrated_cross_entropy_bps": float(cross_entropy.sum() * frames_per_second),
        "cdf_cross_entropy_bps": float(cdf_cross_entropy.sum() * frames_per_second),
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

    hyper_parameters = copy.deepcopy(checkpoint["hyper_parameters"])
    config = hyper_parameters["config"]
    config.model.use_entropy_model = True
    config.model.entropy_cdf_precision = int(model.generator.entropy_model.cdf_precision)
    config.model.entropy_temperature = float(
        model.config.model.get("entropy_temperature", 0.1)
    )
    config.training.rate_loss_weight = 0.0
    hyper_parameters["config"] = config
    checkpoint["hyper_parameters"] = hyper_parameters
    checkpoint["entropy_calibration"] = metadata

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_checkpoint.with_suffix(output_checkpoint.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(output_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit static per-codebook categorical PMFs from RVQ code counts."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_checkpoint", default=DEFAULT_OUTPUT_CHECKPOINT)
    parser.add_argument("--data_name", default="ETHZ")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum calibration windows; 0 uses the complete split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--cdf_precision", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if not math.isfinite(args.smoothing) or args.smoothing <= 0:
        raise ValueError("smoothing must be finite and positive")

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
    add_entropy_model(model, args.cdf_precision)

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
        f"Calibrating on {sample_count} {args.data_name} {args.split} windows "
        f"with batch_size={args.batch_size}"
    )
    counts, processed, frames_per_window = collect_code_counts(
        model,
        loader,
        device=args.device,
        expected_samples=sample_count,
    )
    model.generator.entropy_model.calibrate_from_counts(
        counts,
        smoothing=args.smoothing,
    )
    statistics = calibration_statistics(
        model,
        counts,
        samples=processed,
        frames_per_window=frames_per_window,
        smoothing=args.smoothing,
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

    summary_path = output_checkpoint.parent / "calibration_summary.json"
    counts_path = output_checkpoint.parent / "calibration_counts.npz"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(counts_path, counts=counts.numpy())

    print(
        "Train empirical entropy: "
        f"{statistics['empirical_entropy_bps']:.2f} bps"
    )
    print(
        "Calibrated cross-entropy: "
        f"{statistics['calibrated_cross_entropy_bps']:.2f} bps"
    )
    print(f"Saved checkpoint: {output_checkpoint}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved counts: {counts_path}")


if __name__ == "__main__":
    main()
