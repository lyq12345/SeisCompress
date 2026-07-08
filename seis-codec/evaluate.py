"""Evaluate SeisDAC reconstruction quality from a Lightning checkpoint."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
import torch
import torch.nn.functional as F

from train import SeisDACLightning


def compute_snr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    noise = reference - estimate
    signal_power = torch.sum(reference ** 2)
    noise_power = torch.sum(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return (10.0 * torch.log10(signal_power / noise_power)).item()


def compute_bitrate_bps(
    *,
    n_samples: int,
    n_codebooks: int,
    codebook_size: int,
    hop_length: int,
    sample_rate: int,
    n_channels: int = 3,
    bits_per_sample: int = 32,
) -> Tuple[float, float]:
    """Return (compressed_bps, compression_ratio vs float32 waveform)."""
    latent_frames = int(np.ceil(n_samples / hop_length))
    bits_per_frame = n_codebooks * np.log2(codebook_size)
    compressed_bps = bits_per_frame * (sample_rate / hop_length)
    original_bps = n_channels * bits_per_sample * sample_rate
    return compressed_bps, original_bps / compressed_bps


@torch.no_grad()
def evaluate_batch(
    model: SeisDACLightning,
    waveforms: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    waveforms = waveforms.to(model.device)
    out = model.generator(waveforms, sample_rate=model.config.model.sample_rate)
    reconstructed = out["audio"]
    return {
        "real": waveforms,
        "fake": reconstructed,
        "codes": out["codes"],
    }


def plot_waveforms(
    real: np.ndarray,
    fake: np.ndarray,
    output_path: Path,
    sample_rate: int,
    title: str,
) -> None:
    """Plot Z/N/E components for one example. Arrays are (C, T)."""
    component_names = ["Z", "N", "E"]
    n_components = min(real.shape[0], 3)
    time_axis = np.arange(real.shape[-1]) / sample_rate

    fig, axes = plt.subplots(n_components, 1, figsize=(12, 2.5 * n_components), sharex=True)
    if n_components == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.plot(time_axis, real[idx], label="original", linewidth=0.8, alpha=0.9)
        ax.plot(time_axis, fake[idx], label="reconstructed", linewidth=0.8, alpha=0.8)
        ax.set_ylabel(component_names[idx])
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right")
            ax.set_title(title)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_dev_generator(model: SeisDACLightning, data_name: str):
    dataset = getattr(sbd, data_name)(
        sampling_rate=model.config.model.sample_rate,
        component_order="ZNE",
        dimension_order="NCW",
    )
    generator = sbg.GenericGenerator(dataset.dev())
    generator.add_augmentations(model.get_val_augmentations())
    return generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SeisDAC checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="lightning_logs/version_4/checkpoints/epoch=99-step=141400.ckpt",
        help="Path to Lightning checkpoint.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="ETHZ",
        help="SeisBench dataset name for evaluation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of windows to evaluate.",
    )
    parser.add_argument(
        "--num_plots",
        type=int,
        default=3,
        help="Number of waveform comparison figures to save.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_outputs",
        help="Directory for plots and summary.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = SeisDACLightning.load_from_checkpoint(
        args.checkpoint,
        map_location=args.device,
        weights_only=False,
    )
    model.eval()
    model.to(args.device)

    generator = load_dev_generator(model, args.data_name)
    indices = np.random.default_rng(42).choice(
        len(generator), size=min(args.num_samples, len(generator)), replace=False
    )

    l1_values: List[float] = []
    snr_values: List[float] = []
    plot_examples: List[Tuple[np.ndarray, np.ndarray, float, float]] = []

    generator_model = model.generator
    sample_rate = model.config.model.sample_rate
    compressed_bps, compression_ratio = compute_bitrate_bps(
        n_samples=model.config.data.window_length,
        n_codebooks=generator_model.n_codebooks,
        codebook_size=generator_model.codebook_size,
        hop_length=generator_model.hop_length,
        sample_rate=sample_rate,
        n_channels=model.config.model.in_channels,
    )

    for i, idx in enumerate(indices):
        sample = generator[int(idx)]
        waveforms = torch.from_numpy(sample["X"]).unsqueeze(0).float()
        result = evaluate_batch(model, waveforms)

        real = result["real"]
        fake = result["fake"]
        l1 = F.l1_loss(fake, real).item()
        snr = compute_snr(real, fake)
        l1_values.append(l1)
        snr_values.append(snr)

        if len(plot_examples) < args.num_plots:
            plot_examples.append((real[0].cpu().numpy(), fake[0].cpu().numpy(), l1, snr))

    print("\n=== Reconstruction metrics ===")
    print(f"Dataset: {args.data_name} (dev split)")
    print(f"Samples: {len(l1_values)}")
    print(f"L1  mean: {np.mean(l1_values):.4f}  std: {np.std(l1_values):.4f}")
    print(f"L1  min : {np.min(l1_values):.4f}  max: {np.max(l1_values):.4f}")
    print(f"SNR mean: {np.mean(snr_values):.2f} dB  std: {np.std(snr_values):.2f} dB")
    print(f"SNR min : {np.min(snr_values):.2f} dB  max: {np.max(snr_values):.2f} dB")

    print("\n=== Compression (theoretical, from codec codes) ===")
    print(f"Compressed bitrate: ~{compressed_bps:.1f} bps")
    print(f"Compression ratio vs float32 ZNE: ~{compression_ratio:.0f}x")

    summary_path = output_dir / "metrics.txt"
    summary_path.write_text(
        "\n".join([
            f"checkpoint: {args.checkpoint}",
            f"dataset: {args.data_name}",
            f"samples: {len(l1_values)}",
            f"l1_mean: {np.mean(l1_values):.6f}",
            f"l1_std: {np.std(l1_values):.6f}",
            f"snr_mean_db: {np.mean(snr_values):.4f}",
            f"snr_std_db: {np.std(snr_values):.4f}",
            f"compressed_bps: {compressed_bps:.4f}",
            f"compression_ratio_vs_float32: {compression_ratio:.4f}",
        ]) + "\n",
        encoding="utf-8",
    )

    for plot_idx, (real, fake, l1, snr) in enumerate(plot_examples):
        plot_path = output_dir / f"waveform_{plot_idx:02d}.png"
        plot_waveforms(
            real,
            fake,
            plot_path,
            sample_rate=sample_rate,
            title=f"{args.data_name} example {plot_idx} | L1={l1:.4f} | SNR={snr:.2f} dB",
        )
        print(f"Saved plot: {plot_path}")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
