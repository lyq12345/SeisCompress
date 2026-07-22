"""Summarize cross-dataset SeisDAC evaluation outputs."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


DISPLAY_NAMES = {
    "ethz": "ETHZ",
    "stead": "STEAD",
    "geofon": "GEOFON",
    "instance_counts_combined": "INSTANCE",
    "iquique": "Iquique",
    "m_l_a_a_p_d_e": "MLAAPDE",
    "mlaapde": "MLAAPDE",
    "p_n_w": "PNW",
    "pnw": "PNW",
    "o_b_s_t2024": "OBST2024",
    "obst2024": "OBST2024",
}


def parse_metrics(path: Path) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            values[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return values


def dataset_key_from_recon_dir(path: Path) -> str:
    suffix = "_nogan_latreg0p003_best137"
    name = path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def dataset_key_from_picking_dir(path: Path) -> str:
    suffix = "_nogan_latreg0p003_picking"
    name = path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def display_name(key: str) -> str:
    return DISPLAY_NAMES.get(key, key)


def load_reconstruction(out_root: Path) -> Dict[str, Dict[str, float]]:
    results = {}
    for metrics_path in sorted(out_root.glob("*_nogan_latreg0p003_best137/metrics.txt")):
        key = dataset_key_from_recon_dir(metrics_path.parent)
        results[key] = parse_metrics(metrics_path)
    return results


def load_picking(out_root: Path) -> Dict[str, Dict]:
    results = {}
    for metrics_path in sorted(out_root.glob("*_nogan_latreg0p003_picking/picking_metrics.json")):
        key = dataset_key_from_picking_dir(metrics_path.parent)
        results[key] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return results


def fmt(value: Optional[float], digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def build_markdown(reconstruction: Dict[str, Dict], picking: Dict[str, Dict]) -> str:
    lines = []
    lines.append("# Cross-Dataset Evaluation Summary")
    lines.append("")
    lines.append("## Reconstruction")
    lines.append("")
    lines.append("| Dataset | Samples | L1 mean | L1 std | SNR mean (dB) | SNR std (dB) | Bitrate (bps) | Ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(reconstruction):
        rec = reconstruction[key]
        lines.append(
            f"| {display_name(key)} | "
            f"{int(rec.get('samples', 0))} | "
            f"{fmt(rec.get('l1_mean'), 4)} | "
            f"{fmt(rec.get('l1_std'), 4)} | "
            f"{fmt(rec.get('snr_mean_db'), 2)} | "
            f"{fmt(rec.get('snr_std_db'), 2)} | "
            f"{fmt(rec.get('compressed_bps'), 0)} | "
            f"{fmt(rec.get('compression_ratio_vs_float32'), 1)}x |"
        )

    lines.append("")
    lines.append("## Phase Picking")
    lines.append("")
    lines.append("| Dataset | Phase | N | Original recall | Reconstructed recall | Delta recall (pp) | Original MAE (s) | Reconstructed MAE (s) | Delta MAE (ms) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(picking):
        pick = picking[key]
        for phase in ["P", "S"]:
            phase_metrics = pick.get("phases", {}).get(phase)
            degradation = pick.get("degradation", {}).get(phase)
            if not phase_metrics or not degradation:
                continue
            original = phase_metrics["original"]
            reconstructed = phase_metrics["reconstructed"]
            lines.append(
                f"| {display_name(key)} | {phase} | "
                f"{original['n']} | "
                f"{original['recall_at_tolerance']:.3f} | "
                f"{reconstructed['recall_at_tolerance']:.3f} | "
                f"{100.0 * degradation['delta_recall_at_tolerance']:+.1f} | "
                f"{original['mae_sec']:.4f} | "
                f"{reconstructed['mae_sec']:.4f} | "
                f"{1000.0 * degradation['delta_mae_sec']:+.1f} |"
            )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize cross-dataset SeisDAC outputs.")
    parser.add_argument(
        "--out_root",
        default="/data/seismic/seis-codec-eval",
        help="Directory containing *_nogan_latreg0p003_* evaluation outputs.",
    )
    parser.add_argument(
        "--output",
        default="/data/seismic/seis-codec-eval/cross_dataset_summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    reconstruction = load_reconstruction(out_root)
    picking = load_picking(out_root)
    markdown = build_markdown(reconstruction, picking)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"Saved summary: {output}")


if __name__ == "__main__":
    main()
