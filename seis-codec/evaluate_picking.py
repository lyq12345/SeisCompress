"""Evaluate downstream phase picking after SeisDAC reconstruction.

The script compares a pretrained SeisBench picker on original waveforms and on
codec reconstructions. Metrics are computed at labelled P/S arrivals in the
same ETHZ-style windows used for reconstruction evaluation.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import seisbench.data as sbd
import seisbench.models as sbm
import torch

from train import SeisDACLightning, phase_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dataset_split(dataset) -> None:
    """Add a deterministic auxiliary split if the dataset metadata has none."""
    columns = getattr(getattr(dataset, "metadata", None), "columns", [])
    if "split" in columns:
        return

    split = np.array(["train"] * len(dataset), dtype=object)
    split[int(0.6 * len(dataset)) : int(0.7 * len(dataset))] = "dev"
    split[int(0.7 * len(dataset)) :] = "test"
    dataset._metadata["split"] = split  # pylint: disable=protected-access
    print(
        "Dataset has no split; added auxiliary split "
        f"(train={int(np.sum(split == 'train'))}, "
        f"dev={int(np.sum(split == 'dev'))}, "
        f"test={int(np.sum(split == 'test'))})."
    )


def load_split(dataset, split_name: str):
    if split_name == "train":
        return dataset.train()
    if split_name == "dev":
        return dataset.dev()
    if split_name == "test":
        return dataset.test()
    raise ValueError(f"Unknown split '{split_name}'. Expected train/dev/test.")


def load_windowed_state(model: SeisDACLightning, dataset, idx: int) -> Dict:
    """Apply the model validation augmentations while preserving metadata."""
    state = {"X": dataset.get_sample(idx)}
    for augmentation in model.get_val_augmentations():
        augmentation(state)
    return state


def scalar_values(value) -> Iterable[float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    for item in arr:
        if np.isfinite(item):
            yield float(item)


def collect_phase_targets(
    metadata: Dict,
    *,
    window_length: int,
    dedup_samples: float = 1.0,
) -> List[Tuple[str, float, str]]:
    """Return labelled P/S targets inside the current window.

    Each target is (phase, sample_index, source_metadata_key). Multiple phase
    labels mapping to the same P/S sample are de-duplicated.
    """
    candidates: List[Tuple[str, float, str]] = []
    for key, phase in phase_dict.items():
        if phase not in {"P", "S"} or key not in metadata:
            continue
        for sample in scalar_values(metadata[key]):
            if 0 <= sample < window_length:
                candidates.append((phase, sample, key))

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    targets: List[Tuple[str, float, str]] = []
    last_by_phase: Dict[str, float] = {}
    for phase, sample, key in candidates:
        last_sample = last_by_phase.get(phase)
        if last_sample is not None and abs(sample - last_sample) <= dedup_samples:
            continue
        targets.append((phase, sample, key))
        last_by_phase[phase] = sample

    return targets


def load_picker(args) -> torch.nn.Module:
    picker_name = args.picker.lower()
    if picker_name == "phasenet":
        picker = sbm.PhaseNet.from_pretrained(
            args.picker_weights,
            update=args.picker_update,
        )
    elif picker_name in {"eqtransformer", "eqt"}:
        picker = sbm.EQTransformer.from_pretrained(
            args.picker_weights,
            update=args.picker_update,
        )
    else:
        raise ValueError("Supported pickers: phasenet, eqtransformer")
    return picker


def picker_labels(picker: torch.nn.Module) -> List[str]:
    labels = getattr(picker, "labels", None)
    if labels is None:
        raise ValueError("Picker does not expose labels; cannot locate P/S channels.")
    return list(labels)


def preprocess_for_picker(picker: torch.nn.Module, waveforms: torch.Tensor) -> torch.Tensor:
    """Use SeisBench preprocessing when available, with a conservative fallback."""
    if hasattr(picker, "annotate_batch_pre"):
        try:
            return picker.annotate_batch_pre(waveforms, {})
        except TypeError:
            return picker.annotate_batch_pre(waveforms)

    waveforms = waveforms - waveforms.mean(dim=-1, keepdim=True)
    scale = waveforms.std(dim=-1, keepdim=True)
    return waveforms / (scale + 1e-10)


@torch.no_grad()
def predict_picker_probs(
    picker: torch.nn.Module,
    waveforms: torch.Tensor,
) -> torch.Tensor:
    x = preprocess_for_picker(picker, waveforms)
    probs = picker(x)
    if isinstance(probs, tuple):
        probs = probs[0]
    if probs.ndim != 3:
        raise ValueError(f"Expected picker output with shape (B, C, T), got {tuple(probs.shape)}")
    return probs


def evaluate_local_pick(
    prob_curve: np.ndarray,
    *,
    true_sample: float,
    input_length: int,
    sample_rate: float,
    search_radius_sec: float,
) -> Dict[str, float]:
    """Find the strongest picker response near a labelled arrival."""
    output_length = len(prob_curve)
    if output_length <= 1 or input_length <= 1:
        scale = 1.0
    else:
        scale = (output_length - 1) / (input_length - 1)

    true_output_sample = int(round(true_sample * scale))
    true_output_sample = int(np.clip(true_output_sample, 0, output_length - 1))
    search_radius_output = max(1, int(round(search_radius_sec * sample_rate * scale)))
    lo = max(0, true_output_sample - search_radius_output)
    hi = min(output_length, true_output_sample + search_radius_output + 1)

    local = prob_curve[lo:hi]
    pred_output_sample = lo + int(np.nanargmax(local))
    pred_sample = pred_output_sample / scale if scale != 0 else float(pred_output_sample)
    signed_error_sec = (pred_sample - true_sample) / sample_rate

    return {
        "pred_sample": float(pred_sample),
        "signed_error_sec": float(signed_error_sec),
        "abs_error_sec": float(abs(signed_error_sec)),
        "peak_prob": float(prob_curve[pred_output_sample]),
        "prob_at_true": float(prob_curve[true_output_sample]),
    }


def summarize_phase(
    rows: List[Dict],
    *,
    phase: str,
    domain: str,
    tolerance_sec: float,
    prob_threshold: float,
) -> Dict[str, Optional[float]]:
    subset = [r for r in rows if r["phase"] == phase]
    prefix = f"{domain}_"
    values = {
        "n": len(subset),
        "mae_sec": None,
        "median_abs_error_sec": None,
        "rmse_sec": None,
        "recall_at_tolerance": None,
        "mean_peak_prob": None,
        "median_peak_prob": None,
    }
    if not subset:
        return values

    abs_errors = np.asarray([r[f"{prefix}abs_error_sec"] for r in subset], dtype=float)
    signed_errors = np.asarray([r[f"{prefix}signed_error_sec"] for r in subset], dtype=float)
    peak_probs = np.asarray([r[f"{prefix}peak_prob"] for r in subset], dtype=float)
    hits = (abs_errors <= tolerance_sec) & (peak_probs >= prob_threshold)

    values.update(
        {
            "mae_sec": float(abs_errors.mean()),
            "median_abs_error_sec": float(np.median(abs_errors)),
            "rmse_sec": float(np.sqrt(np.mean(signed_errors**2))),
            "recall_at_tolerance": float(hits.mean()),
            "mean_peak_prob": float(peak_probs.mean()),
            "median_peak_prob": float(np.median(peak_probs)),
        }
    )
    return values


def summarize(rows: List[Dict], args) -> Dict:
    summary = {
        "settings": {
            "tolerance_sec": args.tolerance_sec,
            "prob_threshold": args.prob_threshold,
            "search_radius_sec": args.search_radius_sec,
        },
        "phases": {},
        "degradation": {},
    }

    for phase in ["P", "S"]:
        summary["phases"][phase] = {
            "original": summarize_phase(
                rows,
                phase=phase,
                domain="original",
                tolerance_sec=args.tolerance_sec,
                prob_threshold=args.prob_threshold,
            ),
            "reconstructed": summarize_phase(
                rows,
                phase=phase,
                domain="reconstructed",
                tolerance_sec=args.tolerance_sec,
                prob_threshold=args.prob_threshold,
            ),
        }

        subset = [r for r in rows if r["phase"] == phase]
        if subset:
            summary["degradation"][phase] = {
                "delta_mae_sec": (
                    summary["phases"][phase]["reconstructed"]["mae_sec"]
                    - summary["phases"][phase]["original"]["mae_sec"]
                ),
                "delta_recall_at_tolerance": (
                    summary["phases"][phase]["reconstructed"]["recall_at_tolerance"]
                    - summary["phases"][phase]["original"]["recall_at_tolerance"]
                ),
                "mean_picker_shift_sec": float(
                    np.mean([r["picker_shift_sec"] for r in subset])
                ),
                "median_picker_shift_sec": float(
                    np.median([r["picker_shift_sec"] for r in subset])
                ),
                "mean_peak_prob_drop": float(
                    np.mean([r["peak_prob_drop"] for r in subset])
                ),
            }
        else:
            summary["degradation"][phase] = None

    return summary


def format_optional(value: Optional[float], fmt: str) -> str:
    if value is None:
        return "n/a"
    return format(value, fmt)


def summary_lines(summary: Dict) -> List[str]:
    lines = ["=== Downstream phase picking ==="]
    settings = summary["settings"]
    lines.append(
        "tolerance: "
        f"{settings['tolerance_sec']:.3f}s, "
        f"prob_threshold: {settings['prob_threshold']:.3f}, "
        f"search_radius: {settings['search_radius_sec']:.3f}s"
    )
    lines.append("")

    for phase in ["P", "S"]:
        lines.append(f"{phase} phase")
        for domain in ["original", "reconstructed"]:
            metrics = summary["phases"][phase][domain]
            lines.append(
                f"  {domain:13s} n={metrics['n']:5d} "
                f"MAE={format_optional(metrics['mae_sec'], '.4f')}s "
                f"median={format_optional(metrics['median_abs_error_sec'], '.4f')}s "
                f"recall={format_optional(metrics['recall_at_tolerance'], '.3f')} "
                f"mean_prob={format_optional(metrics['mean_peak_prob'], '.3f')}"
            )

        degradation = summary["degradation"][phase]
        if degradation is None:
            lines.append("  degradation   n/a")
        else:
            lines.append(
                "  degradation   "
                f"delta_MAE={degradation['delta_mae_sec']:+.4f}s "
                f"delta_recall={degradation['delta_recall_at_tolerance']:+.3f} "
                f"picker_shift={degradation['mean_picker_shift_sec']:.4f}s "
                f"prob_drop={degradation['mean_peak_prob_drop']:+.3f}"
            )
        lines.append("")

    return lines


def write_csv(path: Path, rows: List[Dict]) -> None:
    fieldnames = [
        "sample_index",
        "trace_name",
        "phase",
        "phase_key",
        "true_sample",
        "original_pred_sample",
        "reconstructed_pred_sample",
        "original_signed_error_sec",
        "reconstructed_signed_error_sec",
        "original_abs_error_sec",
        "reconstructed_abs_error_sec",
        "original_peak_prob",
        "reconstructed_peak_prob",
        "original_prob_at_true",
        "reconstructed_prob_at_true",
        "picker_shift_sec",
        "peak_prob_drop",
        "original_hit",
        "reconstructed_hit",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate phase-picking degradation after SeisDAC compression."
    )
    parser.add_argument("--checkpoint", required=True, help="SeisDAC Lightning checkpoint.")
    parser.add_argument("--data_name", default="ETHZ", help="SeisBench dataset name.")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--picker", default="phasenet", choices=["phasenet", "eqtransformer", "eqt"])
    parser.add_argument("--picker_weights", default="ethz")
    parser.add_argument("--picker_update", action="store_true")
    parser.add_argument("--prob_threshold", type=float, default=0.3)
    parser.add_argument("--tolerance_sec", type=float, default=0.2)
    parser.add_argument("--search_radius_sec", type=float, default=1.0)
    parser.add_argument("--output_dir", default="picking_evaluation_outputs")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print(f"Loading codec checkpoint: {args.checkpoint}")
    codec = SeisDACLightning.load_from_checkpoint(
        args.checkpoint,
        map_location=args.device,
        weights_only=False,
    )
    codec.eval()
    codec.to(args.device)

    print(f"Loading picker: {args.picker} weights={args.picker_weights}")
    picker = load_picker(args)
    picker.eval()
    picker.to(args.device)
    labels = picker_labels(picker)
    label_to_channel = {label: i for i, label in enumerate(labels)}
    missing = {"P", "S"} - set(label_to_channel)
    if missing:
        raise ValueError(f"Picker labels {labels} do not contain {sorted(missing)}.")

    sample_rate = float(codec.config.model.sample_rate)
    picker_sample_rate = getattr(picker, "sampling_rate", sample_rate)
    if picker_sample_rate and abs(float(picker_sample_rate) - sample_rate) > 1e-6:
        raise ValueError(
            f"Codec sample rate is {sample_rate} Hz, but picker expects "
            f"{picker_sample_rate} Hz. Use compatible dataset/model settings."
        )

    dataset_cls = getattr(sbd, args.data_name)
    dataset = dataset_cls(
        sampling_rate=sample_rate,
        component_order="ZNE",
        dimension_order="NCW",
    )
    ensure_dataset_split(dataset)
    split_dataset = load_split(dataset, args.split)

    rng = np.random.default_rng(args.seed)
    max_attempts = args.max_attempts or min(len(split_dataset), args.num_samples * 5)
    candidate_indices = rng.choice(
        len(split_dataset),
        size=min(max_attempts, len(split_dataset)),
        replace=False,
    )

    rows: List[Dict] = []
    evaluated_windows = 0
    skipped_without_targets = 0

    for idx in candidate_indices:
        if evaluated_windows >= args.num_samples:
            break

        state = load_windowed_state(codec, split_dataset, int(idx))
        wave_np, metadata = state["X"]
        wave_np = np.asarray(wave_np, dtype=np.float32)
        if wave_np.ndim != 2:
            raise ValueError(f"Expected waveform shape (C, T), got {wave_np.shape}")

        targets = collect_phase_targets(metadata, window_length=wave_np.shape[-1])
        if not targets:
            skipped_without_targets += 1
            continue

        waveforms = torch.from_numpy(wave_np).unsqueeze(0).to(args.device)
        with torch.no_grad():
            reconstructed = codec.generator(
                waveforms,
                sample_rate=codec.config.model.sample_rate,
            )["audio"]

            original_probs = predict_picker_probs(picker, waveforms.float())
            reconstructed_probs = predict_picker_probs(picker, reconstructed.float())

        original_probs_np = original_probs[0].detach().cpu().numpy()
        reconstructed_probs_np = reconstructed_probs[0].detach().cpu().numpy()
        input_length = wave_np.shape[-1]
        trace_name = metadata.get("trace_name", "")

        for phase, true_sample, phase_key in targets:
            channel = label_to_channel[phase]
            original = evaluate_local_pick(
                original_probs_np[channel],
                true_sample=true_sample,
                input_length=input_length,
                sample_rate=sample_rate,
                search_radius_sec=args.search_radius_sec,
            )
            reconstructed_metric = evaluate_local_pick(
                reconstructed_probs_np[channel],
                true_sample=true_sample,
                input_length=input_length,
                sample_rate=sample_rate,
                search_radius_sec=args.search_radius_sec,
            )
            original_hit = (
                original["abs_error_sec"] <= args.tolerance_sec
                and original["peak_prob"] >= args.prob_threshold
            )
            reconstructed_hit = (
                reconstructed_metric["abs_error_sec"] <= args.tolerance_sec
                and reconstructed_metric["peak_prob"] >= args.prob_threshold
            )

            row = {
                "sample_index": int(idx),
                "trace_name": trace_name,
                "phase": phase,
                "phase_key": phase_key,
                "true_sample": float(true_sample),
                "original_pred_sample": original["pred_sample"],
                "reconstructed_pred_sample": reconstructed_metric["pred_sample"],
                "original_signed_error_sec": original["signed_error_sec"],
                "reconstructed_signed_error_sec": reconstructed_metric[
                    "signed_error_sec"
                ],
                "original_abs_error_sec": original["abs_error_sec"],
                "reconstructed_abs_error_sec": reconstructed_metric["abs_error_sec"],
                "original_peak_prob": original["peak_prob"],
                "reconstructed_peak_prob": reconstructed_metric["peak_prob"],
                "original_prob_at_true": original["prob_at_true"],
                "reconstructed_prob_at_true": reconstructed_metric["prob_at_true"],
                "picker_shift_sec": abs(
                    reconstructed_metric["pred_sample"] - original["pred_sample"]
                )
                / sample_rate,
                "peak_prob_drop": original["peak_prob"]
                - reconstructed_metric["peak_prob"],
                "original_hit": bool(original_hit),
                "reconstructed_hit": bool(reconstructed_hit),
            }
            rows.append(row)

        evaluated_windows += 1
        if evaluated_windows % 100 == 0:
            print(f"Processed {evaluated_windows} labelled windows...")

    if not rows:
        raise RuntimeError(
            "No labelled P/S arrivals were found. Increase --max_attempts or check the dataset split."
        )

    summary = summarize(rows, args)
    summary["checkpoint"] = args.checkpoint
    summary["data_name"] = args.data_name
    summary["split"] = args.split
    summary["picker"] = args.picker
    summary["picker_weights"] = args.picker_weights
    summary["windows_evaluated"] = evaluated_windows
    summary["targets_evaluated"] = len(rows)
    summary["skipped_without_targets"] = skipped_without_targets

    csv_path = output_dir / "picking_rows.csv"
    json_path = output_dir / "picking_metrics.json"
    txt_path = output_dir / "picking_metrics.txt"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_lines = [
        f"checkpoint: {args.checkpoint}",
        f"dataset: {args.data_name} ({args.split} split)",
        f"picker: {args.picker} ({args.picker_weights})",
        f"windows_evaluated: {evaluated_windows}",
        f"targets_evaluated: {len(rows)}",
        "",
        *summary_lines(summary),
    ]
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(summary_lines(summary)))
    print(f"Saved rows: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved summary: {txt_path}")


if __name__ == "__main__":
    main()
