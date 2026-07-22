"""Evaluate downstream phase-picking degradation for codec baselines."""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import seisbench.data as sbd
import torch

from evaluate_codec_baselines import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    FAMILY_COLORS,
    FAMILY_ORDER,
    CodecSpec,
    build_specs,
    compute_seisdac_bitrate,
    format_float,
    parse_int_list,
)
from evaluate_picking import (
    collect_phase_targets,
    ensure_dataset_split,
    evaluate_local_pick,
    load_picker,
    load_split,
    load_windowed_state,
    picker_labels,
    predict_picker_probs,
    set_seed,
    summarize,
)
from make_paper_figures import COLOR_AXIS, COLOR_GRAY, COLOR_LIGHT, Figure, draw_axes, draw_panel_title
from train import SeisDACLightning


DEFAULT_PICKING_OUTPUT_DIR = "/data/seismic/seis-codec-eval/codec_baselines_ethz_picking"
PHASE_COLORS = {"P": "#2C6B73", "S": "#D97706"}


def load_labelled_windows(args, codec: SeisDACLightning) -> Tuple[List[np.ndarray], List[Dict], Dict]:
    dataset_cls = getattr(sbd, args.data_name)
    dataset = dataset_cls(
        sampling_rate=codec.config.model.sample_rate,
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

    waveforms: List[np.ndarray] = []
    targets_by_window: List[List[Dict]] = []
    skipped_without_targets = 0
    for idx in candidate_indices:
        if len(waveforms) >= args.num_samples:
            break
        state = load_windowed_state(codec, split_dataset, int(idx))
        wave_np, metadata = state["X"]
        wave_np = np.asarray(wave_np, dtype=np.float32)
        targets = collect_phase_targets(metadata, window_length=wave_np.shape[-1])
        if not targets:
            skipped_without_targets += 1
            continue

        window_targets = []
        for phase, true_sample, phase_key in targets:
            window_targets.append(
                {
                    "sample_index": int(idx),
                    "trace_name": metadata.get("trace_name", ""),
                    "phase": phase,
                    "phase_key": phase_key,
                    "true_sample": float(true_sample),
                }
            )
        waveforms.append(np.ascontiguousarray(wave_np))
        targets_by_window.append(window_targets)

    if not waveforms:
        raise RuntimeError("No labelled windows found. Increase --max_attempts or check the split.")

    info = {
        "dataset": args.data_name,
        "split": args.split,
        "sample_rate": float(codec.config.model.sample_rate),
        "window_length": int(waveforms[0].shape[-1]),
        "n_channels": int(waveforms[0].shape[0]),
        "windows_evaluated": int(len(waveforms)),
        "targets_evaluated": int(sum(len(targets) for targets in targets_by_window)),
        "skipped_without_targets": int(skipped_without_targets),
    }
    return waveforms, targets_by_window, info


@torch.no_grad()
def predict_windows(
    picker: torch.nn.Module,
    waveforms: Sequence[np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> List[np.ndarray]:
    outputs: List[np.ndarray] = []
    for start in range(0, len(waveforms), batch_size):
        batch_np = np.stack(waveforms[start : start + batch_size], axis=0).astype(np.float32, copy=False)
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        probs = predict_picker_probs(picker, batch)
        outputs.extend([item for item in probs.detach().cpu().numpy()])
    return outputs


def evaluate_targets_for_codec(
    *,
    codec_id: str,
    family: str,
    setting: str,
    compressed_bps: float,
    compression_ratio: float,
    targets_by_window: Sequence[Sequence[Dict]],
    original_probs: Sequence[np.ndarray],
    reconstructed_probs: Sequence[np.ndarray],
    label_to_channel: Dict[str, int],
    input_length: int,
    sample_rate: float,
    tolerance_sec: float,
    prob_threshold: float,
    search_radius_sec: float,
) -> Tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    for window_idx, targets in enumerate(targets_by_window):
        for target in targets:
            phase = target["phase"]
            channel = label_to_channel[phase]
            original = evaluate_local_pick(
                original_probs[window_idx][channel],
                true_sample=target["true_sample"],
                input_length=input_length,
                sample_rate=sample_rate,
                search_radius_sec=search_radius_sec,
            )
            reconstructed = evaluate_local_pick(
                reconstructed_probs[window_idx][channel],
                true_sample=target["true_sample"],
                input_length=input_length,
                sample_rate=sample_rate,
                search_radius_sec=search_radius_sec,
            )
            original_hit = (
                original["abs_error_sec"] <= tolerance_sec
                and original["peak_prob"] >= prob_threshold
            )
            reconstructed_hit = (
                reconstructed["abs_error_sec"] <= tolerance_sec
                and reconstructed["peak_prob"] >= prob_threshold
            )
            rows.append(
                {
                    "codec_id": codec_id,
                    "family": family,
                    "setting": setting,
                    "compressed_bps": compressed_bps,
                    "compression_ratio_vs_float32": compression_ratio,
                    "sample_index": target["sample_index"],
                    "trace_name": target["trace_name"],
                    "phase": phase,
                    "phase_key": target["phase_key"],
                    "true_sample": target["true_sample"],
                    "original_pred_sample": original["pred_sample"],
                    "reconstructed_pred_sample": reconstructed["pred_sample"],
                    "original_signed_error_sec": original["signed_error_sec"],
                    "reconstructed_signed_error_sec": reconstructed["signed_error_sec"],
                    "original_abs_error_sec": original["abs_error_sec"],
                    "reconstructed_abs_error_sec": reconstructed["abs_error_sec"],
                    "original_peak_prob": original["peak_prob"],
                    "reconstructed_peak_prob": reconstructed["peak_prob"],
                    "original_prob_at_true": original["prob_at_true"],
                    "reconstructed_prob_at_true": reconstructed["prob_at_true"],
                    "picker_shift_sec": abs(reconstructed["pred_sample"] - original["pred_sample"]) / sample_rate,
                    "peak_prob_drop": original["peak_prob"] - reconstructed["peak_prob"],
                    "original_hit": bool(original_hit),
                    "reconstructed_hit": bool(reconstructed_hit),
                }
            )

    args_like = argparse.Namespace(
        tolerance_sec=tolerance_sec,
        prob_threshold=prob_threshold,
        search_radius_sec=search_radius_sec,
    )
    summary = summarize(rows, args_like)
    summary.update(
        {
            "codec_id": codec_id,
            "family": family,
            "setting": setting,
            "compressed_bps": compressed_bps,
            "compression_ratio_vs_float32": compression_ratio,
        }
    )
    return rows, summary


def summarize_to_flat_row(summary: Dict, info: Dict, reconstruction_lookup: Dict[Tuple[str, str], Dict]) -> Dict:
    row = {
        "codec_id": summary["codec_id"],
        "family": summary["family"],
        "setting": summary["setting"],
        "windows": info["windows_evaluated"],
        "compressed_bps": summary["compressed_bps"],
        "compression_ratio_vs_float32": summary["compression_ratio_vs_float32"],
    }
    recon = reconstruction_lookup.get((summary["family"], summary["setting"]))
    if recon:
        row["l1_mean"] = recon.get("l1_mean", "")
        row["snr_mean_db"] = recon.get("snr_mean_db", "")
    else:
        row["l1_mean"] = ""
        row["snr_mean_db"] = ""

    for phase in ["P", "S"]:
        phase_summary = summary["phases"][phase]
        degradation = summary["degradation"][phase]
        row[f"{phase}_n"] = phase_summary["original"]["n"]
        row[f"{phase}_original_mae_sec"] = phase_summary["original"]["mae_sec"]
        row[f"{phase}_reconstructed_mae_sec"] = phase_summary["reconstructed"]["mae_sec"]
        row[f"{phase}_delta_mae_ms"] = 1000.0 * degradation["delta_mae_sec"]
        row[f"{phase}_original_recall"] = phase_summary["original"]["recall_at_tolerance"]
        row[f"{phase}_reconstructed_recall"] = phase_summary["reconstructed"]["recall_at_tolerance"]
        row[f"{phase}_delta_recall_pp"] = 100.0 * degradation["delta_recall_at_tolerance"]
        row[f"{phase}_mean_picker_shift_ms"] = 1000.0 * degradation["mean_picker_shift_sec"]
        row[f"{phase}_mean_prob_drop"] = degradation["mean_peak_prob_drop"]
    return row


def write_target_rows(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: Sequence[Dict]) -> None:
    fieldnames = [
        "codec_id",
        "family",
        "setting",
        "windows",
        "compressed_bps",
        "compression_ratio_vs_float32",
        "l1_mean",
        "snr_mean_db",
        "P_n",
        "P_original_mae_sec",
        "P_reconstructed_mae_sec",
        "P_delta_mae_ms",
        "P_original_recall",
        "P_reconstructed_recall",
        "P_delta_recall_pp",
        "P_mean_picker_shift_ms",
        "P_mean_prob_drop",
        "S_n",
        "S_original_mae_sec",
        "S_reconstructed_mae_sec",
        "S_delta_mae_ms",
        "S_original_recall",
        "S_reconstructed_recall",
        "S_delta_recall_pp",
        "S_mean_picker_shift_ms",
        "S_mean_prob_drop",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_reconstruction_lookup(path: Path) -> Dict[Tuple[str, str], Dict]:
    if not path.exists():
        return {}
    lookup: Dict[Tuple[str, str], Dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in ["compressed_bps", "compression_ratio_vs_float32", "l1_mean", "snr_mean_db"]:
                try:
                    parsed[key] = float(parsed[key])
                except (ValueError, KeyError):
                    pass
            lookup[(row["family"], row["setting"])] = parsed
    return lookup


def codec_id(family: str, setting: str) -> str:
    safe = f"{family}_{setting}"
    for old, new in [(" ", "_"), ("=", ""), (".", "p"), ("/", "_"), ("+", "plus"), ("-", "_")]:
        safe = safe.replace(old, new)
    return safe


def selected_specs(args, sample_rate: int) -> List[CodecSpec]:
    specs = build_specs(args, sample_rate)
    include = set(item.strip() for item in args.include_families.split(",") if item.strip())
    if include:
        specs = [spec for spec in specs if spec.family in include]
    return specs


def reconstruct_classical_batch(
    spec: CodecSpec,
    waveforms: Sequence[np.ndarray],
    *,
    sample_rate: float,
) -> Tuple[List[np.ndarray], float]:
    reconstructed: List[np.ndarray] = []
    byte_values: List[int] = []
    duration_sec = waveforms[0].shape[-1] / sample_rate
    for wave in waveforms:
        output = spec.run(wave)
        reconstructed.append(output.reconstructed)
        byte_values.append(output.nbytes)
    compressed_bps = float(np.mean(byte_values) * 8.0 / duration_sec)
    return reconstructed, compressed_bps


def original_bps(info: Dict) -> float:
    return float(info["n_channels"]) * 32.0 * float(info["sample_rate"])


@torch.no_grad()
def reconstruct_seisdac(
    waveforms: Sequence[np.ndarray],
    codec: SeisDACLightning,
    n_quantizers: int,
    *,
    device: str,
    batch_size: int,
) -> List[np.ndarray]:
    reconstructed: List[np.ndarray] = []
    generator = codec.generator
    generator.eval()
    codec.to(device)
    sample_rate = int(codec.config.model.sample_rate)
    for start in range(0, len(waveforms), batch_size):
        batch_np = np.stack(waveforms[start : start + batch_size], axis=0).astype(np.float32, copy=False)
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        out = generator(batch, sample_rate=sample_rate, n_quantizers=n_quantizers)
        reconstructed.extend([item for item in out["audio"].detach().cpu().numpy().astype(np.float32, copy=False)])
    return reconstructed


def group_rows(rows: Sequence[Dict], family: str) -> List[Dict]:
    return sorted(
        [row for row in rows if row["family"] == family],
        key=lambda row: float(row["compressed_bps"]),
    )


def value_range(values: Sequence[float], *, include_zero: bool = True, pad_fraction: float = 0.12) -> Tuple[float, float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if include_zero:
        clean.append(0.0)
    if not clean:
        return -1.0, 1.0
    lo = min(clean)
    hi = max(clean)
    if abs(hi - lo) < 1e-9:
        delta = max(abs(hi) * pad_fraction, 1.0)
        return lo - delta, hi + delta
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def marker(fig: Figure, x: float, y: float, color: str, *, size: float = 5.5) -> None:
    fig.rect(x - size / 2, y - size / 2, size, size, fill=color, stroke="#FFFFFF", stroke_width=0.7)


def draw_legend(fig: Figure, x: float, y: float, rows: Sequence[Dict]) -> None:
    col_w = 150
    row_h = 16
    families = [family for family in FAMILY_ORDER if any(row["family"] == family for row in rows)]
    for idx, family in enumerate(families):
        px = x + (idx % 3) * col_w
        py = y + (idx // 3) * row_h
        fig.rect(px, py - 9, 11, 8, fill=FAMILY_COLORS[family])
        fig.text(px + 16, py - 1, family, size=8.2, color=COLOR_GRAY)


def draw_degradation_panel(
    fig: Figure,
    rows: Sequence[Dict],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    letter: str,
    title: str,
    field: str,
    ylabel: str,
    x_max: float,
    y_range: Tuple[float, float],
) -> None:
    y_min, y_max = y_range
    y_ticks = np.linspace(y_min, y_max, 5)
    x_ticks = [tick for tick in [0, 500, 1000, 2000, 4000, 6000, 8000, 9600] if tick <= x_max]
    draw_panel_title(fig, x - 34, y - 30, letter, title)
    sx, sy = draw_axes(
        fig,
        x,
        y,
        w,
        h,
        xlim=(0, x_max),
        ylim=(y_min, y_max),
        xticks=x_ticks,
        yticks=y_ticks,
        xlabel="Bitrate (bps)",
        ylabel=ylabel,
    )
    zero_y = sy(0.0)
    fig.line(x, zero_y, x + w, zero_y, color=COLOR_AXIS, width=0.8)
    for family in FAMILY_ORDER:
        family_rows = group_rows(rows, family)
        if not family_rows:
            continue
        color = FAMILY_COLORS[family]
        points = [(sx(float(row["compressed_bps"])), sy(float(row[field]))) for row in family_rows]
        if len(points) > 1:
            fig.polyline(points, color=color, width=1.5)
        for point_idx, (px, py) in enumerate(points):
            marker(fig, px, py, color, size=7.0 if family == "SeisDAC" else 5.5)
            if family == "SeisDAC" and point_idx == len(points) - 1:
                fig.text(px + 8, py - 8, "SeisDAC", size=8.2, weight="bold")


def draw_picking_degradation_figure(rows: Sequence[Dict], out_dir: Path) -> None:
    x_max = max(9600.0, max(float(row["compressed_bps"]) for row in rows) * 1.03)
    fields = ["P_delta_mae_ms", "S_delta_mae_ms", "P_delta_recall_pp", "S_delta_recall_pp"]
    ranges = {field: value_range([row[field] for row in rows]) for field in fields}

    fig = Figure(980, 650)
    draw_degradation_panel(
        fig,
        rows,
        x=74,
        y=64,
        w=390,
        h=175,
        letter="A",
        title="P-phase MAE degradation",
        field="P_delta_mae_ms",
        ylabel="Delta MAE (ms)",
        x_max=x_max,
        y_range=ranges["P_delta_mae_ms"],
    )
    draw_degradation_panel(
        fig,
        rows,
        x=560,
        y=64,
        w=365,
        h=175,
        letter="B",
        title="S-phase MAE degradation",
        field="S_delta_mae_ms",
        ylabel="Delta MAE (ms)",
        x_max=x_max,
        y_range=ranges["S_delta_mae_ms"],
    )
    draw_degradation_panel(
        fig,
        rows,
        x=74,
        y=370,
        w=390,
        h=175,
        letter="C",
        title="P-phase recall degradation",
        field="P_delta_recall_pp",
        ylabel="Delta recall (pp)",
        x_max=x_max,
        y_range=ranges["P_delta_recall_pp"],
    )
    draw_degradation_panel(
        fig,
        rows,
        x=560,
        y=370,
        w=365,
        h=175,
        letter="D",
        title="S-phase recall degradation",
        field="S_delta_recall_pp",
        ylabel="Delta recall (pp)",
        x_max=x_max,
        y_range=ranges["S_delta_recall_pp"],
    )
    draw_legend(fig, 260, 610, rows)
    fig.render_svg(out_dir / "fig_codec_picking_degradation.svg")
    fig.render_pdf(out_dir / "fig_codec_picking_degradation.pdf")


def nearest_rows(rows: Sequence[Dict], target_bps: float) -> List[Dict]:
    selected = []
    for family in FAMILY_ORDER:
        family_rows = group_rows(rows, family)
        if not family_rows:
            continue
        if family == "zstd-float32":
            selected.extend(family_rows)
        else:
            selected.append(min(family_rows, key=lambda row: abs(float(row["compressed_bps"]) - target_bps)))
    return selected


def write_latex(rows: Sequence[Dict], out_dir: Path) -> None:
    seisdac_bps = max(float(row["compressed_bps"]) for row in rows if row["family"] == "SeisDAC")
    selected = nearest_rows(rows, seisdac_bps)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Downstream phase-picking degradation for codec baselines nearest to the full-rate SeisDAC point on ETHZ development windows. Recall uses a 0.2 s tolerance and picker probability threshold of 0.3.}",
        r"\label{tab:codec_picking_degradation}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Codec & Setting & Bitrate (bps) & P $\Delta$MAE (ms) & P $\Delta$Recall (pp) & S $\Delta$MAE (ms) & S $\Delta$Recall (pp) \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['family']} & {row['setting']} & "
            f"{float(row['compressed_bps']):.0f} & "
            f"{float(row['P_delta_mae_ms']):+.1f} & "
            f"{float(row['P_delta_recall_pp']):+.1f} & "
            f"{float(row['S_delta_mae_ms']):+.1f} & "
            f"{float(row['S_delta_recall_pp']):+.1f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\linewidth]{fig_codec_picking_degradation.pdf}",
            r"\caption{Downstream phase-picking degradation as a function of bitrate for classical codec baselines and SeisDAC on ETHZ development windows.}",
            r"\label{fig:codec_picking_degradation}",
            r"\end{figure}",
        ]
    )
    (out_dir / "codec_picking_tables_and_figure.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_txt(rows: Sequence[Dict], info: Dict, out_dir: Path) -> None:
    seisdac_bps = max(float(row["compressed_bps"]) for row in rows if row["family"] == "SeisDAC")
    selected = nearest_rows(rows, seisdac_bps)
    lines = [
        "=== Codec baseline downstream phase picking ===",
        f"Dataset: {info['dataset']} ({info['split']} split)",
        f"Windows: {info['windows_evaluated']}",
        f"Targets: {info['targets_evaluated']}",
        "Recall tolerance: 0.2 s; picker probability threshold: 0.3",
        "",
        "Closest points to full-rate SeisDAC:",
    ]
    for row in selected:
        lines.append(
            f"  {row['family']:18s} {row['setting']:24s} "
            f"bps={float(row['compressed_bps']):7.1f} "
            f"P_dMAE={float(row['P_delta_mae_ms']):+6.1f}ms "
            f"P_dRecall={float(row['P_delta_recall_pp']):+5.1f}pp "
            f"S_dMAE={float(row['S_delta_mae_ms']):+6.1f}ms "
            f"S_dRecall={float(row['S_delta_recall_pp']):+5.1f}pp"
        )
    (out_dir / "codec_picking_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate downstream phase-picking degradation for codec baselines."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_name", default="ETHZ")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--picker", default="phasenet", choices=["phasenet", "eqtransformer", "eqt"])
    parser.add_argument("--picker_weights", default="ethz")
    parser.add_argument("--picker_update", action="store_true")
    parser.add_argument("--prob_threshold", type=float, default=0.3)
    parser.add_argument("--tolerance_sec", type=float, default=0.2)
    parser.add_argument("--search_radius_sec", type=float, default=1.0)
    parser.add_argument("--output_dir", default=DEFAULT_PICKING_OUTPUT_DIR)
    parser.add_argument("--reconstruction_results", default=str(Path(DEFAULT_OUTPUT_DIR) / "codec_baseline_results.csv"))
    parser.add_argument("--quant_bits", default="4,6,8,10,12,14,16")
    parser.add_argument("--zstd_level", type=int, default=9)
    parser.add_argument("--zfp_rates", default="1,1.5,2,2.5,3,3.75,4.5,6,8,12")
    parser.add_argument("--sz3_abs_bounds", default="0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--miniseed_reclen", type=int, default=4096)
    parser.add_argument("--seisdac_quantizers", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--include_families", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_target_rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    waveforms, targets_by_window, info = load_labelled_windows(args, codec)
    sample_rate = float(info["sample_rate"])
    picker_sample_rate = getattr(picker, "sampling_rate", sample_rate)
    if picker_sample_rate and abs(float(picker_sample_rate) - sample_rate) > 1e-6:
        raise ValueError(
            f"Codec sample rate is {sample_rate} Hz, but picker expects {picker_sample_rate} Hz."
        )

    print(
        f"Loaded {info['windows_evaluated']} labelled windows "
        f"with {info['targets_evaluated']} P/S targets."
    )
    print("Predicting original waveforms once...")
    original_probs = predict_windows(picker, waveforms, device=args.device, batch_size=args.batch_size)
    reconstruction_lookup = read_reconstruction_lookup(Path(args.reconstruction_results))

    summaries: List[Dict] = []
    flat_rows: List[Dict] = []
    all_target_rows: List[Dict] = []

    specs = selected_specs(args, int(sample_rate))
    for spec_idx, spec in enumerate(specs, start=1):
        print(f"[{spec_idx:02d}/{len(specs):02d}] {spec.family} | {spec.setting}")
        start_time = time.perf_counter()
        reconstructed, compressed_bps = reconstruct_classical_batch(
            spec,
            waveforms,
            sample_rate=sample_rate,
        )
        reconstructed_probs = predict_windows(
            picker,
            reconstructed,
            device=args.device,
            batch_size=args.batch_size,
        )
        ratio = original_bps(info) / compressed_bps
        rows, summary = evaluate_targets_for_codec(
            codec_id=codec_id(spec.family, spec.setting),
            family=spec.family,
            setting=spec.setting,
            compressed_bps=compressed_bps,
            compression_ratio=ratio,
            targets_by_window=targets_by_window,
            original_probs=original_probs,
            reconstructed_probs=reconstructed_probs,
            label_to_channel=label_to_channel,
            input_length=int(info["window_length"]),
            sample_rate=sample_rate,
            tolerance_sec=args.tolerance_sec,
            prob_threshold=args.prob_threshold,
            search_radius_sec=args.search_radius_sec,
        )
        summaries.append(summary)
        flat = summarize_to_flat_row(summary, info, reconstruction_lookup)
        flat_rows.append(flat)
        if args.save_target_rows:
            all_target_rows.extend(rows)
        print(
            "    "
            f"P dMAE={float(flat['P_delta_mae_ms']):+.1f} ms, "
            f"P dRecall={float(flat['P_delta_recall_pp']):+.1f} pp, "
            f"S dRecall={float(flat['S_delta_recall_pp']):+.1f} pp, "
            f"elapsed={time.perf_counter() - start_time:.1f}s"
        )

    seisdac_quantizers = parse_int_list(args.seisdac_quantizers)
    for idx, n_quantizers in enumerate(seisdac_quantizers, start=1):
        print(f"[SeisDAC {idx:02d}/{len(seisdac_quantizers):02d}] n_quantizers={n_quantizers}")
        start_time = time.perf_counter()
        reconstructed = reconstruct_seisdac(
            waveforms,
            codec,
            n_quantizers,
            device=args.device,
            batch_size=args.batch_size,
        )
        reconstructed_probs = predict_windows(
            picker,
            reconstructed,
            device=args.device,
            batch_size=args.batch_size,
        )
        compressed_bps, ratio = compute_seisdac_bitrate(
            n_quantizers=n_quantizers,
            codebook_size=int(codec.generator.codebook_size),
            hop_length=int(codec.generator.hop_length),
            sample_rate=int(sample_rate),
            n_channels=int(info["n_channels"]),
        )
        setting = f"nq={n_quantizers}"
        rows, summary = evaluate_targets_for_codec(
            codec_id=codec_id("SeisDAC", setting),
            family="SeisDAC",
            setting=setting,
            compressed_bps=compressed_bps,
            compression_ratio=ratio,
            targets_by_window=targets_by_window,
            original_probs=original_probs,
            reconstructed_probs=reconstructed_probs,
            label_to_channel=label_to_channel,
            input_length=int(info["window_length"]),
            sample_rate=sample_rate,
            tolerance_sec=args.tolerance_sec,
            prob_threshold=args.prob_threshold,
            search_radius_sec=args.search_radius_sec,
        )
        summaries.append(summary)
        flat = summarize_to_flat_row(summary, info, reconstruction_lookup)
        flat_rows.append(flat)
        if args.save_target_rows:
            all_target_rows.extend(rows)
        print(
            "    "
            f"P dMAE={float(flat['P_delta_mae_ms']):+.1f} ms, "
            f"P dRecall={float(flat['P_delta_recall_pp']):+.1f} pp, "
            f"S dRecall={float(flat['S_delta_recall_pp']):+.1f} pp, "
            f"elapsed={time.perf_counter() - start_time:.1f}s"
        )

    summary_csv = out_dir / "codec_picking_summary.csv"
    summary_json = out_dir / "codec_picking_summary.json"
    write_summary_csv(summary_csv, flat_rows)
    summary_json.write_text(
        json.dumps(json_safe({"info": info, "summaries": summaries}), indent=2) + "\n",
        encoding="utf-8",
    )
    if args.save_target_rows:
        write_target_rows(out_dir / "codec_picking_rows.csv", all_target_rows)
    draw_picking_degradation_figure(flat_rows, out_dir)
    write_latex(flat_rows, out_dir)
    write_summary_txt(flat_rows, info, out_dir)

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved summary TXT: {out_dir / 'codec_picking_summary.txt'}")
    print(f"Saved figure: {out_dir / 'fig_codec_picking_degradation.pdf'}")
    print(f"Saved LaTeX: {out_dir / 'codec_picking_tables_and_figure.tex'}")


if __name__ == "__main__":
    main()
