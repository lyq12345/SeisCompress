"""Generate cross-dataset summary tables and figures for SeisDAC results."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from make_paper_figures import (
    COLOR_AXIS,
    COLOR_GRAY,
    COLOR_LIGHT,
    COLOR_ORIGINAL,
    COLOR_RECON,
    Figure,
    draw_axes,
    draw_legend,
    draw_panel_title,
    parse_metrics_txt,
)


EVAL_ROOT = Path("/data/seismic/seis-codec-eval")

RECON_DATASETS: Sequence[Tuple[str, str]] = (
    ("ETHZ", "ethz"),
    ("STEAD", "stead"),
    ("GEOFON", "geofon"),
    ("INSTANCE", "instance_counts_combined"),
    ("Iquique", "iquique"),
    ("MLAAPDE", "mlaapde"),
    ("PNW", "pnw"),
    ("OBST2024", "obst2024"),
)

PICKING_DATASETS: Sequence[Tuple[str, str]] = (
    ("ETHZ", "ethz"),
    ("STEAD", "stead"),
    ("GEOFON", "geofon"),
    ("INSTANCE", "instance_counts_combined"),
    ("Iquique", "iquique"),
)

PHASE_COLORS = {"P": "#2C6B73", "S": "#D97706"}


def reconstruction_path(key: str, eval_root: Path) -> Path:
    return eval_root / f"{key}_nogan_latreg0p003_best137" / "metrics.txt"


def picking_path(key: str, eval_root: Path) -> Path:
    return eval_root / f"{key}_nogan_latreg0p003_picking" / "picking_metrics.json"


def load_results(eval_root: Path) -> Dict[str, Dict[str, Dict]]:
    reconstruction: Dict[str, Dict] = {}
    picking: Dict[str, Dict] = {}

    for dataset, key in RECON_DATASETS:
        path = reconstruction_path(key, eval_root)
        if not path.exists():
            raise FileNotFoundError(path)
        reconstruction[dataset] = parse_metrics_txt(path)

    for dataset, key in PICKING_DATASETS:
        path = picking_path(key, eval_root)
        if not path.exists():
            raise FileNotFoundError(path)
        picking[dataset] = json.loads(path.read_text(encoding="utf-8"))

    return {"reconstruction": reconstruction, "picking": picking}


def fmt_pm(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_summary_tables(results: Dict[str, Dict], path: Path) -> None:
    reconstruction = results["reconstruction"]
    picking = results["picking"]

    lines: List[str] = []
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Cross-dataset reconstruction quality and theoretical compression rate for the no-GAN SeisDAC model with latent regularization weight $3\times10^{-3}$. Metrics are computed over 512 development windows per dataset.}",
            r"\label{tab:cross_dataset_reconstruction}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Dataset & L1 error & SNR (dB) & Bitrate (bps) & Compression ratio \\",
            r"\midrule",
        ]
    )
    for dataset, _ in RECON_DATASETS:
        rec = reconstruction[dataset]
        lines.append(
            f"{dataset} & "
            f"{fmt_pm(rec['l1_mean'], rec['l1_std'], 4)} & "
            f"{fmt_pm(rec['snr_mean_db'], rec['snr_std_db'], 2)} & "
            f"{rec['compressed_bps']:.0f} & "
            f"{rec['compression_ratio_vs_float32']:.1f}$\\times$ \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize OBST2024 shows a large L1 standard deviation, indicating a small number of difficult or outlier windows despite a moderate mean SNR.",
            r"\end{flushleft}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Cross-dataset phase-picking degradation after compression. $\Delta$MAE is reported in milliseconds and $\Delta$Recall in percentage points. Recall uses a 0.2 s tolerance and a picker probability threshold of 0.3.}",
            r"\label{tab:cross_dataset_picking_degradation}",
            r"\begin{tabular}{llrrrrrrr}",
            r"\toprule",
            r"Dataset & Phase & N & Original recall & Reconstructed recall & $\Delta$Recall (pp) & Original MAE (s) & Reconstructed MAE (s) & $\Delta$MAE (ms) \\",
            r"\midrule",
        ]
    )
    for dataset, _ in PICKING_DATASETS:
        dataset_picking = picking[dataset]
        for phase in ["P", "S"]:
            original = dataset_picking["phases"][phase]["original"]
            reconstructed = dataset_picking["phases"][phase]["reconstructed"]
            degradation = dataset_picking["degradation"][phase]
            lines.append(
                f"{dataset} & {phase} & "
                f"{original['n']} & "
                f"{original['recall_at_tolerance']:.3f} & "
                f"{reconstructed['recall_at_tolerance']:.3f} & "
                f"{100.0 * degradation['delta_recall_at_tolerance']:+.1f} & "
                f"{original['mae_sec']:.4f} & "
                f"{reconstructed['mae_sec']:.4f} & "
                f"{1000.0 * degradation['delta_mae_sec']:+.1f} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize Only datasets with completed phase-picking evaluations are included. GEOFON picking should be interpreted cautiously because the original-waveform PhaseNet baseline is weak, especially for S phases.",
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_snippet(out_dir: Path) -> None:
    snippet = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_cross_dataset_summary.pdf}
\caption{Cross-dataset summary for the no-GAN SeisDAC model with latent regularization weight $3\times10^{-3}$. Reconstruction quality is reported by L1 error and SNR; downstream degradation is reported by the change in phase-picking MAE and recall after compression.}
\label{fig:cross_dataset_summary}
\end{figure}
"""
    (out_dir / "cross_dataset_summary_figure.tex").write_text(snippet + "\n", encoding="utf-8")


def draw_bar(
    fig: Figure,
    x: float,
    base_y: float,
    value_y: float,
    width: float,
    color: str,
    label: str,
    *,
    text_size: float = 7.5,
) -> None:
    fig.rect(x - width / 2, value_y, width, base_y - value_y, fill=color)
    fig.text(x, value_y - 6, label, size=text_size, anchor="middle")


def draw_reconstruction_panel(
    fig: Figure,
    reconstruction: Dict[str, Dict],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title_letter: str,
    title: str,
    metric: str,
    ylim: Tuple[float, float],
    yticks: Sequence[float],
    ylabel: str,
    color: str,
    label_fmt: str,
) -> None:
    draw_panel_title(fig, x - 34, y - 30, title_letter, title)
    _, sy = draw_axes(
        fig,
        x,
        y,
        w,
        h,
        xlim=(-0.5, len(RECON_DATASETS) - 0.5),
        ylim=ylim,
        xticks=[],
        yticks=yticks,
        ylabel=ylabel,
    )
    span = w / len(RECON_DATASETS)
    bar_width = min(30, span * 0.55)
    base_y = sy(0)
    for i, (dataset, _) in enumerate(RECON_DATASETS):
        value = reconstruction[dataset][metric]
        cx = x + span * (i + 0.5)
        draw_bar(fig, cx, base_y, sy(value), bar_width, color, format(value, label_fmt))
        fig.text(cx, y + h + 17, dataset, size=6.8, color=COLOR_GRAY, weight="bold", anchor="middle")


def draw_degradation_panel(
    fig: Figure,
    picking: Dict[str, Dict],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title_letter: str,
    title: str,
    metric: str,
    ylim: Tuple[float, float],
    yticks: Sequence[float],
    ylabel: str,
    scale: float,
    label_fmt: str,
) -> None:
    draw_panel_title(fig, x - 34, y - 30, title_letter, title)
    _, sy = draw_axes(
        fig,
        x,
        y,
        w,
        h,
        xlim=(-0.5, len(PICKING_DATASETS) - 0.5),
        ylim=ylim,
        xticks=[],
        yticks=yticks,
        ylabel=ylabel,
    )
    base_y = sy(0)
    fig.line(x, base_y, x + w, base_y, color=COLOR_AXIS, width=0.9)
    span = w / len(PICKING_DATASETS)
    bar_w = min(20, span * 0.24)
    for i, (dataset, _) in enumerate(PICKING_DATASETS):
        cx = x + span * (i + 0.5)
        for phase, dx in [("P", -bar_w * 0.65), ("S", bar_w * 0.65)]:
            value = scale * picking[dataset]["degradation"][phase][metric]
            value_y = sy(value)
            y0 = min(base_y, value_y)
            fig.rect(cx + dx - bar_w / 2, y0, bar_w, abs(base_y - value_y), fill=PHASE_COLORS[phase])
            label_y = value_y - 5 if value >= 0 else value_y + 12
            fig.text(cx + dx, label_y, format(value, label_fmt), size=7.2, anchor="middle")
        fig.text(cx, y + h + 17, dataset, size=7.2, color=COLOR_GRAY, weight="bold", anchor="middle")


def draw_summary_figure(results: Dict[str, Dict], out_dir: Path) -> None:
    reconstruction = results["reconstruction"]
    picking = results["picking"]

    fig = Figure(980, 640)
    draw_reconstruction_panel(
        fig,
        reconstruction,
        x=74,
        y=64,
        w=390,
        h=165,
        title_letter="A",
        title="Reconstruction error",
        metric="l1_mean",
        ylim=(0, 0.06),
        yticks=[0, 0.02, 0.04, 0.06],
        ylabel="L1 error",
        color=COLOR_RECON,
        label_fmt=".3f",
    )
    draw_reconstruction_panel(
        fig,
        reconstruction,
        x=560,
        y=64,
        w=365,
        h=165,
        title_letter="B",
        title="Signal-to-noise ratio",
        metric="snr_mean_db",
        ylim=(0, 25),
        yticks=[0, 5, 10, 15, 20, 25],
        ylabel="SNR (dB)",
        color=COLOR_ORIGINAL,
        label_fmt=".1f",
    )
    draw_legend(fig, 420, 308, [("P phase", PHASE_COLORS["P"]), ("S phase", PHASE_COLORS["S"])])
    draw_degradation_panel(
        fig,
        picking,
        x=74,
        y=372,
        w=390,
        h=165,
        title_letter="C",
        title="Picking MAE degradation",
        metric="delta_mae_sec",
        ylim=(0, 120),
        yticks=[0, 40, 80, 120],
        ylabel="Delta MAE (ms)",
        scale=1000.0,
        label_fmt="+.0f",
    )
    draw_degradation_panel(
        fig,
        picking,
        x=560,
        y=372,
        w=365,
        h=165,
        title_letter="D",
        title="Picking recall degradation",
        metric="delta_recall_at_tolerance",
        ylim=(-17, 2),
        yticks=[-15, -10, -5, 0],
        ylabel="Delta recall (pp)",
        scale=100.0,
        label_fmt="+.1f",
    )
    fig.text(
        490,
        610,
        "All reconstruction evaluations use the same ETHZ-trained codec at 1125 bps (8.5x float32 ZNE compression).",
        size=9,
        color=COLOR_GRAY,
        anchor="middle",
    )
    fig.render_svg(out_dir / "fig_cross_dataset_summary.svg")
    fig.render_pdf(out_dir / "fig_cross_dataset_summary.pdf")
    fig.render_svg(out_dir / "fig_multidataset_summary.svg")
    fig.render_pdf(out_dir / "fig_multidataset_summary.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cross-dataset SeisDAC paper tables and figures.")
    parser.add_argument(
        "--eval_root",
        default=str(EVAL_ROOT),
        help="Directory containing *_nogan_latreg0p003_* evaluation outputs.",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/seismic/seis-codec-eval/cross_dataset_summary_figures",
    )
    parser.add_argument(
        "--table_path",
        default="/data/seismic/seis-codec-eval/cross_dataset_summary_results.tex",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(Path(args.eval_root))
    table_path = Path(args.table_path)
    write_summary_tables(results, table_path)
    draw_summary_figure(results, out_dir)
    write_figure_snippet(out_dir)

    out_table = out_dir / table_path.name
    if out_table.resolve() != table_path.resolve():
        out_table.write_text(table_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Saved LaTeX table: {table_path}")
    print(f"Saved figure directory: {out_dir}")


if __name__ == "__main__":
    main()
