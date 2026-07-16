"""Generate multi-dataset summary tables and figures for SeisDAC results."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

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


DATASETS = {
    "ETHZ": {
        "reconstruction": "/data/seismic/seis-codec-eval/ethz_nogan_latreg0p003_best137/metrics.txt",
        "picking": "/data/seismic/seis-codec-eval/ethz_nogan_latreg0p003_picking/picking_metrics.json",
    },
    "STEAD": {
        "reconstruction": "/data/seismic/seis-codec-eval/stead_nogan_latreg0p003_best137/metrics.txt",
        "picking": "/data/seismic/seis-codec-eval/stead_nogan_latreg0p003_picking/picking_metrics.json",
    },
    "GEOFON": {
        "reconstruction": "/data/seismic/seis-codec-eval/geofon_nogan_latreg0p003_best137/metrics.txt",
        "picking": "/data/seismic/seis-codec-eval/geofon_nogan_latreg0p003_picking/picking_metrics.json",
    },
}

PHASE_COLORS = {"P": "#2C6B73", "S": "#C44E52"}


def load_results() -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for dataset, paths in DATASETS.items():
        reconstruction_path = Path(paths["reconstruction"])
        picking_path = Path(paths["picking"])
        if not reconstruction_path.exists():
            raise FileNotFoundError(reconstruction_path)
        if not picking_path.exists():
            raise FileNotFoundError(picking_path)
        results[dataset] = {
            "reconstruction": parse_metrics_txt(reconstruction_path),
            "picking": json.loads(picking_path.read_text(encoding="utf-8")),
        }
    return results


def fmt_pm(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_summary_tables(results: Dict[str, Dict], path: Path) -> None:
    lines: List[str] = []
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Cross-dataset reconstruction quality and theoretical compression rate for the no-GAN SeisDAC model with latent regularization weight $3\times10^{-3}$. Metrics are computed over 512 development windows per dataset.}",
            r"\label{tab:multidataset_reconstruction}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Dataset & L1 error & SNR (dB) & Bitrate (bps) & Compression ratio \\",
            r"\midrule",
        ]
    )
    for dataset in ["ETHZ", "STEAD", "GEOFON"]:
        rec = results[dataset]["reconstruction"]
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
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Cross-dataset phase-picking degradation after compression. $\Delta$MAE is reported in milliseconds and $\Delta$Recall in percentage points. Recall uses a 0.2 s tolerance and a picker probability threshold of 0.3.}",
            r"\label{tab:multidataset_picking_degradation}",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule",
            r"Dataset & Phase & Original recall & Reconstructed recall & $\Delta$Recall (pp) & Original MAE (s) & Reconstructed MAE (s) & $\Delta$MAE (ms) \\",
            r"\midrule",
        ]
    )
    for dataset in ["ETHZ", "STEAD", "GEOFON"]:
        picking = results[dataset]["picking"]
        for phase in ["P", "S"]:
            original = picking["phases"][phase]["original"]
            reconstructed = picking["phases"][phase]["reconstructed"]
            degradation = picking["degradation"][phase]
            lines.append(
                f"{dataset} & {phase} & "
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
            r"\footnotesize GEOFON phase-picking degradation should be interpreted cautiously because the original-waveform PhaseNet baseline is weak, especially for S phases.",
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_snippet(out_dir: Path) -> None:
    snippet = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_multidataset_summary.pdf}
\caption{Cross-dataset summary for the no-GAN SeisDAC model with latent regularization weight $3\times10^{-3}$. Reconstruction quality is reported by L1 error and SNR; downstream degradation is reported by the change in phase-picking MAE and recall after compression. GEOFON picking is confounded by a weak original-waveform picker baseline, especially for S phases.}
\label{fig:multidataset_summary}
\end{figure}
"""
    (out_dir / "multidataset_summary_figure.tex").write_text(snippet + "\n", encoding="utf-8")


def bar(
    fig: Figure,
    x: float,
    base_y: float,
    value_y: float,
    width: float,
    color: str,
    label: str,
    *,
    text_size: float = 8,
) -> None:
    fig.rect(x - width / 2, value_y, width, base_y - value_y, fill=color)
    fig.text(x, value_y - 6, label, size=text_size, anchor="middle")


def draw_reconstruction_panels(fig: Figure, results: Dict[str, Dict]) -> None:
    datasets = ["ETHZ", "STEAD", "GEOFON"]

    draw_panel_title(fig, 42, 32, "A", "Reconstruction error")
    _, sy = draw_axes(
        fig,
        78,
        62,
        240,
        150,
        xlim=(-0.5, 2.5),
        ylim=(0, 0.04),
        xticks=[],
        yticks=[0, 0.01, 0.02, 0.03, 0.04],
        ylabel="L1 error",
    )
    for i, dataset in enumerate(datasets):
        value = results[dataset]["reconstruction"]["l1_mean"]
        cx = 78 + 240 * (i + 0.5) / 3
        bar(fig, cx, 212, sy(value), 42, COLOR_RECON, f"{value:.3f}")
        fig.text(cx, 230, dataset, size=9, weight="bold", anchor="middle")

    draw_panel_title(fig, 414, 32, "B", "Signal-to-noise ratio")
    _, sy = draw_axes(
        fig,
        450,
        62,
        240,
        150,
        xlim=(-0.5, 2.5),
        ylim=(0, 25),
        xticks=[],
        yticks=[0, 5, 10, 15, 20, 25],
        ylabel="SNR (dB)",
    )
    for i, dataset in enumerate(datasets):
        value = results[dataset]["reconstruction"]["snr_mean_db"]
        cx = 450 + 240 * (i + 0.5) / 3
        bar(fig, cx, 212, sy(value), 42, COLOR_ORIGINAL, f"{value:.1f}")
        fig.text(cx, 230, dataset, size=9, weight="bold", anchor="middle")


def draw_degradation_panel(
    fig: Figure,
    results: Dict[str, Dict],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title_letter: str,
    title: str,
    metric: str,
    ylim,
    yticks,
    ylabel: str,
    scale: float,
    fmt: str,
) -> None:
    datasets = ["ETHZ", "STEAD", "GEOFON"]
    draw_panel_title(fig, x - 36, y - 30, title_letter, title)
    _, sy = draw_axes(
        fig,
        x,
        y,
        w,
        h,
        xlim=(-0.5, 2.5),
        ylim=ylim,
        xticks=[],
        yticks=yticks,
        ylabel=ylabel,
    )
    base_y = sy(0)
    fig.line(x, base_y, x + w, base_y, color=COLOR_AXIS, width=0.9)
    span = w / 3
    for i, dataset in enumerate(datasets):
        cx = x + span * (i + 0.5)
        for phase, dx in [("P", -14), ("S", 14)]:
            value = scale * results[dataset]["picking"]["degradation"][phase][metric]
            value_y = sy(value)
            y0 = min(base_y, value_y)
            height = abs(base_y - value_y)
            fig.rect(cx + dx - 11, y0, 22, height, fill=PHASE_COLORS[phase])
            label_y = value_y - 6 if value >= 0 else value_y + 14
            fig.text(cx + dx, label_y, format(value, fmt), size=7.5, anchor="middle")
        fig.text(cx, y + h + 18, dataset, size=9, weight="bold", anchor="middle")


def draw_summary_figure(results: Dict[str, Dict], out_dir: Path) -> None:
    fig = Figure(760, 520)
    draw_reconstruction_panels(fig, results)
    draw_legend(fig, 276, 268, [("P phase", PHASE_COLORS["P"]), ("S phase", PHASE_COLORS["S"])])

    draw_degradation_panel(
        fig,
        results,
        x=78,
        y=318,
        w=240,
        h=145,
        title_letter="C",
        title="Picking MAE degradation",
        metric="delta_mae_sec",
        ylim=(-20, 120),
        yticks=[-20, 0, 40, 80, 120],
        ylabel="Delta MAE (ms)",
        scale=1000.0,
        fmt="+.0f",
    )
    draw_degradation_panel(
        fig,
        results,
        x=450,
        y=318,
        w=240,
        h=145,
        title_letter="D",
        title="Picking recall degradation",
        metric="delta_recall_at_tolerance",
        ylim=(-18, 2),
        yticks=[-15, -10, -5, 0],
        ylabel="Delta recall (pp)",
        scale=100.0,
        fmt="+.1f",
    )
    fig.text(
        380,
        505,
        "All datasets use the same ETHZ-trained codec at ~9x theoretical compression; GEOFON picking is baseline-limited.",
        size=9,
        color=COLOR_GRAY,
        anchor="middle",
    )
    fig.render_svg(out_dir / "fig_multidataset_summary.svg")
    fig.render_pdf(out_dir / "fig_multidataset_summary.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-dataset SeisDAC summary.")
    parser.add_argument(
        "--output_dir",
        default="/data/seismic/seis-codec-eval/multidataset_summary_figures",
    )
    parser.add_argument(
        "--table_path",
        default="multidataset_summary_results.tex",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_path = Path(args.table_path)
    write_summary_tables(results, table_path)
    draw_summary_figure(results, out_dir)
    write_figure_snippet(out_dir)

    out_table = out_dir / table_path.name
    out_table.write_text(table_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Saved LaTeX table: {table_path}")
    print(f"Saved figure directory: {out_dir}")


if __name__ == "__main__":
    main()
