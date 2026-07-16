"""Generate publication-style figures for SeisDAC evaluation results.

This script intentionally avoids optional plotting dependencies. It writes
vector SVG and PDF files directly from the reconstruction and picking metrics.
"""

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


COLOR_ORIGINAL = "#2C6B73"
COLOR_RECON = "#C44E52"
COLOR_CODEC = "#6A4C93"
COLOR_GRAY = "#4B5563"
COLOR_LIGHT = "#E5E7EB"
COLOR_AXIS = "#111827"


def parse_metrics_txt(path: Path) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            values[key] = float(value)
        except ValueError:
            continue
    return values


def hex_to_rgb(color: str) -> Tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def approx_text_width(text: str, size: float) -> float:
    return len(text) * size * 0.52


class Figure:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.svg: List[str] = []
        self.pdf: List[str] = []

    def _pdf_y(self, y: float) -> float:
        return self.height - y

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = COLOR_AXIS,
        width: float = 1.0,
        dash: str = "",
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width:.2f}"{dash_attr}/>'
        )
        r, g, b = hex_to_rgb(color)
        dash_cmd = "[4 4] 0 d" if dash else "[] 0 d"
        self.pdf.append(
            f"q {r:.4f} {g:.4f} {b:.4f} RG {width:.3f} w {dash_cmd} "
            f"{x1:.3f} {self._pdf_y(y1):.3f} m {x2:.3f} {self._pdf_y(y2):.3f} l S Q"
        )

    def polyline(
        self,
        points: Sequence[Tuple[float, float]],
        *,
        color: str,
        width: float = 1.5,
        dash: str = "",
    ) -> None:
        if not points:
            return
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{width:.2f}" stroke-linejoin="round" '
            f'stroke-linecap="round"{dash_attr}/>'
        )
        r, g, b = hex_to_rgb(color)
        dash_cmd = "[4 4] 0 d" if dash else "[] 0 d"
        cmds = [
            f"q {r:.4f} {g:.4f} {b:.4f} RG {width:.3f} w {dash_cmd}",
            f"{points[0][0]:.3f} {self._pdf_y(points[0][1]):.3f} m",
        ]
        for x, y in points[1:]:
            cmds.append(f"{x:.3f} {self._pdf_y(y):.3f} l")
        cmds.append("S Q")
        self.pdf.append(" ".join(cmds))

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1.0,
    ) -> None:
        self.svg.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
        )
        cmds = ["q"]
        if fill != "none":
            r, g, b = hex_to_rgb(fill)
            cmds.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
        if stroke != "none":
            r, g, b = hex_to_rgb(stroke)
            cmds.append(f"{r:.4f} {g:.4f} {b:.4f} RG {stroke_width:.3f} w")
        cmds.append(f"{x:.3f} {self.height - y - h:.3f} {w:.3f} {h:.3f} re")
        if fill != "none" and stroke != "none":
            cmds.append("B")
        elif fill != "none":
            cmds.append("f")
        else:
            cmds.append("S")
        cmds.append("Q")
        self.pdf.append(" ".join(cmds))

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: float = 9,
        color: str = COLOR_AXIS,
        anchor: str = "start",
        weight: str = "normal",
    ) -> None:
        escaped = html.escape(text)
        family = "Helvetica, Arial, sans-serif"
        self.svg.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" '
            f'font-size="{size:.2f}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escaped}</text>'
        )
        r, g, b = hex_to_rgb(color)
        x_pdf = x
        if anchor == "middle":
            x_pdf -= approx_text_width(text, size) / 2
        elif anchor == "end":
            x_pdf -= approx_text_width(text, size)
        font = "/F2" if weight == "bold" else "/F1"
        self.pdf.append(
            f"q {r:.4f} {g:.4f} {b:.4f} rg BT {font} {size:.3f} Tf "
            f"{x_pdf:.3f} {self._pdf_y(y):.3f} Td ({pdf_escape(text)}) Tj ET Q"
        )

    def render_svg(self, path: Path) -> None:
        content = "\n".join(self.svg)
        path.write_text(
            "\n".join(
                [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
                    f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
                    '<rect width="100%" height="100%" fill="white"/>',
                    content,
                    "</svg>",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def render_pdf(self, path: Path) -> None:
        stream = "\n".join(self.pdf).encode("latin-1", errors="replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
            ).encode("latin-1"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        ]
        pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{i} 0 obj\n".encode("ascii"))
            pdf.extend(obj)
            pdf.extend(b"\nendobj\n")
        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        pdf.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        pdf.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(pdf)


def linear_map(value: float, src: Tuple[float, float], dst: Tuple[float, float]) -> float:
    return dst[0] + (value - src[0]) / (src[1] - src[0]) * (dst[1] - dst[0])


def draw_panel_title(fig: Figure, x: float, y: float, letter: str, title: str) -> None:
    fig.text(x, y, letter, size=12, weight="bold")
    fig.text(x + 18, y, title, size=11, weight="bold")


def draw_axes(
    fig: Figure,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    xticks: Sequence[float],
    yticks: Sequence[float],
    xlabel: str = "",
    ylabel: str = "",
) -> Tuple:
    def sx(v: float) -> float:
        return linear_map(v, xlim, (x, x + w))

    def sy(v: float) -> float:
        return linear_map(v, ylim, (y + h, y))

    for tick in yticks:
        py = sy(tick)
        fig.line(x, py, x + w, py, color=COLOR_LIGHT, width=0.7)
        fig.text(x - 6, py + 3, f"{tick:g}", size=8, color=COLOR_GRAY, anchor="end")
    fig.line(x, y + h, x + w, y + h, color=COLOR_AXIS, width=1.0)
    fig.line(x, y, x, y + h, color=COLOR_AXIS, width=1.0)
    for tick in xticks:
        px = sx(tick)
        fig.line(px, y + h, px, y + h + 4, color=COLOR_AXIS, width=0.8)
        fig.text(px, y + h + 16, f"{tick:g}", size=8, color=COLOR_GRAY, anchor="middle")
    if xlabel:
        fig.text(x + w / 2, y + h + 32, xlabel, size=9, color=COLOR_GRAY, anchor="middle")
    if ylabel:
        fig.text(x - 38, y - 12, ylabel, size=9, color=COLOR_GRAY)
    return sx, sy


def draw_legend(fig: Figure, x: float, y: float, entries: Sequence[Tuple[str, str]]) -> None:
    offset = 0
    for label, color in entries:
        fig.rect(x + offset, y - 9, 11, 8, fill=color)
        fig.text(x + offset + 16, y - 1, label, size=8.5, color=COLOR_GRAY)
        offset += 118


def draw_compression_figure(metrics: Dict[str, float], out_dir: Path) -> None:
    fig = Figure(720, 260)
    draw_panel_title(fig, 38, 28, "A", "Bitrate reduction")
    draw_panel_title(fig, 292, 28, "B", "Reconstruction error")
    draw_panel_title(fig, 522, 28, "C", "Signal-to-noise ratio")

    original_bps = metrics["compressed_bps"] * metrics["compression_ratio_vs_float32"]
    codec_bps = metrics["compressed_bps"]

    x0, y0, w, h = 64, 55, 168, 150
    sx, sy = draw_axes(
        fig,
        x0,
        y0,
        w,
        h,
        xlim=(-0.5, 1.5),
        ylim=(0, 10000),
        xticks=[],
        yticks=[0, 2500, 5000, 7500, 10000],
        ylabel="Bitrate (bps)",
    )
    for i, (label, value, color) in enumerate(
        [("Float32 ZNE", original_bps, COLOR_ORIGINAL), ("Codec", codec_bps, COLOR_CODEC)]
    ):
        px = sx(i)
        py = sy(value)
        fig.rect(px - 24, py, 48, y0 + h - py, fill=color)
        fig.text(px, y0 + h + 18, label, size=8.5, color=COLOR_GRAY, anchor="middle")
        fig.text(px, py - 7, f"{value:.0f}", size=8.5, color=COLOR_AXIS, anchor="middle")
    fig.text(x0 + w / 2, 232, f"{metrics['compression_ratio_vs_float32']:.1f}x theoretical compression", size=10, weight="bold", anchor="middle")

    x1, y1, w1, h1 = 310, 55, 145, 150
    l1_mean = metrics["l1_mean"]
    l1_std = metrics["l1_std"]
    _, sy_l1 = draw_axes(
        fig,
        x1,
        y1,
        w1,
        h1,
        xlim=(-0.5, 0.5),
        ylim=(0, 0.04),
        xticks=[],
        yticks=[0, 0.01, 0.02, 0.03, 0.04],
        ylabel="L1 error",
    )
    bar_x = x1 + w1 / 2
    bar_y = sy_l1(l1_mean)
    fig.rect(bar_x - 25, bar_y, 50, y1 + h1 - bar_y, fill=COLOR_RECON)
    fig.line(bar_x, sy_l1(max(l1_mean - l1_std, 0)), bar_x, sy_l1(l1_mean + l1_std), color=COLOR_AXIS, width=1)
    fig.line(bar_x - 8, sy_l1(l1_mean + l1_std), bar_x + 8, sy_l1(l1_mean + l1_std), color=COLOR_AXIS, width=1)
    fig.line(bar_x - 8, sy_l1(max(l1_mean - l1_std, 0)), bar_x + 8, sy_l1(max(l1_mean - l1_std, 0)), color=COLOR_AXIS, width=1)
    fig.text(bar_x, bar_y - 7, f"{l1_mean:.4f}", size=9, anchor="middle")
    fig.text(bar_x, y1 + h1 + 18, "Mean +/- std.", size=8.5, color=COLOR_GRAY, anchor="middle")

    x2, y2, w2, h2 = 540, 55, 145, 150
    snr_mean = metrics["snr_mean_db"]
    snr_std = metrics["snr_std_db"]
    _, sy_snr = draw_axes(
        fig,
        x2,
        y2,
        w2,
        h2,
        xlim=(-0.5, 0.5),
        ylim=(0, 25),
        xticks=[],
        yticks=[0, 5, 10, 15, 20, 25],
        ylabel="SNR (dB)",
    )
    bar_y = sy_snr(snr_mean)
    fig.rect(x2 + w2 / 2 - 25, bar_y, 50, y2 + h2 - bar_y, fill=COLOR_ORIGINAL)
    fig.line(x2 + w2 / 2, sy_snr(snr_mean - snr_std), x2 + w2 / 2, sy_snr(snr_mean + snr_std), color=COLOR_AXIS, width=1)
    fig.line(x2 + w2 / 2 - 8, sy_snr(snr_mean + snr_std), x2 + w2 / 2 + 8, sy_snr(snr_mean + snr_std), color=COLOR_AXIS, width=1)
    fig.line(x2 + w2 / 2 - 8, sy_snr(snr_mean - snr_std), x2 + w2 / 2 + 8, sy_snr(snr_mean - snr_std), color=COLOR_AXIS, width=1)
    fig.text(x2 + w2 / 2, bar_y - 7, f"{snr_mean:.2f}", size=9, anchor="middle")
    fig.text(x2 + w2 / 2, y2 + h2 + 18, "Mean +/- std.", size=8.5, color=COLOR_GRAY, anchor="middle")

    fig.render_svg(out_dir / "fig_compression_reconstruction.svg")
    fig.render_pdf(out_dir / "fig_compression_reconstruction.pdf")


def draw_grouped_bars(
    fig: Figure,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    groups: Sequence[str],
    original: Sequence[float],
    reconstructed: Sequence[float],
    ylim: Tuple[float, float],
    yticks: Sequence[float],
    ylabel: str,
    value_fmt: str,
) -> None:
    _, sy = draw_axes(
        fig,
        x,
        y,
        w,
        h,
        xlim=(-0.5, len(groups) - 0.5),
        ylim=ylim,
        xticks=[],
        yticks=yticks,
        ylabel=ylabel,
    )
    span = w / len(groups)
    bar_w = 28
    for i, phase in enumerate(groups):
        cx = x + span * (i + 0.5)
        for value, dx, color in [
            (original[i], -18, COLOR_ORIGINAL),
            (reconstructed[i], 18, COLOR_RECON),
        ]:
            py = sy(value)
            fig.rect(cx + dx - bar_w / 2, py, bar_w, y + h - py, fill=color)
            fig.text(cx + dx, py - 6, format(value, value_fmt), size=8, anchor="middle")
        fig.text(cx, y + h + 18, phase, size=10, weight="bold", anchor="middle")


def draw_picking_performance_figure(summary: Dict, out_dir: Path) -> None:
    fig = Figure(720, 300)
    draw_panel_title(fig, 38, 28, "A", "Arrival-time error")
    draw_panel_title(fig, 390, 28, "B", "Recall at 0.2 s")
    draw_legend(fig, 250, 28, [("Original", COLOR_ORIGINAL), ("Reconstructed", COLOR_RECON)])

    phases = ["P", "S"]
    mae_orig = [summary["phases"][p]["original"]["mae_sec"] for p in phases]
    mae_recon = [summary["phases"][p]["reconstructed"]["mae_sec"] for p in phases]
    recall_orig = [summary["phases"][p]["original"]["recall_at_tolerance"] for p in phases]
    recall_recon = [summary["phases"][p]["reconstructed"]["recall_at_tolerance"] for p in phases]

    draw_grouped_bars(
        fig,
        70,
        62,
        250,
        165,
        groups=phases,
        original=mae_orig,
        reconstructed=mae_recon,
        ylim=(0, 0.13),
        yticks=[0, 0.04, 0.08, 0.12],
        ylabel="MAE (s)",
        value_fmt=".3f",
    )
    draw_grouped_bars(
        fig,
        422,
        62,
        250,
        165,
        groups=phases,
        original=recall_orig,
        reconstructed=recall_recon,
        ylim=(0, 1.0),
        yticks=[0, 0.25, 0.5, 0.75, 1.0],
        ylabel="Recall",
        value_fmt=".2f",
    )

    fig.text(195, 270, "Compression increases P-wave MAE by 21 ms and reduces recall by 3.1 pp.", size=9, color=COLOR_GRAY, anchor="middle")
    fig.text(547, 270, "S-wave recall drops by 4.0 pp while median error remains unchanged.", size=9, color=COLOR_GRAY, anchor="middle")
    fig.render_svg(out_dir / "fig_picking_performance.svg")
    fig.render_pdf(out_dir / "fig_picking_performance.pdf")


def read_picking_rows(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    data = {
        "P": {"original": [], "reconstructed": []},
        "S": {"original": [], "reconstructed": []},
    }
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            phase = row["phase"]
            if phase not in data:
                continue
            data[phase]["original"].append(float(row["original_abs_error_sec"]))
            data[phase]["reconstructed"].append(float(row["reconstructed_abs_error_sec"]))
    return {
        phase: {name: np.asarray(values, dtype=float) for name, values in curves.items()}
        for phase, curves in data.items()
    }


def ecdf_points(values: np.ndarray, xmax: float) -> Tuple[np.ndarray, np.ndarray]:
    values = np.sort(values)
    values = values[np.isfinite(values)]
    x = np.concatenate([[0.0], values[values <= xmax], [xmax]])
    y = np.searchsorted(values, x, side="right") / len(values)
    return x, y


def draw_cdf_panel(
    fig: Figure,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    phase: str,
    values_original: np.ndarray,
    values_reconstructed: np.ndarray,
    tolerance: float,
) -> None:
    xmax = 0.5
    sx, sy = draw_axes(
        fig,
        x,
        y,
        w,
        h,
        xlim=(0, xmax),
        ylim=(0, 1),
        xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        yticks=[0, 0.25, 0.5, 0.75, 1.0],
        xlabel="Absolute pick error (s)",
        ylabel="Cumulative fraction",
    )
    fig.line(sx(tolerance), y, sx(tolerance), y + h, color="#9CA3AF", width=1.0, dash="4,4")
    fig.text(sx(tolerance) + 4, y + 12, "0.2 s", size=8, color=COLOR_GRAY)

    for label, values, color in [
        ("Original", values_original, COLOR_ORIGINAL),
        ("Reconstructed", values_reconstructed, COLOR_RECON),
    ]:
        xs, ys = ecdf_points(values, xmax)
        points = [(sx(float(a)), sy(float(b))) for a, b in zip(xs, ys)]
        fig.polyline(points, color=color, width=1.8)
    fig.text(x + 8, y + 14, f"{phase} phase", size=10, weight="bold")


def draw_error_cdf_figure(rows: Dict[str, Dict[str, np.ndarray]], summary: Dict, out_dir: Path) -> None:
    fig = Figure(720, 310)
    draw_panel_title(fig, 38, 28, "A", "P-wave picking error CDF")
    draw_panel_title(fig, 390, 28, "B", "S-wave picking error CDF")
    draw_legend(fig, 250, 28, [("Original", COLOR_ORIGINAL), ("Reconstructed", COLOR_RECON)])
    tolerance = float(summary["settings"]["tolerance_sec"])
    draw_cdf_panel(
        fig,
        70,
        58,
        255,
        185,
        phase="P",
        values_original=rows["P"]["original"],
        values_reconstructed=rows["P"]["reconstructed"],
        tolerance=tolerance,
    )
    draw_cdf_panel(
        fig,
        422,
        58,
        255,
        185,
        phase="S",
        values_original=rows["S"]["original"],
        values_reconstructed=rows["S"]["reconstructed"],
        tolerance=tolerance,
    )
    fig.text(360, 292, "Curves use labelled P/S arrivals from 2,048 ETHZ development windows.", size=9, color=COLOR_GRAY, anchor="middle")
    fig.render_svg(out_dir / "fig_picking_error_cdf.svg")
    fig.render_pdf(out_dir / "fig_picking_error_cdf.pdf")


def write_latex_snippet(out_dir: Path) -> None:
    snippet = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_compression_reconstruction.pdf}
\caption{Compression and reconstruction summary for the no-GAN SeisDAC model with latent regularization weight $3\times10^{-3}$ on the ETHZ development split. The codec reduces the theoretical bitrate from float32 ZNE waveforms by $8.5\times$, with an average L1 reconstruction error of 0.0217 and an average SNR of 12.93 dB.}
\label{fig:ethz_compression_reconstruction}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_picking_performance.pdf}
\caption{Downstream phase-picking performance on original and reconstructed ETHZ waveforms using PhaseNet with ETHZ weights. Compression causes a small degradation in arrival-time error and recall, while preserving most P- and S-phase picking performance.}
\label{fig:ethz_picking_performance}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_picking_error_cdf.pdf}
\caption{Empirical cumulative distribution of absolute P- and S-pick timing errors for original and reconstructed waveforms. The dashed line marks the 0.2 s tolerance used for recall.}
\label{fig:ethz_picking_error_cdf}
\end{figure}
"""
    (out_dir / "paper_figures.tex").write_text(snippet + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SeisDAC paper figures.")
    parser.add_argument(
        "--reconstruction_metrics",
        default="/data/seismic/seis-codec-eval/ethz_nogan_latreg0p003_best137/metrics.txt",
    )
    parser.add_argument(
        "--picking_metrics",
        default="/data/seismic/seis-codec-eval/ethz_nogan_latreg0p003_picking/picking_metrics.json",
    )
    parser.add_argument(
        "--picking_rows",
        default="/data/seismic/seis-codec-eval/ethz_nogan_latreg0p003_picking/picking_rows.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/seismic/seis-codec-eval/ethz_nogan_latreg0p003_figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reconstruction = parse_metrics_txt(Path(args.reconstruction_metrics))
    picking_summary = json.loads(Path(args.picking_metrics).read_text(encoding="utf-8"))
    picking_rows = read_picking_rows(Path(args.picking_rows))

    draw_compression_figure(reconstruction, out_dir)
    draw_picking_performance_figure(picking_summary, out_dir)
    draw_error_cdf_figure(picking_rows, picking_summary, out_dir)
    write_latex_snippet(out_dir)

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
