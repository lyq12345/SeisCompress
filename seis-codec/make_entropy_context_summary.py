"""Summarize universal, domain-specific, and causal entropy coding results."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT_DIR = "/data/seismic/seis-codec-eval/entropy_context_summary"
DATASET_PATHS = {
    "ETHZ": {
        "universal": "/data/seismic/seis-codec-eval/seisdac_entropy_ethz_calibrated_train_pmf",
        "context": "/data/seismic/seis-codec-eval/seisdac_entropy_ethz_first_order_domain_pmf",
    },
    "STEAD": {
        "universal": "/data/seismic/seis-codec-eval/seisdac_entropy_stead_ethz_pmf",
        "context": "/data/seismic/seis-codec-eval/seisdac_entropy_stead_first_order_domain_pmf",
    },
    "GEOFON": {
        "universal": "/data/seismic/seis-codec-eval/seisdac_entropy_geofon_ethz_pmf",
        "context": "/data/seismic/seis-codec-eval/seisdac_entropy_geofon_first_order_domain_pmf",
    },
}
METHODS = (
    "Fixed 10-bit",
    "Best zstd",
    "Factorized, ETHZ PMF",
    "Factorized, domain PMF",
    "First-order, domain PMF",
)
COLORS = {
    "Fixed 10-bit": "#6B7280",
    "Best zstd": "#C44E52",
    "Factorized, ETHZ PMF": "#6A4C93",
    "Factorized, domain PMF": "#2C6B73",
    "First-order, domain PMF": "#2A7F62",
}
LINE_STYLES = {
    "Fixed 10-bit": ("--", "o"),
    "Best zstd": ("-.", "s"),
    "Factorized, ETHZ PMF": (":", "^"),
    "Factorized, domain PMF": ("-", "D"),
    "First-order, domain PMF": ("-", "P"),
}


def load_json(directory: str) -> Dict:
    path = Path(directory) / "seisdac_entropy_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def full_rate_rows(data: Dict) -> List[Dict]:
    return [row for row in data["results"] if int(row["n_quantizers"]) == 9]


def row_for_coding(data: Dict, coding: str) -> Dict:
    return next(row for row in full_rate_rows(data) if row["coding"] == coding)


def best_zstd_row(data: Dict) -> Dict:
    return min(
        (row for row in full_rate_rows(data) if row["coding"].startswith("zstd")),
        key=lambda row: float(row["compressed_bps"]),
    )


def load_sample_rates(directory: str, coding: str) -> np.ndarray:
    path = Path(directory) / "seisdac_entropy_sample_rows.csv"
    values = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["n_quantizers"]) == 9 and row["coding"] == coding:
                values[int(row["sample_index"])] = float(row["compressed_bps"])
    if not values:
        raise ValueError(f"No n_quantizers=9 rows for {coding} in {path}")
    return np.asarray([values[idx] for idx in sorted(values)], dtype=np.float64)


def bootstrap_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    rounds: int,
) -> Tuple[float, float]:
    indexes = rng.integers(0, len(values), size=(rounds, len(values)))
    means = values[indexes].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def make_row(
    dataset: str,
    method: str,
    result: Dict,
    *,
    source_dir: str,
    zstd_rates: np.ndarray,
    rng: np.random.Generator,
    bootstrap_rounds: int,
) -> Dict:
    coding = str(result["coding"])
    bitrate = float(result["compressed_bps"])
    if method == "Fixed 10-bit":
        rates = np.full_like(zstd_rates, bitrate)
        ci_low = ci_high = bitrate
    else:
        rates = load_sample_rates(source_dir, coding)
        ci_low, ci_high = bootstrap_ci(rates, rng=rng, rounds=bootstrap_rounds)
    if rates.shape != zstd_rates.shape:
        raise ValueError(f"Cannot pair {dataset} {method} with zstd samples")
    delta = rates - zstd_rates
    delta_low, delta_high = bootstrap_ci(delta, rng=rng, rounds=bootstrap_rounds)
    return {
        "dataset": dataset,
        "method": method,
        "coding": coding,
        "compressed_bps": bitrate,
        "bitrate_ci95_low": ci_low,
        "bitrate_ci95_high": ci_high,
        "delta_vs_zstd_bps": float(delta.mean()),
        "delta_vs_zstd_ci95_low": delta_low,
        "delta_vs_zstd_ci95_high": delta_high,
        "compression_ratio_vs_float32": float(result["compression_ratio_vs_float32"]),
        "savings_vs_fixed_percent": float(result["savings_vs_fixed_percent"]),
        "l1_mean": float(result["l1_mean"]),
        "snr_mean_db": float(result["snr_mean_db"]),
        "entropy_encode_ms_per_window": result.get("entropy_encode_ms_per_window"),
        "entropy_decode_ms_per_window": result.get("entropy_decode_ms_per_window"),
        "source_dir": source_dir,
    }


def collect_rows(bootstrap_rounds: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, paths in DATASET_PATHS.items():
        universal_data = load_json(paths["universal"])
        context_data = load_json(paths["context"])
        zstd = best_zstd_row(context_data)
        zstd_rates = load_sample_rates(paths["context"], str(zstd["coding"]))
        specifications = (
            ("Fixed 10-bit", row_for_coding(context_data, "fixed-theoretical"), paths["context"]),
            ("Best zstd", zstd, paths["context"]),
            (
                "Factorized, ETHZ PMF",
                row_for_coding(universal_data, "rans-factorized"),
                paths["universal"],
            ),
            (
                "Factorized, domain PMF",
                row_for_coding(context_data, "rans-factorized"),
                paths["context"],
            ),
            (
                "First-order, domain PMF",
                row_for_coding(context_data, "rans-first-order"),
                paths["context"],
            ),
        )
        for method, result, source_dir in specifications:
            rows.append(
                make_row(
                    dataset,
                    method,
                    result,
                    source_dir=source_dir,
                    zstd_rates=zstd_rates,
                    rng=rng,
                    bootstrap_rounds=bootstrap_rounds,
                )
            )
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(rows: List[Dict], output_dir: Path) -> None:
    datasets = tuple(DATASET_PATHS)
    x = np.arange(len(datasets), dtype=np.float64)
    width = 0.16
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for method_idx, method in enumerate(METHODS):
        method_rows = [
            next(row for row in rows if row["dataset"] == dataset and row["method"] == method)
            for dataset in datasets
        ]
        offsets = x + (method_idx - (len(METHODS) - 1) / 2) * width
        bitrates = [row["compressed_bps"] for row in method_rows]
        savings = [row["savings_vs_fixed_percent"] for row in method_rows]
        axes[0].bar(offsets, bitrates, width, color=COLORS[method], label=method)
        axes[1].bar(offsets, savings, width, color=COLORS[method], label=method)

    for axis in axes:
        axis.set_xticks(x, datasets)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("A  Measured full-rate bitrate", loc="left", fontweight="bold")
    axes[0].set_ylabel("Bitrate (bps)")
    axes[0].set_ylim(0, 1225)
    axes[1].set_title("B  Saving relative to fixed 10-bit codes", loc="left", fontweight="bold")
    axes[1].set_ylabel("Bitrate saving (%)")
    axes[1].set_ylim(-2, 52)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(output_dir / "fig_entropy_context_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig_entropy_context_comparison.svg", bbox_inches="tight")
    fig.savefig(
        output_dir / "fig_entropy_context_comparison.png",
        bbox_inches="tight",
        dpi=180,
    )
    plt.close(fig)


def rate_distortion_rows(data: Dict, method: str) -> List[Dict]:
    rows = data["results"]
    output = []
    for n_quantizers in range(1, 10):
        candidates = [row for row in rows if int(row["n_quantizers"]) == n_quantizers]
        if method == "Fixed 10-bit":
            selected = next(row for row in candidates if row["coding"] == "fixed-theoretical")
        elif method == "Best zstd":
            selected = min(
                (row for row in candidates if row["coding"].startswith("zstd")),
                key=lambda row: float(row["compressed_bps"]),
            )
        elif method in {"Factorized, ETHZ PMF", "Factorized, domain PMF"}:
            selected = next(row for row in candidates if row["coding"] == "rans-factorized")
        elif method == "First-order, domain PMF":
            selected = next(row for row in candidates if row["coding"] == "rans-first-order")
        else:
            raise ValueError(f"Unsupported R-D method: {method}")
        output.append(selected)
    return output


def draw_rate_distortion_figure(output_dir: Path) -> None:
    datasets = tuple(DATASET_PATHS)
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0))
    for column, dataset in enumerate(datasets):
        paths = DATASET_PATHS[dataset]
        universal_data = load_json(paths["universal"])
        context_data = load_json(paths["context"])
        for method in METHODS:
            source = universal_data if method == "Factorized, ETHZ PMF" else context_data
            curve = rate_distortion_rows(source, method)
            bitrate = [float(row["compressed_bps"]) for row in curve]
            line_style, marker = LINE_STYLES[method]
            axes[0, column].plot(
                bitrate,
                [float(row["l1_mean"]) for row in curve],
                color=COLORS[method],
                linestyle=line_style,
                marker=marker,
                markersize=4.2,
                linewidth=1.6,
                label=method,
            )
            axes[1, column].plot(
                bitrate,
                [float(row["snr_mean_db"]) for row in curve],
                color=COLORS[method],
                linestyle=line_style,
                marker=marker,
                markersize=4.2,
                linewidth=1.6,
                label=method,
            )

        axes[0, column].set_title(
            f"{'ABC'[column]}  {dataset}: waveform error", loc="left", fontweight="bold"
        )
        axes[1, column].set_title(
            f"{'DEF'[column]}  {dataset}: signal fidelity", loc="left", fontweight="bold"
        )
        for row in range(2):
            axis = axes[row, column]
            axis.set_xlim(0, 1200)
            axis.set_xlabel("Measured bitrate (bps)")
            axis.grid(color="#D1D5DB", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0, column].set_ylabel("Mean L1 error")
        axes[1, column].set_ylabel("Mean SNR (dB)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=2.0, w_pad=1.4)
    for extension in ("pdf", "svg", "png"):
        kwargs = {"dpi": 180} if extension == "png" else {}
        fig.savefig(
            output_dir / f"fig_entropy_context_rate_distortion.{extension}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def write_latex(rows: List[Dict], path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Measured full-rate bitrate of lossless entropy coders for SeisDAC RVQ indices. Probability tables use training data only, either the shared ETHZ table or a domain-specific table. Rates include a 28-byte independently decodable window header.}",
        r"\label{tab:entropy_context}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Method & Bitrate (bps) & Ratio & Saving \\",
        r"\midrule",
    ]
    for dataset in DATASET_PATHS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for idx, row in enumerate(dataset_rows):
            dataset_cell = dataset if idx == 0 else ""
            method = row["method"].replace("First-order", r"First-order")
            lines.append(
                f"{dataset_cell} & {method} & {row['compressed_bps']:.1f} & "
                f"{row['compression_ratio_vs_float32']:.2f}$\\times$ & "
                f"{row['savings_vs_fixed_percent']:+.1f}\\% \\\\"
            )
        lines.append(r"\midrule" if dataset != tuple(DATASET_PATHS)[-1] else r"\bottomrule")
    lines.extend(
        [
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{figure*}[t]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{fig_entropy_context_rate_distortion.pdf}",
            r"\caption{Rate--distortion curves for SeisDAC with measured lossless code-stream rates. Each curve sweeps one through nine RVQ codebooks. Entropy coding changes the rate but preserves the decoded RVQ indices and waveform distortion at each operating point.}",
            r"\label{fig:entropy_context_rd}",
            r"\end{figure*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(rows: List[Dict], path: Path) -> None:
    lines = ["=== Entropy model comparison (n_quantizers=9) ==="]
    for dataset in DATASET_PATHS:
        lines.append("")
        lines.append(dataset)
        for row in (item for item in rows if item["dataset"] == dataset):
            lines.append(
                f"  {row['method']:30s} {row['compressed_bps']:7.1f} bps "
                f"({row['compression_ratio_vs_float32']:5.2f}x, "
                f"saving={row['savings_vs_fixed_percent']:+5.1f}%)"
            )
    lines.extend(
        [
            "",
            "95% confidence intervals in the CSV/JSON use paired per-window bootstrap resampling.",
            "Entropy coding is lossless with respect to RVQ indices; reconstruction metrics are unchanged.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap_rounds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_rounds < 100:
        raise ValueError("bootstrap_rounds must be at least 100")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.bootstrap_rounds, args.seed)
    write_csv(rows, output_dir / "entropy_context_summary.csv")
    (output_dir / "entropy_context_summary.json").write_text(
        json.dumps(
            {
                "bootstrap_rounds": args.bootstrap_rounds,
                "bootstrap_seed": args.seed,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_summary(rows, output_dir / "entropy_context_summary.txt")
    write_latex(rows, output_dir / "entropy_context_summary.tex")
    draw_figure(rows, output_dir)
    draw_rate_distortion_figure(output_dir)
    print(f"Saved entropy summary: {output_dir / 'entropy_context_summary.txt'}")
    print(f"Saved comparison figure: {output_dir / 'fig_entropy_context_comparison.pdf'}")
    print(
        "Saved R-D figure: "
        f"{output_dir / 'fig_entropy_context_rate_distortion.pdf'}"
    )


if __name__ == "__main__":
    main()
