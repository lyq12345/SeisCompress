"""Download and verify SeisBench datasets for SeisDAC experiments.

The script prepares datasets with the same waveform configuration used by
train.py/evaluate.py: 100 Hz, ZNE component order, and NCW tensor layout.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List


DATASET_ALIASES = {
    "ethz": "ETHZ",
    "geofon": "GEOFON",
    "stead": "STEAD",
    "instance": "InstanceCountsCombined",
    "iquique": "Iquique",
    "lendb": "LenDB",
    "neic": "NEIC",
    "scedc": "SCEDC",
}


def canonical_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name.lower(), name)


def import_seisbench():
    try:
        import seisbench
        import seisbench.data as sbd
    except ImportError as exc:
        raise SystemExit(
            "Failed to import SeisBench. Run this in the same Python environment "
            "used for training/evaluation, for example:\n\n"
            "  cd /home/coder/src/SeisCompress/seis-codec\n"
            "  source .env.seis-codec\n"
            "  python prepare_seisbench_datasets.py\n"
        ) from exc
    return seisbench, sbd


def split_lengths(dataset) -> Dict[str, int]:
    lengths: Dict[str, int] = {}
    for split_name in ("train", "dev", "test"):
        split_fn = getattr(dataset, split_name, None)
        if split_fn is None:
            lengths[split_name] = 0
            continue
        try:
            lengths[split_name] = len(split_fn())
        except Exception:
            lengths[split_name] = -1
    return lengths


def format_lengths(lengths: Dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in lengths.items())


def metadata_columns(dataset, contains: Iterable[str]) -> List[str]:
    columns = getattr(getattr(dataset, "metadata", None), "columns", [])
    needles = tuple(contains)
    return [column for column in columns if any(needle in column for needle in needles)]


def verify_sample(dataset, split_name: str, num_samples: int) -> None:
    if num_samples <= 0:
        return

    split = getattr(dataset, split_name)()
    total = len(split)
    if total == 0:
        print(f"  smoke read skipped: {split_name} split is empty")
        return

    n = min(num_samples, total)
    indices = [int(round(i * (total - 1) / max(n - 1, 1))) for i in range(n)]
    for idx in indices:
        sample = split.get_sample(idx)
        waveforms, metadata = sample
        shape = getattr(waveforms, "shape", None)
        trace_name = metadata.get("trace_name", "")
        print(f"  smoke read {split_name}[{idx}]: shape={shape}, trace_name={trace_name}")


def prepare_dataset(args, sbd, dataset_name: str) -> None:
    cls = getattr(sbd, dataset_name, None)
    if cls is None:
        raise ValueError(f"Unknown SeisBench dataset '{dataset_name}'.")

    print("=" * 80)
    print(f"Preparing {dataset_name}")
    print(
        "  config: "
        f"sampling_rate={args.sample_rate}, "
        f"component_order={args.component_order}, "
        f"dimension_order={args.dimension_order}"
    )

    dataset = cls(
        sampling_rate=args.sample_rate,
        component_order=args.component_order,
        dimension_order=args.dimension_order,
    )

    print(f"  traces: {len(dataset)}")
    print(f"  splits: {format_lengths(split_lengths(dataset))}")

    split_cols = metadata_columns(dataset, ["split"])
    arrival_cols = metadata_columns(dataset, ["arrival_sample"])
    print(f"  split columns: {split_cols if split_cols else 'none'}")
    print(f"  arrival columns: {len(arrival_cols)}")
    if arrival_cols:
        print(f"  first arrival columns: {', '.join(arrival_cols[:8])}")

    verify_sample(dataset, args.smoke_split, args.smoke_samples)

    if args.preload:
        print("  preloading train/dev/test waveforms into memory cache")
        for split_name in ("train", "dev", "test"):
            split = getattr(dataset, split_name)()
            if len(split) == 0:
                continue
            print(f"  preload {split_name}: {len(split)} traces")
            split.preload_waveforms(pbar=True)

    print(f"Finished {dataset_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify STEAD/GEOFON SeisBench datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["STEAD", "GEOFON"],
        help="Dataset names or aliases. Defaults to STEAD GEOFON.",
    )
    parser.add_argument(
        "--cache_root",
        default=os.environ.get("SEISBENCH_CACHE_ROOT", "/data/seismic/seisbench"),
        help="SeisBench cache root. Must be set before importing seisbench.",
    )
    parser.add_argument("--sample_rate", type=int, default=100)
    parser.add_argument("--component_order", default="ZNE")
    parser.add_argument("--dimension_order", default="NCW")
    parser.add_argument(
        "--smoke_split",
        default="dev",
        choices=["train", "dev", "test"],
        help="Split used for sample-read verification.",
    )
    parser.add_argument(
        "--smoke_samples",
        type=int,
        default=2,
        help="Number of samples to read from each dataset after download.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload waveforms into RAM. Usually leave disabled for large datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    os.environ["SEISBENCH_CACHE_ROOT"] = str(cache_root)
    seisbench, sbd = import_seisbench()

    print(f"SeisBench cache root: {cache_root}")
    print(f"SeisBench version: {getattr(seisbench, '__version__', 'unknown')}")
    print("Requested datasets:", ", ".join(args.datasets))

    failures = []
    for requested_name in args.datasets:
        dataset_name = canonical_dataset_name(requested_name)
        try:
            prepare_dataset(args, sbd, dataset_name)
        except Exception as exc:  # Keep preparing remaining datasets.
            failures.append((dataset_name, exc))
            print(f"ERROR while preparing {dataset_name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print("=" * 80)
    if failures:
        print("Some datasets failed:")
        for dataset_name, exc in failures:
            print(f"  {dataset_name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print("All requested datasets are prepared.")
    print(f"Cache directory: {cache_root / 'datasets'}")


if __name__ == "__main__":
    main()
