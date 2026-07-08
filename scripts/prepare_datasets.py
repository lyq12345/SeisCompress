#!/usr/bin/env python3
"""Download and prepare datasets required by seis-codec/train.py."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# train.py default + common alternatives
DEFAULT_SEISBENCH_DATASETS = ["ETHZ"]
ALL_SEISBENCH_DATASETS = [
    "GEOFON",
    "ETHZ",
    "STEAD",
    "MLAAPDE",
    "InstanceCounts",
    "Iquique",
    "PNW",
    "OBST2024",
]

FORESHOCK_GDRIVE_ID = "1saaRH175pSFgl0zfQWFpedgj44pJKK3_"
FORESHOCK_ZIP = "wetransfer_classify_generic_norcia-py_2024-06-24_1530.zip"
FORESHOCK_REQUIRED = [
    "dataframe_pre_NRCA.csv",
    "dataframe_post_NRCA.csv",
    "dataframe_visso_NRCA.csv",
]

REPO_ROOT: Path
DATA_ROOT: Path
SEISBENCH_CACHE: Path
SEISLM_DATA: Path
LOG_FILE: Path


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], **kwargs) -> None:
    log(f"RUN: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def ensure_seisbench_config() -> None:
    SEISBENCH_CACHE.mkdir(parents=True, exist_ok=True)
    config_path = SEISBENCH_CACHE / "config.json"
    config = {
        "component_order": "ZNE",
        "dimension_order": "NCW",
        "remote_root": (
            "https://hifis-storage.desy.de:2880/Helmholtz/HelmholtzAI/SeisBench/"
        ),
    }
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config.update(json.load(f))
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, sort_keys=True)
    log(f"SeisBench config: {config_path}")


def dataset_is_complete(name: str) -> bool:
    dataset_dir = SEISBENCH_CACHE / "datasets" / name.lower()
    return (dataset_dir / "metadata.csv").exists() and (
        dataset_dir / "waveforms.hdf5"
    ).exists()


def wait_for_in_progress_download(name: str) -> None:
    dataset_dir = SEISBENCH_CACHE / "datasets" / name.lower()
    partial_files = list(dataset_dir.glob("*.partial"))
    while partial_files:
        sizes = ", ".join(
            f"{p.name}={p.stat().st_size / 1e9:.2f}GB" for p in partial_files
        )
        log(f"Waiting for in-progress download of {name}: {sizes}")
        time.sleep(60)
        partial_files = list(dataset_dir.glob("*.partial"))


def download_seisbench_dataset(name: str) -> None:
    code = f"""
import seisbench.data as sbd
print('Loading {name}...')
ds = sbd.{name}(
    sampling_rate=100,
    component_order='ZNE',
    dimension_order='NCW',
)
print('{name} OK, traces=', len(ds))
"""
    env = os.environ.copy()
    env["SEISBENCH_CACHE_ROOT"] = str(SEISBENCH_CACHE)
    run([sys.executable, "-c", code], env=env)


def download_foreshock_aftershock() -> None:
    out_dir = SEISLM_DATA / "foreshock_aftershock_NRCA"
    out_dir.mkdir(parents=True, exist_ok=True)

    if all((out_dir / name).exists() for name in FORESHOCK_REQUIRED):
        log("Foreshock-aftershock dataset already present; skipping.")
        return

    try:
        import gdown  # noqa: F401
    except ImportError:
        log("Installing gdown for foreshock-aftershock download...")
        run([sys.executable, "-m", "pip", "install", "gdown"])

    zip_path = out_dir / FORESHOCK_ZIP
    if not zip_path.exists():
        run([
            "gdown",
            f"https://drive.google.com/uc?id={FORESHOCK_GDRIVE_ID}",
            "-O",
            str(zip_path),
        ])

    run(["unzip", "-o", str(zip_path), "-d", str(out_dir)])


def ensure_repo_data_symlink() -> None:
    link_path = REPO_ROOT / "data"
    target = SEISLM_DATA.resolve()

    if link_path.is_symlink():
        current = link_path.resolve()
        if current == target:
            log(f"Data symlink OK: {link_path} -> {target}")
            return
        log(f"Replacing data symlink: {link_path} ({current}) -> {target}")
        link_path.unlink()
    elif link_path.exists():
        raise RuntimeError(
            f"{link_path} exists and is not a symlink. "
            f"Move it aside, then rerun this script."
        )

    link_path.symlink_to(target, target_is_directory=True)
    log(f"Created data symlink: {link_path} -> {target}")


def write_env_file() -> None:
  env_path = DATA_ROOT / "seislm_env.sh"
  content = (
      "# Source this file before running seis-codec/train.py\n"
      f"export SEISBENCH_CACHE_ROOT={SEISBENCH_CACHE}\n"
  )
  env_path.write_text(content, encoding="utf-8")
  log(f"Wrote environment file: {env_path}")


def verify_setup(datasets: list[str]) -> bool:
    ok = True

    for name in datasets:
        if dataset_is_complete(name):
            log(f"[OK] SeisBench {name}")
        else:
            log(f"[MISSING] SeisBench {name}")
            ok = False

    shock_dir = SEISLM_DATA / "foreshock_aftershock_NRCA"
    if all((shock_dir / name).exists() for name in FORESHOCK_REQUIRED):
        log("[OK] Foreshock-aftershock NRCA")
    else:
        log("[MISSING] Foreshock-aftershock NRCA")
        ok = False

    link_path = REPO_ROOT / "data"
    if link_path.is_symlink() and link_path.resolve() == SEISLM_DATA.resolve():
        log(f"[OK] Repo data symlink -> {SEISLM_DATA}")
    else:
        log("[MISSING] Repo data symlink (SeisCompress/data)")
        ok = False

    return ok


def parse_args() -> argparse.Namespace:
    default_root = os.environ.get(
        "SEISCOMPRESS_DATA_ROOT",
        "/srv/disk01/yuqiao-datasets/Seismic",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare datasets for seis-codec/train.py. "
            "By default downloads ETHZ (training) and foreshock-aftershock "
            "(validation)."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(default_root),
        help=(
            "Root directory for datasets. Creates seisbench/ and seislm/ "
            "subdirectories. Defaults to $SEISCOMPRESS_DATA_ROOT or "
            "/srv/disk01/yuqiao-datasets/Seismic."
        ),
    )
    parser.add_argument(
        "--all-seisbench",
        action="store_true",
        help="Download all SeisBench datasets used by seisLM (150+ GB).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_SEISBENCH_DATASETS,
        help="Explicit SeisBench datasets to download.",
    )
    parser.add_argument(
        "--skip-foreshock",
        action="store_true",
        help="Skip foreshock-aftershock download (validation loader will fail).",
    )
    parser.add_argument(
        "--skip-symlink",
        action="store_true",
        help="Do not create SeisCompress/data symlink.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing setup, do not download.",
    )
    return parser.parse_args()


def configure_paths(data_root: Path) -> None:
    global REPO_ROOT, DATA_ROOT, SEISBENCH_CACHE, SEISLM_DATA, LOG_FILE
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = data_root.resolve()
    SEISBENCH_CACHE = DATA_ROOT / "seisbench"
    SEISLM_DATA = DATA_ROOT / "seislm"
    LOG_FILE = DATA_ROOT / "seiscompress_prepare_datasets.log"


def select_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return args.datasets
    if args.all_seisbench:
        return ALL_SEISBENCH_DATASETS
    return DEFAULT_SEISBENCH_DATASETS


def main() -> int:
    args = parse_args()
    configure_paths(args.data_root)
    datasets = select_datasets(args)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SEISBENCH_CACHE.mkdir(parents=True, exist_ok=True)
    SEISLM_DATA.mkdir(parents=True, exist_ok=True)

    log(f"Repo root: {REPO_ROOT}")
    log(f"Data root: {DATA_ROOT}")
    log(f"SeisBench cache: {SEISBENCH_CACHE}")
    log(f"SeisLM data: {SEISLM_DATA}")
    log(f"Datasets: {', '.join(datasets)}")

    if args.verify_only:
        return 0 if verify_setup(datasets) else 1

    ensure_seisbench_config()

    if not args.skip_foreshock:
        try:
            download_foreshock_aftershock()
        except Exception:
            log("Foreshock-aftershock download failed:")
            log(traceback.format_exc())

    for name in datasets:
        if dataset_is_complete(name):
            log(f"SeisBench dataset {name} already cached; skipping.")
            continue
        wait_for_in_progress_download(name)
        if dataset_is_complete(name):
            log(f"SeisBench dataset {name} finished by another process; skipping.")
            continue
        try:
            log(f"Downloading SeisBench dataset: {name}")
            download_seisbench_dataset(name)
        except Exception:
            log(f"SeisBench dataset {name} failed:")
            log(traceback.format_exc())

    if not args.skip_symlink:
        try:
            ensure_repo_data_symlink()
        except Exception:
            log("Failed to create repo data symlink:")
            log(traceback.format_exc())

    write_env_file()

    log("Verification:")
    ok = verify_setup(datasets)
    if ok:
        log("Dataset preparation complete.")
        log(f"Before training: source {DATA_ROOT / 'seislm_env.sh'}")
        log("Then: cd seis-codec && python train.py --test_run")
        return 0

    log("Dataset preparation finished with missing items. See log above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
