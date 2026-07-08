#!/usr/bin/env python3
"""Split VCTK speakers into train/val CSV file lists for audiotools."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def find_wav_root(vctk_root: Path) -> Path:
    candidates: list[Path] = [
        vctk_root,
        vctk_root / "wav48_silence_trimmed",
        vctk_root / "wav48",
    ]
    for path in vctk_root.rglob("wav48_silence_trimmed"):
        if path.is_dir():
            candidates.append(path)
    for path in vctk_root.rglob("wav48"):
        if path.is_dir() and any(path.glob("p*")):
            candidates.append(path)

    best = None
    best_count = 0
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        speakers = [p for p in candidate.iterdir() if p.is_dir() and p.name.startswith("p")]
        if len(speakers) > best_count:
            best = candidate
            best_count = len(speakers)
    if best is None:
        raise FileNotFoundError(f"Could not find VCTK wav root under {vctk_root}")
    return best


def write_csv(paths: list[Path], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path"])
        writer.writeheader()
        for path in sorted(paths):
            writer.writerow({"path": str(path.resolve())})


def rebuild_split(
    wav_root: Path,
    train_csv: Path,
    val_csv: Path,
    val_ratio: float,
    seed: int,
) -> None:
    speakers = sorted(p for p in wav_root.iterdir() if p.is_dir() and p.name.startswith("p"))
    if not speakers:
        raise RuntimeError(f"No speaker folders found in {wav_root}")

    rng = random.Random(seed)
    speakers = speakers[:]
    rng.shuffle(speakers)
    n_val = max(1, int(len(speakers) * val_ratio))
    val_speakers = speakers[:n_val]
    train_speakers = speakers[n_val:]

    train_files: list[Path] = []
    val_files: list[Path] = []
    for speaker in train_speakers:
        train_files.extend(speaker.glob("*.wav"))
    for speaker in val_speakers:
        val_files.extend(speaker.glob("*.wav"))

    if not train_files or not val_files:
        raise RuntimeError("Train or validation split is empty.")

    write_csv(train_files, train_csv)
    write_csv(val_files, val_csv)

    print(f"VCTK wav root: {wav_root}")
    print(f"Train speakers: {len(train_speakers)}, files: {len(train_files)} -> {train_csv}")
    print(f"Val speakers:   {len(val_speakers)}, files: {len(val_files)} -> {val_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split VCTK into train/val CSV lists for DAC training."
    )
    parser.add_argument(
        "--vctk-root",
        type=Path,
        default=Path("/data/audio/vctk"),
        help="Root directory where VCTK was extracted.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("/data/audio/vctk_train.csv"),
        help="CSV file listing training audio paths.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("/data/audio/vctk_val.csv"),
        help="CSV file listing validation audio paths.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wav_root = find_wav_root(args.vctk_root.resolve())
    rebuild_split(wav_root, args.train_csv, args.val_csv, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
