#!/usr/bin/env python3
"""Download and prepare datasets for descript-audio-codec training."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT: Path
CACHE_DIR: Path
LOG_FILE: Path

AZURE_DNS_URL = (
    "https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
)
ZENODO_DAPS = (
    "https://zenodo.org/api/records/4660670/files/daps.tar.gz/content"
)
ZENODO_MUSDB = (
    "https://zenodo.org/api/records/3338373/files/musdb18hq.zip/content"
)
ZENODO_VOCALSET = (
    "https://zenodo.org/api/records/1442513/files/VocalSet11.zip/content"
)
VCTK_URL = (
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/"
    "VCTK-Corpus.zip?sequence=6&isAllowed=y"
)
CV_CORPUS_VERSION = "cv-corpus-17.0-2024-03-15"
CV_BASE_URL = (
    "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/"
    f"{CV_CORPUS_VERSION}"
)
AUDIOSET_CSV_URLS = {
    "balanced_train_segments.csv": (
        "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
        "balanced_train_segments.csv"
    ),
    "unbalanced_train_segments.csv": (
        "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
        "unbalanced_train_segments.csv"
    ),
    "eval_segments.csv": (
        "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
        "eval_segments.csv"
    ),
}
JAMENDO_REPO = "https://github.com/MTG/mtg-jamendo-dataset.git"

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}

DNS_BLOBS: dict[str, list[str]] = {
    "emotional_speech": [
        "clean_fullband/datasets_fullband.clean_fullband.emotional_speech_000_NA_NA.tar.bz2",
    ],
    "vocalset": [
        "clean_fullband/datasets_fullband.clean_fullband.VocalSet_48kHz_mono_000_NA_NA.tar.bz2",
    ],
    "french_speech": [
        f"clean_fullband/datasets_fullband.clean_fullband.french_speech_{i:03d}_NA_NA.tar.bz2"
        for i in range(9)
    ],
    "german_speech": [
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_000_0.00_3.47.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_001_3.47_3.64.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_002_3.64_3.74.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_003_3.74_3.81.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_004_3.81_3.86.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_005_3.86_3.91.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_006_3.91_3.96.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_007_3.96_4.00.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_008_4.00_4.04.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_009_4.04_4.08.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_010_4.08_4.12.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_011_4.12_4.16.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_012_4.16_4.21.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_013_4.21_4.26.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_014_4.26_4.33.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.german_speech_015_4.33_4.43.tar.bz2",
    ]
    + [
        f"clean_fullband/datasets_fullband.clean_fullband.german_speech_{i:03d}_NA_NA.tar.bz2"
        for i in range(16, 43)
    ],
    "read_speech": [
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_001_3.75_3.88.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_002_3.88_3.96.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_003_3.96_4.02.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_004_4.02_4.06.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_005_4.06_4.10.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_006_4.10_4.13.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_007_4.13_4.16.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_008_4.16_4.19.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_009_4.19_4.21.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_010_4.21_4.24.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_011_4.24_4.26.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_012_4.26_4.29.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_013_4.29_4.31.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_014_4.31_4.33.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_015_4.33_4.35.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_016_4.35_4.38.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_017_4.38_4.40.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_018_4.40_4.42.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_019_4.42_4.45.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_020_4.45_4.48.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_021_4.48_4.52.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_022_4.52_4.57.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_023_4.57_4.67.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.read_speech_024_4.67_NA.tar.bz2",
    ]
    + [
        f"clean_fullband/datasets_fullband.clean_fullband.read_speech_{i:03d}_NA_NA.tar.bz2"
        for i in range(25, 40)
    ],
    "russian_speech": [
        "clean_fullband/datasets_fullband.clean_fullband.russian_speech_000_0.00_4.31.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.russian_speech_001_4.31_NA.tar.bz2",
    ],
    "spanish_speech": [
        "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_000_0.00_4.09.tar.bz2",
        "clean_fullband/datasets_fullband.clean_fullband.spanish_speech_001_4.09_NA.tar.bz2",
    ]
    + [
        f"clean_fullband/datasets_fullband.clean_fullband.spanish_speech_{i:03d}_NA_NA.tar.bz2"
        for i in range(2, 9)
    ],
    "vctk_dns": [
        f"clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_{i:03d}.tar.bz2"
        for i in range(5)
    ],
}

DNS_FOLDER_MAP = {
    "emotional_speech": "emotional_speech",
    "vocalset": "VocalSet_48kHz_mono",
    "french_speech": "french_speech",
    "german_speech": "german_speech",
    "read_speech": "read_speech",
    "russian_speech": "russian_speech",
    "spanish_speech": "spanish_speech",
    "vctk_dns": "vctk_wav48_silence_trimmed",
}

TIER_DATASETS = {
    "minimal": [
        "daps",
        "musdb",
        "vctk",
        "dns_emotional_speech",
        "dns_vocalset",
    ],
    "speech": [
        "daps",
        "musdb",
        "vctk",
        "dns_emotional_speech",
        "dns_vocalset",
        "dns_french_speech",
        "dns_german_speech",
        "dns_russian_speech",
        "dns_spanish_speech",
        "dns_read_speech",
        "common_voice_en",
    ],
    "full": [
        "daps",
        "musdb",
        "vctk",
        "vocalset_zenodo",
        "common_voice_en",
        "common_voice_multilingual",
        "dns_emotional_speech",
        "dns_french_speech",
        "dns_german_speech",
        "dns_russian_speech",
        "dns_spanish_speech",
        "dns_read_speech",
        "jamendo",
        "audioset_balanced",
        "audioset_eval",
    ],
}

ALL_DATASETS = sorted(
    set(sum(TIER_DATASETS.values(), []) + ["audioset_unbalanced"])
)

CV_LANGUAGES = {
    "common_voice_en": ["en"],
    "common_voice_multilingual": ["fr", "de", "ru", "es"],
}


@dataclass
class DatasetStatus:
    name: str
    ready: bool
    audio_files: int
    path: Path
    note: str = ""


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], **kwargs) -> None:
    log(f"RUN: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def count_audio(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            if Path(name).suffix.lower() in AUDIO_EXTS:
                total += 1
    return total


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def download_file(url: str, dest: Path, dry_run: bool = False) -> None:
    ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"Already downloaded: {dest.name}")
        return
    if dry_run:
        log(f"[dry-run] would download {url} -> {dest}")
        return
    cmd = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5", "-C", "-", "-o", str(dest), url]
    run(cmd)


def extract_archive(archive: Path, dest: Path, dry_run: bool = False) -> None:
    if dry_run:
        log(f"[dry-run] would extract {archive} -> {dest}")
        return
    ensure_dir(dest)
    suffix = "".join(archive.suffixes).lower()
    log(f"Extracting {archive.name} -> {dest}")
    if suffix.endswith(".tar.gz") or suffix.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest)
    elif suffix.endswith(".tar.bz2"):
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(dest)
    elif suffix.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive type: {archive}")


def dns_root() -> Path:
    return CACHE_DIR / "dns" / "datasets_fullband"


def dns_category_dir(category: str) -> Path:
    folder = DNS_FOLDER_MAP[category]
    return dns_root() / "clean_fullband" / folder


def publish_dns_category(category: str, public_name: str) -> None:
    src = dns_category_dir(category)
    if not src.exists():
        return
    dst = DATA_ROOT / public_name
    if public_name == "vctk":
        wav_root = src
        if not count_audio(wav_root):
            for candidate in src.rglob("wav48_silence_trimmed"):
                if candidate.is_dir():
                    wav_root = candidate
                    break
        replace_symlink(dst, wav_root)
    else:
        replace_symlink(dst, src)


def download_dns_category(
    category: str,
    shard_limit: int | None,
    dry_run: bool = False,
) -> None:
    blobs = DNS_BLOBS[category]
    if shard_limit is not None:
        blobs = blobs[:shard_limit]
    for blob in blobs:
        archive = CACHE_DIR / "dns_archives" / blob.replace("/", "__")
        if not archive.exists():
            download_file(f"{AZURE_DNS_URL}/{blob}", archive, dry_run=dry_run)
        if dry_run:
            continue
        extract_archive(archive, dns_root())


def dataset_ready_daps() -> DatasetStatus:
    path = DATA_ROOT / "daps"
    train = path / "train"
    ready = count_audio(train) >= 50
    return DatasetStatus("daps", ready, count_audio(path), path)


def dataset_ready_musdb() -> DatasetStatus:
    path = DATA_ROOT / "musdb" / "train"
    ready = count_audio(path) >= 50
    return DatasetStatus("musdb", ready, count_audio(path), path)


def dataset_ready_simple(name: str, min_files: int = 20) -> DatasetStatus:
    path = DATA_ROOT / name
    count = count_audio(path)
    return DatasetStatus(name, count >= min_files, count, path)


def prepare_daps(dry_run: bool = False) -> None:
    status = dataset_ready_daps()
    if status.ready and (DATA_ROOT / "daps" / "train").exists():
        log("DAPS already organized; skipping.")
        return

    archive = CACHE_DIR / "daps.tar.gz"
    extract_dir = CACHE_DIR / "daps_extract"
    daps_dir = DATA_ROOT / "daps"
    if count_audio(daps_dir) < 50:
        download_file(ZENODO_DAPS, archive, dry_run=dry_run)
        if not dry_run:
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_archive(archive, extract_dir)
            ensure_dir(daps_dir)
            for produced in extract_dir.rglob("produced"):
                if produced.is_dir():
                    for wav in produced.glob("*.wav"):
                        shutil.copy2(wav, daps_dir / wav.name)
            if count_audio(daps_dir) < 50:
                for wav in extract_dir.rglob("*.wav"):
                    if "breaths" in str(wav):
                        continue
                    target = daps_dir / wav.name
                    if not target.exists():
                        shutil.copy2(wav, target)

    if dry_run:
        log("[dry-run] would run organize_daps.py")
        return

    if count_audio(daps_dir / "train") < 50:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "organize_daps.py"),
            "--dataset",
            "daps",
            "--data_path",
            str(DATA_ROOT),
        ]
        run(cmd)


def prepare_musdb(dry_run: bool = False) -> None:
    if dataset_ready_musdb().ready:
        log("MUSDB already prepared; skipping.")
        return
    archive = CACHE_DIR / "musdb18hq.zip"
    extract_dir = CACHE_DIR / "musdb_extract"
    download_file(ZENODO_MUSDB, archive, dry_run=dry_run)
    if dry_run:
        return
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_archive(archive, extract_dir)
    train_src = None
    for candidate in extract_dir.rglob("train"):
        if candidate.is_dir() and any(candidate.iterdir()):
            train_src = candidate
            break
    if train_src is None:
        raise RuntimeError("Could not find MUSDB train split after extraction.")
    replace_symlink(DATA_ROOT / "musdb" / "train", train_src)


def prepare_vctk(dry_run: bool = False) -> None:
    if dataset_ready_simple("vctk", min_files=100).ready:
        log("VCTK already prepared; skipping.")
        return
    archive = CACHE_DIR / "VCTK-Corpus.zip"
    extract_dir = CACHE_DIR / "vctk_extract"
    download_file(VCTK_URL, archive, dry_run=dry_run)
    if dry_run:
        return
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_archive(archive, extract_dir)
    wav_root = None
    for candidate in extract_dir.rglob("wav48_silence_trimmed"):
        if candidate.is_dir():
            wav_root = candidate
            break
    if wav_root is None:
        raise RuntimeError("Could not find wav48_silence_trimmed in VCTK archive.")
    replace_symlink(DATA_ROOT / "vctk", wav_root)


def prepare_vocalset_zenodo(dry_run: bool = False) -> None:
    if dataset_ready_simple("vocalset", min_files=50).ready:
        log("VocalSet already prepared; skipping.")
        return
    archive = CACHE_DIR / "VocalSet11.zip"
    extract_dir = CACHE_DIR / "vocalset_extract"
    download_file(ZENODO_VOCALSET, archive, dry_run=dry_run)
    if dry_run:
        return
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_archive(archive, extract_dir)
    wav_root = extract_dir
    for candidate in extract_dir.rglob("*"):
        if candidate.is_dir() and count_audio(candidate) >= 50:
            wav_root = candidate
            break
    replace_symlink(DATA_ROOT / "vocalset", wav_root)


def prepare_dns(
    category: str,
    public_name: str,
    shard_limit: int | None,
    dry_run: bool = False,
) -> None:
    if dataset_ready_simple(public_name, min_files=20).ready:
        log(f"DNS {public_name} already prepared; skipping.")
        return
    download_dns_category(category, shard_limit=shard_limit, dry_run=dry_run)
    if not dry_run:
        publish_dns_category(category, public_name)


def prepare_common_voice(languages: list[str], dest_name: str, dry_run: bool = False) -> None:
    dest = DATA_ROOT / dest_name.strip("/")
    if count_audio(dest) >= 100:
        log(f"{dest_name} already prepared; skipping.")
        return
    ensure_dir(dest)
    for lang in languages:
        archive_name = f"{CV_CORPUS_VERSION}-{lang}.tar.gz"
        url = f"{CV_BASE_URL}/{lang}/{archive_name}"
        archive = CACHE_DIR / "common_voice" / archive_name
        extract_dir = CACHE_DIR / "common_voice" / lang
        download_file(url, archive, dry_run=dry_run)
        if dry_run:
            continue
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_archive(archive, extract_dir)
        clips = None
        for candidate in extract_dir.rglob("clips"):
            if candidate.is_dir():
                clips = candidate
                break
        if clips is None:
            raise RuntimeError(f"Could not find clips/ for Common Voice language {lang}.")
        lang_dest = dest / lang
        replace_symlink(lang_dest, clips)


def prepare_jamendo(dry_run: bool = False) -> None:
    dest = DATA_ROOT / "jamendo"
    if count_audio(dest) >= 1000:
        log("Jamendo already prepared; skipping.")
        return
    repo_dir = CACHE_DIR / "mtg-jamendo-dataset"
    if dry_run:
        log("[dry-run] would clone MTG jamendo repo and run download.py")
        return
    if not (repo_dir / ".git").exists():
        run(["git", "clone", "--depth", "1", JAMENDO_REPO, str(repo_dir)])
    download_dir = CACHE_DIR / "jamendo_download"
    ensure_dir(download_dir)
    cmd = [
        sys.executable,
        str(repo_dir / "scripts" / "download" / "download.py"),
        "--dataset",
        "autotagging_moodtheme",
        "--type",
        "audio-low",
        "--from",
        "mtg-fast",
        "--unpack",
        "--remove",
        str(download_dir),
    ]
    run(cmd)
    audio_root = None
    for candidate in download_dir.rglob("autotagging_moodtheme"):
        if candidate.is_dir() and count_audio(candidate) > 0:
            audio_root = candidate
            break
    if audio_root is None:
        for candidate in download_dir.rglob("audio"):
            if candidate.is_dir() and count_audio(candidate) > 1000:
                audio_root = candidate
                break
    if audio_root is None:
        raise RuntimeError("Could not locate Jamendo audio after download.")
    replace_symlink(dest, audio_root)


def download_audioset_csvs(dry_run: bool = False) -> Path:
    csv_dir = DATA_ROOT / "audioset" / "data"
    ensure_dir(csv_dir)
    for name, url in AUDIOSET_CSV_URLS.items():
        download_file(url, csv_dir / name, dry_run=dry_run)
    return csv_dir


def prepare_audioset_from_csv(
    csv_name: str,
    out_subdir: str,
    max_clips: int | None,
    dry_run: bool = False,
) -> None:
    out_dir = DATA_ROOT / "audioset" / "data" / out_subdir
    min_files = max_clips or 50
    if count_audio(out_dir) >= min_files:
        log(f"AudioSet {out_subdir} already prepared; skipping.")
        return
    csv_dir = download_audioset_csvs(dry_run=dry_run)
    if dry_run:
        log(f"[dry-run] would download AudioSet {out_subdir} clips with yt-dlp")
        return
    if shutil.which("yt-dlp") is None:
        log("Installing yt-dlp for AudioSet download...")
        run([sys.executable, "-m", "pip", "install", "yt-dlp"])
    ensure_dir(out_dir)
    csv_path = csv_dir / csv_name
    lines = csv_path.read_text(encoding="utf-8").splitlines()[1:]
    if max_clips is not None:
        lines = lines[:max_clips]
    for idx, line in enumerate(lines):
        parts = line.split(",")
        if len(parts) < 3:
            continue
        ytid = parts[0].strip()
        start = parts[1].strip()
        end = parts[2].strip()
        out_wav = out_dir / f"{ytid}_{start}_{end}.wav"
        if out_wav.exists():
            continue
        url = f"https://www.youtube.com/watch?v={ytid}"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cmd = [
                "yt-dlp",
                url,
                "-x",
                "--audio-format",
                "wav",
                "--download-sections",
                f"*{start}-{end}",
                "-o",
                str(tmp_path / f"{ytid}.%(ext)s"),
                "--quiet",
                "--no-warnings",
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                log(f"AudioSet clip failed ({idx + 1}/{len(lines)}): {ytid}")
                continue
            produced = list(tmp_path.glob(f"{ytid}*"))
            if produced:
                shutil.move(str(produced[0]), out_wav)
        if (idx + 1) % 25 == 0:
            log(f"AudioSet {out_subdir} progress: {idx + 1}/{len(lines)}")


def prepare_audioset_balanced(
    max_clips: int | None,
    dry_run: bool = False,
) -> None:
    prepare_audioset_from_csv(
        "balanced_train_segments.csv",
        "balanced_train_segments",
        max_clips=max_clips,
        dry_run=dry_run,
    )


def prepare_audioset_eval(dry_run: bool = False) -> None:
    prepare_audioset_from_csv(
        "eval_segments.csv",
        "eval_segments",
        max_clips=50,
        dry_run=dry_run,
    )


def write_training_config() -> Path:
  conf_path = REPO_ROOT / "conf" / "local_data.yml"
  root = str(DATA_ROOT)

  def p(*parts: str) -> str:
      return str(Path(root).joinpath(*parts))

  content = f"""# Auto-generated by scripts/prepare_datasets.py
$include:
  - conf/ablations/baseline.yml

train/build_dataset.folders:
  speech_fb:
    - {p('daps', 'train')}
  speech_hq:
    - {p('vctk')}
    - {p('vocalset')}
    - {p('read_speech')}
    - {p('french_speech')}
  speech_uq:
    - {p('emotional_speech')}
    - {p('common_voice')}
    - {p('german_speech')}
    - {p('russian_speech')}
    - {p('spanish_speech')}
  music_hq:
    - {p('musdb', 'train')}
  music_uq:
    - {p('jamendo')}
  general:
    - {p('audioset', 'data', 'unbalanced_train_segments')}
    - {p('audioset', 'data', 'balanced_train_segments')}

val/build_dataset.folders:
  speech_hq:
    - {p('daps', 'val')}
  music_hq:
    - {p('musdb', 'test')}
  general:
    - {p('audioset', 'data', 'eval_segments')}

test/build_dataset.folders:
  speech_hq:
    - {p('daps', 'test')}
  music_hq:
    - {p('musdb', 'test')}
  general:
    - {p('audioset', 'data', 'eval_segments')}
"""
  conf_path.write_text(content, encoding="utf-8")
  log(f"Wrote training config: {conf_path}")
  return conf_path


def write_env_script() -> Path:
    env_path = DATA_ROOT / "dac_data_env.sh"
    env_path.write_text(
        f"""#!/usr/bin/env bash
# Source before training: source {env_path}
export DAC_DATA_ROOT="{DATA_ROOT}"
""",
        encoding="utf-8",
    )
    env_path.chmod(0o755)
    log(f"Wrote env script: {env_path}")
    return env_path


def verify_selected(datasets: Iterable[str]) -> list[DatasetStatus]:
    checks: dict[str, DatasetStatus] = {
        "daps": dataset_ready_daps(),
        "musdb": dataset_ready_musdb(),
        "vctk": dataset_ready_simple("vctk", min_files=100),
        "vocalset": dataset_ready_simple("vocalset", min_files=50),
        "vocalset_zenodo": dataset_ready_simple("vocalset", min_files=50),
        "read_speech": dataset_ready_simple("read_speech", min_files=20),
        "french_speech": dataset_ready_simple("french_speech", min_files=20),
        "german_speech": dataset_ready_simple("german_speech", min_files=20),
        "russian_speech": dataset_ready_simple("russian_speech", min_files=20),
        "spanish_speech": dataset_ready_simple("spanish_speech", min_files=20),
        "emotional_speech": dataset_ready_simple("emotional_speech", min_files=20),
        "common_voice_en": dataset_ready_simple("common_voice", min_files=100),
        "common_voice_multilingual": dataset_ready_simple("common_voice", min_files=100),
        "jamendo": dataset_ready_simple("jamendo", min_files=1000),
        "audioset_balanced": dataset_ready_simple(
            "audioset/data/balanced_train_segments", min_files=50
        ),
        "audioset_eval": dataset_ready_simple("audioset/data/eval_segments", min_files=10),
    }
    for key in ["dns_emotional_speech", "dns_vocalset", "dns_french_speech", "dns_german_speech",
                "dns_russian_speech", "dns_spanish_speech", "dns_read_speech"]:
        public = key.replace("dns_", "")
        checks[key] = checks.get(public, dataset_ready_simple(public, min_files=20))

    return [checks[name] for name in datasets if name in checks]


def prepare_dataset(
    name: str,
    dns_shards: int | None,
    audioset_max_clips: int | None,
    dry_run: bool,
) -> None:
    log(f"=== Preparing {name} ===")
    if name == "daps":
        prepare_daps(dry_run=dry_run)
    elif name == "musdb":
        prepare_musdb(dry_run=dry_run)
    elif name == "vctk":
        prepare_vctk(dry_run=dry_run)
    elif name == "vocalset_zenodo":
        prepare_vocalset_zenodo(dry_run=dry_run)
    elif name == "common_voice_en":
        prepare_common_voice(CV_LANGUAGES["common_voice_en"], "common_voice", dry_run=dry_run)
    elif name == "common_voice_multilingual":
        prepare_common_voice(
            CV_LANGUAGES["common_voice_multilingual"], "common_voice", dry_run=dry_run
        )
    elif name == "jamendo":
        prepare_jamendo(dry_run=dry_run)
    elif name == "audioset_balanced":
        prepare_audioset_balanced(max_clips=audioset_max_clips, dry_run=dry_run)
    elif name == "audioset_eval":
        prepare_audioset_eval(dry_run=dry_run)
    elif name.startswith("dns_"):
        category = name.replace("dns_", "")
        public = "vctk" if category == "vctk_dns" else category
        prepare_dns(category, public, shard_limit=dns_shards, dry_run=dry_run)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def select_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return args.datasets
    datasets = list(TIER_DATASETS[args.tier])
    if args.tier == "full" and args.include_audioset_unbalanced:
        datasets.append("audioset_unbalanced")
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for descript-audio-codec training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tiers:
  minimal   DAPS + MUSDB + VCTK + small DNS subsets (~40 GB)
  speech    minimal + more DNS speech + Common Voice (~100+ GB with limited shards)
  full      all speech/music/general sources (TB-scale if all DNS/AudioSet shards used)

Examples:
  python scripts/prepare_datasets.py --data-root /data/audio --tier minimal
  python scripts/prepare_datasets.py --data-root /data/audio --tier speech --dns-shards 1
  python scripts/prepare_datasets.py --verify-only
""",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DAC_DATA_ROOT", "/data/audio"),
        help="Root directory for prepared datasets (default: /data/audio).",
    )
    parser.add_argument(
        "--tier",
        choices=sorted(TIER_DATASETS),
        default="minimal",
        help="Preset bundle of datasets to download.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        help="Explicit dataset list (overrides --tier).",
    )
    parser.add_argument(
        "--dns-shards",
        type=int,
        default=None,
        help="Limit DNS downloads to the first N shards per category.",
    )
    parser.add_argument(
        "--audioset-max-clips",
        type=int,
        default=None,
        help="Limit AudioSet balanced clips (default: all for full tier, 200 otherwise).",
    )
    parser.add_argument(
        "--include-audioset-unbalanced",
        action="store_true",
        help="Also fetch AudioSet unbalanced CSV (audio download not automated).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing files; do not download.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without downloading or extracting.",
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Do not write conf/local_data.yml.",
    )
    return parser.parse_args()


def main() -> int:
    global DATA_ROOT, CACHE_DIR, LOG_FILE
    args = parse_args()
    DATA_ROOT = Path(args.data_root).resolve()
    CACHE_DIR = DATA_ROOT / ".cache"
    LOG_FILE = DATA_ROOT / "dac_prepare_datasets.log"
    ensure_dir(DATA_ROOT)
    ensure_dir(CACHE_DIR)

    if args.dns_shards is None:
        args.dns_shards = 1 if args.tier in {"minimal", "speech"} else None

    if args.audioset_max_clips is None:
        args.audioset_max_clips = None if args.tier == "full" else 200

    datasets = select_datasets(args)
    log(f"Data root: {DATA_ROOT}")
    log(f"Tier: {args.tier}")
    log(f"Datasets: {', '.join(datasets)}")
    if args.dns_shards is not None:
        log(f"DNS shard limit: {args.dns_shards}")

    if args.verify_only:
        statuses = verify_selected(datasets)
        ok = True
        for status in statuses:
            mark = "OK" if status.ready else "MISSING"
            log(f"[{mark}] {status.name}: {status.audio_files} files @ {status.path}")
            ok = ok and status.ready
        return 0 if ok else 1

    if args.dry_run:
        log("Dry run complete.")
        return 0

    failures: list[str] = []
    for name in datasets:
        if name == "audioset_unbalanced":
            download_audioset_csvs(dry_run=args.dry_run)
            log(
                "AudioSet unbalanced CSV downloaded. "
                "Full unbalanced audio download is not automated (multi-TB)."
            )
            continue
        try:
            prepare_dataset(
                name,
                dns_shards=args.dns_shards,
                audioset_max_clips=args.audioset_max_clips,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            failures.append(name)
            log(f"FAILED {name}: {exc}")
            log(traceback.format_exc())

    if not args.dry_run:
        write_env_script()
        if not args.skip_config:
            write_training_config()

    statuses = verify_selected(datasets)
    log("Verification summary:")
    for status in statuses:
        mark = "OK" if status.ready else "MISSING"
        log(f"  [{mark}] {status.name}: {status.audio_files} files")

    if failures:
        log(f"Completed with failures: {', '.join(failures)}")
        return 1
    missing = [s.name for s in statuses if not s.ready]
    if missing:
        log(f"Some datasets are still incomplete: {', '.join(missing)}")
        return 1

    log("All requested datasets are ready.")
    log(
        "Next step: python scripts/train.py "
        f"--args.load {REPO_ROOT / 'conf' / 'local_data.yml'} "
        f"--save_path runs/local/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
