"""Measure the actual bitrate of losslessly encoded SeisDAC code streams.

The existing SeisDAC rate calculation assumes a fixed-width code for every
RVQ index.  This script writes independently decodable per-window streams and
reports their measured byte sizes.  Entropy coding is lossless with respect to
the RVQ indices, so waveform distortion depends only on ``n_quantizers``.
"""

import argparse
import csv
import json
import math
import struct
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

try:
    import zstandard as zstd
except ImportError as exc:  # pragma: no cover - dependency is checked at startup.
    raise RuntimeError("zstandard is required. Run: pip3 install zstandard") from exc

from evaluate_codec_baselines import (
    DEFAULT_CHECKPOINT,
    FAMILY_COLORS,
    compute_snr_np,
    finite_mean,
    finite_std,
    format_float,
    load_waveforms,
    parse_int_list,
)
from make_paper_figures import (
    COLOR_AXIS,
    COLOR_GRAY,
    COLOR_LIGHT,
    Figure,
    draw_axes,
    draw_panel_title,
)
from rans_codec import FactorizedRansCodec, STREAM_HEADER as RANS_STREAM_HEADER


DEFAULT_OUTPUT_DIR = "/data/seismic/seis-codec-eval/seisdac_entropy_ethz"
DEFAULT_BASELINE_CSV = (
    "/data/seismic/seis-codec-eval/codec_baselines_ethz/codec_baseline_results.csv"
)

MAGIC = b"SDC1"
VERSION = 1
STREAM_HEADER = struct.Struct("<4sBBBBIII")
CODING_IDS = {
    "packed10": 0,
    "zstd-packed10": 1,
    "zstd-uint16": 2,
}
CODING_NAMES = {value: key for key, value in CODING_IDS.items()}
RESULT_ORDER = (
    "fixed-theoretical",
    "packed10",
    "zstd-packed10",
    "zstd-uint16",
    "rans-factorized",
)
RESULT_LABELS = {
    "fixed-theoretical": "Fixed-width (theoretical)",
    "packed10": "Packed 10-bit + header",
    "zstd-packed10": "zstd on packed 10-bit",
    "zstd-uint16": "zstd on uint16 indices",
    "rans-factorized": "factorized categorical + rANS",
}
RESULT_COLORS = {
    "fixed-theoretical": "#6B7280",
    "packed10": "#111827",
    "zstd-packed10": "#C44E52",
    "zstd-uint16": "#2C6B73",
    "rans-factorized": "#6A4C93",
}


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned integers into a big-endian, fixed-width bit stream."""
    if bits < 1 or bits > 32:
        raise ValueError(f"bits must be in [1, 32], got {bits}")
    flat = np.asarray(values, dtype=np.uint64).reshape(-1)
    if flat.size and int(flat.max()) >= (1 << bits):
        raise ValueError(f"A code does not fit in {bits} bits")
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    bit_matrix = ((flat[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1), bitorder="big").tobytes()


def unpack_unsigned(payload: bytes, bits: int, count: int) -> np.ndarray:
    """Inverse of :func:`pack_unsigned`; trailing byte padding is ignored."""
    expected_bytes = math.ceil(count * bits / 8)
    if len(payload) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} packed bytes, got {len(payload)}")
    flat_bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8),
        bitorder="big",
        count=count * bits,
    )
    bit_matrix = flat_bits.reshape(count, bits).astype(np.uint64)
    weights = 1 << np.arange(bits - 1, -1, -1, dtype=np.uint64)
    return (bit_matrix @ weights).astype(np.int64, copy=False)


def encode_code_stream(
    codes: np.ndarray,
    *,
    original_length: int,
    bits_per_code: int,
    coding: str,
    compressor: zstd.ZstdCompressor,
) -> bytes:
    """Encode one ``[n_quantizers, n_frames]`` RVQ index array."""
    if coding not in CODING_IDS:
        raise ValueError(f"Unsupported coding: {coding}")
    codes = np.ascontiguousarray(codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError(f"codes must have shape [n_quantizers, n_frames], got {codes.shape}")
    n_quantizers, n_frames = codes.shape
    if n_quantizers > 255 or bits_per_code > 255:
        raise ValueError("Stream header supports at most 255 quantizers/bits")

    if coding in {"packed10", "zstd-packed10"}:
        raw_payload = pack_unsigned(codes, bits_per_code)
    else:
        if bits_per_code > 16:
            raise ValueError("zstd-uint16 cannot store codes wider than 16 bits")
        raw_payload = codes.astype("<u2", copy=False).tobytes(order="C")

    if coding == "packed10":
        payload = raw_payload
    else:
        payload = compressor.compress(raw_payload)

    header = STREAM_HEADER.pack(
        MAGIC,
        VERSION,
        CODING_IDS[coding],
        n_quantizers,
        bits_per_code,
        n_frames,
        original_length,
        len(raw_payload),
    )
    return header + payload


def decode_code_stream(
    stream: bytes,
    *,
    decompressor: zstd.ZstdDecompressor,
) -> Tuple[np.ndarray, int]:
    """Decode a stream produced by :func:`encode_code_stream`."""
    if len(stream) < STREAM_HEADER.size:
        raise ValueError("Truncated SeisDAC stream header")
    (
        magic,
        version,
        coding_id,
        n_quantizers,
        bits_per_code,
        n_frames,
        original_length,
        raw_nbytes,
    ) = STREAM_HEADER.unpack_from(stream)
    if magic != MAGIC or version != VERSION:
        raise ValueError("Unsupported SeisDAC stream format")
    if coding_id not in CODING_NAMES:
        raise ValueError(f"Unknown coding id: {coding_id}")

    coding = CODING_NAMES[coding_id]
    payload = stream[STREAM_HEADER.size :]
    if coding == "packed10":
        raw_payload = payload
    else:
        raw_payload = decompressor.decompress(payload, max_output_size=raw_nbytes)
    if len(raw_payload) != raw_nbytes:
        raise ValueError(f"Decoded payload has {len(raw_payload)} bytes, expected {raw_nbytes}")

    count = n_quantizers * n_frames
    if coding in {"packed10", "zstd-packed10"}:
        flat = unpack_unsigned(raw_payload, bits_per_code, count)
    else:
        expected_bytes = count * np.dtype("<u2").itemsize
        if raw_nbytes != expected_bytes:
            raise ValueError(f"Expected {expected_bytes} uint16 bytes, got {raw_nbytes}")
        flat = np.frombuffer(raw_payload, dtype="<u2").astype(np.int64, copy=False)
    return flat.reshape(n_quantizers, n_frames), original_length


def entropy_bits(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def compute_codebook_stats(
    codes: np.ndarray,
    *,
    codebook_size: int,
    duration_sec: float,
) -> List[Dict[str, float]]:
    """Return marginal and within-window first-order entropy diagnostics."""
    n_samples, n_quantizers, n_frames = codes.shape
    rows: List[Dict[str, float]] = []
    for codebook_idx in range(n_quantizers):
        values = codes[:, codebook_idx, :]
        marginal_counts = np.bincount(values.reshape(-1), minlength=codebook_size)
        marginal_entropy = entropy_bits(marginal_counts)

        previous = values[:, :-1].reshape(-1)
        current = values[:, 1:].reshape(-1)
        pair_ids = previous.astype(np.int64) * codebook_size + current.astype(np.int64)
        _, pair_counts = np.unique(pair_ids, return_counts=True)
        previous_counts = np.bincount(previous, minlength=codebook_size)
        conditional_entropy = max(0.0, entropy_bits(pair_counts) - entropy_bits(previous_counts))

        active = int(np.count_nonzero(marginal_counts))
        max_probability = float(marginal_counts.max() / marginal_counts.sum())
        rows.append(
            {
                "codebook": codebook_idx + 1,
                "samples": n_samples,
                "frames_per_window": n_frames,
                "active_codes": active,
                "utilization_percent": 100.0 * active / codebook_size,
                "max_symbol_probability": max_probability,
                "marginal_entropy_bits_per_symbol": marginal_entropy,
                "conditional_entropy_bits_per_symbol": conditional_entropy,
                "marginal_entropy_estimate_bps": marginal_entropy * n_frames / duration_sec,
                "first_order_entropy_estimate_bps": (
                    marginal_entropy + conditional_entropy * max(0, n_frames - 1)
                )
                / duration_sec,
            }
        )
    return rows


def cumulative_stat(stats: Sequence[Dict[str, float]], n_quantizers: int, key: str) -> float:
    return float(sum(float(row[key]) for row in stats[:n_quantizers]))


@torch.no_grad()
def extract_codes(
    waveforms: Sequence[np.ndarray],
    model,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    generator = model.generator
    sample_rate = int(model.config.model.sample_rate)
    generator.eval().to(device)
    batches: List[np.ndarray] = []
    for start_idx in range(0, len(waveforms), batch_size):
        batch_np = np.stack(waveforms[start_idx : start_idx + batch_size], axis=0)
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        prepared = generator.preprocess(batch, sample_rate)
        encoder_latent = generator.encoder(prepared)
        _, codes, _, _, _, _ = generator.quantizer(
            encoder_latent, n_quantizers=int(generator.n_codebooks)
        )
        batches.append(codes.detach().cpu().numpy().astype(np.int64, copy=False))
    return np.concatenate(batches, axis=0)


@torch.no_grad()
def reconstruct_from_codes(
    codes: np.ndarray,
    references: Sequence[np.ndarray],
    model,
    *,
    device: str,
    batch_size: int,
) -> Tuple[List[float], List[float], float]:
    generator = model.generator
    l1_values: List[float] = []
    snr_values: List[float] = []
    decode_times: List[float] = []
    for start_idx in range(0, len(references), batch_size):
        stop_idx = min(start_idx + batch_size, len(references))
        code_batch = torch.from_numpy(codes[start_idx:stop_idx]).to(device=device, dtype=torch.long)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        z_q, _, _ = generator.quantizer.from_codes(code_batch)
        reconstructed = generator.decode(z_q)[..., : references[0].shape[-1]]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end = time.perf_counter()
        decode_times.extend([(end - start) / (stop_idx - start_idx)] * (stop_idx - start_idx))
        reconstructed_np = reconstructed.detach().cpu().numpy().astype(np.float32, copy=False)
        for reference, estimate in zip(references[start_idx:stop_idx], reconstructed_np):
            l1_values.append(float(np.mean(np.abs(reference - estimate))))
            snr_values.append(compute_snr_np(reference, estimate))
    return l1_values, snr_values, float(np.mean(decode_times) * 1000.0)


def evaluate_entropy_coding(
    waveforms: Sequence[np.ndarray],
    codes: np.ndarray,
    codebook_stats: Sequence[Dict[str, float]],
    model,
    quantizers: Sequence[int],
    *,
    device: str,
    batch_size: int,
    zstd_level: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    generator = model.generator
    sample_rate = int(model.config.model.sample_rate)
    n_channels = int(model.config.model.in_channels)
    original_length = int(waveforms[0].shape[-1])
    duration_sec = original_length / sample_rate
    original_bps = n_channels * 32.0 * sample_rate
    codebook_size = int(generator.codebook_size)
    bits_per_code = math.ceil(math.log2(codebook_size))
    requested = sorted(set(int(q) for q in quantizers if 1 <= int(q) <= codes.shape[1]))
    if not requested:
        raise ValueError("No valid n_quantizers values were requested")

    compressor = zstd.ZstdCompressor(level=zstd_level, write_checksum=True)
    decompressor = zstd.ZstdDecompressor()
    results: List[Dict[str, float]] = []
    sample_rows: List[Dict[str, float]] = []

    for sweep_idx, n_quantizers in enumerate(requested, start=1):
        print(f"[Entropy {sweep_idx:02d}/{len(requested):02d}] n_quantizers={n_quantizers}")
        prefix = np.ascontiguousarray(codes[:, :n_quantizers, :])
        byte_values: Dict[str, List[int]] = {coding: [] for coding in CODING_IDS}
        encode_times: Dict[str, List[float]] = {coding: [] for coding in CODING_IDS}
        decode_times: Dict[str, List[float]] = {coding: [] for coding in CODING_IDS}
        roundtrip_codes: List[np.ndarray] = []

        for sample_idx, sample_codes in enumerate(prefix):
            decoded_reference = None
            for coding in CODING_IDS:
                start = time.perf_counter()
                stream = encode_code_stream(
                    sample_codes,
                    original_length=original_length,
                    bits_per_code=bits_per_code,
                    coding=coding,
                    compressor=compressor,
                )
                middle = time.perf_counter()
                decoded, decoded_length = decode_code_stream(stream, decompressor=decompressor)
                end = time.perf_counter()
                if decoded_length != original_length or not np.array_equal(decoded, sample_codes):
                    raise RuntimeError(f"Code-stream round trip failed for {coding}, sample {sample_idx}")
                if decoded_reference is None:
                    decoded_reference = decoded
                elif not np.array_equal(decoded_reference, decoded):
                    raise RuntimeError("Coding methods decoded different RVQ indices")

                byte_values[coding].append(len(stream))
                encode_times[coding].append(middle - start)
                decode_times[coding].append(end - middle)
                sample_rows.append(
                    {
                        "n_quantizers": n_quantizers,
                        "coding": coding,
                        "sample_index": sample_idx,
                        "compressed_bytes": len(stream),
                        "compressed_bps": len(stream) * 8.0 / duration_sec,
                    }
                )
            roundtrip_codes.append(decoded_reference)

        decoded_prefix = np.stack(roundtrip_codes, axis=0)
        l1_values, snr_values, neural_decode_ms = reconstruct_from_codes(
            decoded_prefix,
            waveforms,
            model,
            device=device,
            batch_size=batch_size,
        )
        # Keep the established SeisDAC rate convention for direct comparison
        # with earlier tables. Actual streams below include finite-window
        # padding (3001 -> 3008 samples here) and their per-window header.
        fixed_bps = (
            n_quantizers
            * bits_per_code
            * sample_rate
            / int(generator.hop_length)
        )
        fixed_bytes_per_window = fixed_bps * duration_sec / 8.0
        common = {
            "n_quantizers": n_quantizers,
            "samples": len(waveforms),
            "frames_per_window": codes.shape[2],
            "bits_per_code": bits_per_code,
            "header_bytes": STREAM_HEADER.size,
            "l1_mean": float(np.mean(l1_values)),
            "l1_std": float(np.std(l1_values)),
            "snr_mean_db": finite_mean(snr_values),
            "snr_std_db": finite_std(snr_values),
            "marginal_entropy_estimate_bps": cumulative_stat(
                codebook_stats, n_quantizers, "marginal_entropy_estimate_bps"
            ),
            "first_order_entropy_estimate_bps": cumulative_stat(
                codebook_stats, n_quantizers, "first_order_entropy_estimate_bps"
            ),
            "neural_decode_ms_per_window": neural_decode_ms,
        }
        theoretical_row = {
            **common,
            "coding": "fixed-theoretical",
            "actual_stream": False,
            "compressed_bytes_mean": fixed_bytes_per_window,
            "compressed_bytes_std": 0.0,
            "compressed_bps": fixed_bps,
            "compression_ratio_vs_float32": original_bps / fixed_bps,
            "savings_vs_fixed_percent": 0.0,
            "entropy_encode_ms_per_window": float("nan"),
            "entropy_decode_ms_per_window": float("nan"),
        }
        results.append(theoretical_row)

        for coding in CODING_IDS:
            mean_bytes = float(np.mean(byte_values[coding]))
            compressed_bps = mean_bytes * 8.0 / duration_sec
            results.append(
                {
                    **common,
                    "coding": coding,
                    "actual_stream": True,
                    "compressed_bytes_mean": mean_bytes,
                    "compressed_bytes_std": float(np.std(byte_values[coding])),
                    "compressed_bps": compressed_bps,
                    "compression_ratio_vs_float32": original_bps / compressed_bps,
                    "savings_vs_fixed_percent": 100.0 * (1.0 - compressed_bps / fixed_bps),
                    "entropy_encode_ms_per_window": float(np.mean(encode_times[coding]) * 1000.0),
                    "entropy_decode_ms_per_window": float(np.mean(decode_times[coding]) * 1000.0),
                }
            )

        full_rows = [row for row in results if row["n_quantizers"] == n_quantizers]
        actual_rows = [row for row in full_rows if row["coding"].startswith("zstd")]
        best = min(actual_rows, key=lambda row: row["compressed_bps"])
        print(
            "    "
            f"fixed={fixed_bps:.1f} bps, best={best['coding']} "
            f"{best['compressed_bps']:.1f} bps "
            f"({best['savings_vs_fixed_percent']:+.1f}%), "
            f"L1={common['l1_mean']:.4f}, SNR={format_float(common['snr_mean_db'], 2)} dB"
        )

    generator.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return results, sample_rows


@torch.no_grad()
def evaluate_factorized_rans(
    codes: np.ndarray,
    existing_rows: Sequence[Dict[str, float]],
    model,
    *,
    original_length: int,
    sample_rate: int,
    n_channels: int,
    batch_size: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    entropy_model = model.generator.entropy_model
    if entropy_model is None:
        print("Skipping factorized rANS: checkpoint has no entropy model.")
        return [], []

    entropy_model.eval().to("cpu")
    cdfs = entropy_model.quantized_cdf()
    codec = FactorizedRansCodec(cdfs, precision=entropy_model.cdf_precision)
    duration_sec = original_length / float(sample_rate)
    original_bps = n_channels * 32.0 * sample_rate
    fixed_rows = {
        int(row["n_quantizers"]): row
        for row in existing_rows
        if row["coding"] == "fixed-theoretical"
    }

    results: List[Dict[str, float]] = []
    sample_rows: List[Dict[str, float]] = []
    for n_quantizers in sorted(fixed_rows):
        prefix = np.ascontiguousarray(codes[:, :n_quantizers, :])
        estimated_bits: List[float] = []
        for start_idx in range(0, len(prefix), batch_size):
            code_batch = torch.from_numpy(prefix[start_idx : start_idx + batch_size]).long()
            estimated_bits.extend(entropy_model.estimate_bits(code_batch).cpu().tolist())

        byte_values: List[int] = []
        encode_times: List[float] = []
        decode_times: List[float] = []
        for sample_idx, sample_codes in enumerate(prefix):
            start = time.perf_counter()
            stream = codec.encode(sample_codes, original_length=original_length)
            middle = time.perf_counter()
            decoded, decoded_length = codec.decode(stream)
            end = time.perf_counter()
            if decoded_length != original_length or not np.array_equal(decoded, sample_codes):
                raise RuntimeError(
                    f"Factorized-rANS round trip failed for nq={n_quantizers}, sample={sample_idx}"
                )
            byte_values.append(len(stream))
            encode_times.append(middle - start)
            decode_times.append(end - middle)
            sample_rows.append(
                {
                    "n_quantizers": n_quantizers,
                    "coding": "rans-factorized",
                    "sample_index": sample_idx,
                    "compressed_bytes": len(stream),
                    "compressed_bps": len(stream) * 8.0 / duration_sec,
                }
            )

        base = fixed_rows[n_quantizers]
        mean_bytes = float(np.mean(byte_values))
        measured_bps = mean_bytes * 8.0 / duration_sec
        estimated_bps = float(np.mean(estimated_bits) / duration_sec)
        row = {
            **base,
            "coding": "rans-factorized",
            "actual_stream": True,
            "header_bytes": RANS_STREAM_HEADER.size,
            "compressed_bytes_mean": mean_bytes,
            "compressed_bytes_std": float(np.std(byte_values)),
            "compressed_bps": measured_bps,
            "compression_ratio_vs_float32": original_bps / measured_bps,
            "savings_vs_fixed_percent": 100.0 * (
                1.0 - measured_bps / float(base["compressed_bps"])
            ),
            "factorized_estimated_bps": estimated_bps,
            "entropy_encode_ms_per_window": float(np.mean(encode_times) * 1000.0),
            "entropy_decode_ms_per_window": float(np.mean(decode_times) * 1000.0),
        }
        results.append(row)
        print(
            f"[rANS nq={n_quantizers}] estimate={estimated_bps:.1f} bps, "
            f"actual={measured_bps:.1f} bps, "
            f"gap={measured_bps - estimated_bps:+.1f} bps"
        )
    return results, sample_rows


def write_csv(rows: Sequence[Dict[str, float]], path: Path) -> None:
    fieldnames = [
        "n_quantizers",
        "coding",
        "actual_stream",
        "samples",
        "frames_per_window",
        "bits_per_code",
        "header_bytes",
        "compressed_bytes_mean",
        "compressed_bytes_std",
        "compressed_bps",
        "compression_ratio_vs_float32",
        "savings_vs_fixed_percent",
        "l1_mean",
        "l1_std",
        "snr_mean_db",
        "snr_std_db",
        "marginal_entropy_estimate_bps",
        "first_order_entropy_estimate_bps",
        "factorized_estimated_bps",
        "entropy_encode_ms_per_window",
        "entropy_decode_ms_per_window",
        "neural_decode_ms_per_window",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        order = {coding: idx for idx, coding in enumerate(RESULT_ORDER)}
        for row in sorted(rows, key=lambda item: (item["n_quantizers"], order[item["coding"]])):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_sample_csv(rows: Sequence[Dict[str, float]], path: Path) -> None:
    fieldnames = ["n_quantizers", "coding", "sample_index", "compressed_bytes", "compressed_bps"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_codebook_csv(rows: Sequence[Dict[str, float]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def load_baseline_rows(path: Path, max_bps: float) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["family"] == "SeisDAC":
                continue
            parsed = {
                "family": row["family"],
                "compressed_bps": float(row["compressed_bps"]),
                "l1_mean": float(row["l1_mean"]),
                "snr_mean_db": float(row["snr_mean_db"]),
            }
            if parsed["compressed_bps"] <= max_bps and math.isfinite(parsed["snr_mean_db"]):
                rows.append(parsed)
    return rows


def nice_upper(value: float, step: float) -> float:
    return max(step, math.ceil(value / step) * step)


def marker(fig: Figure, x: float, y: float, color: str, size: float) -> None:
    fig.rect(x - size / 2, y - size / 2, size, size, fill=color, stroke="#FFFFFF", stroke_width=0.6)


def draw_figure(
    rows: Sequence[Dict[str, float]],
    baseline_rows: Sequence[Dict[str, float]],
    output_dir: Path,
    *,
    plot_max_bps: float,
) -> None:
    visible_entropy = [row for row in rows if row["compressed_bps"] <= plot_max_bps]
    all_rows = list(visible_entropy) + list(baseline_rows)
    x_max = nice_upper(max(float(row["compressed_bps"]) for row in all_rows) * 1.04, 500.0)
    x_max = min(x_max, plot_max_bps)
    l1_max = nice_upper(max(float(row["l1_mean"]) for row in all_rows) * 1.10, 0.02)
    snr_values = [float(row["snr_mean_db"]) for row in all_rows if math.isfinite(float(row["snr_mean_db"]))]
    snr_min = min(0.0, math.floor(min(snr_values) / 5.0) * 5.0)
    snr_max = nice_upper(max(snr_values) + 3.0, 5.0)

    x_step = 500.0 if x_max <= 4000 else 1000.0
    x_ticks = list(np.arange(0.0, x_max + 0.1, x_step))
    l1_ticks = list(np.linspace(0.0, l1_max, 6))
    snr_ticks = list(np.arange(snr_min, snr_max + 0.1, 5.0))

    fig = Figure(980, 445)
    draw_panel_title(fig, 42, 32, "A", "Entropy-coded SeisDAC: waveform error")
    draw_panel_title(fig, 528, 32, "B", "Entropy-coded SeisDAC: SNR")
    sx_l1, sy_l1 = draw_axes(
        fig,
        76,
        68,
        385,
        230,
        xlim=(0, x_max),
        ylim=(0, l1_max),
        xticks=x_ticks,
        yticks=l1_ticks,
        xlabel="Measured bitrate (bps)",
        ylabel="Mean L1 error",
    )
    sx_snr, sy_snr = draw_axes(
        fig,
        562,
        68,
        385,
        230,
        xlim=(0, x_max),
        ylim=(snr_min, snr_max),
        xticks=x_ticks,
        yticks=snr_ticks,
        xlabel="Measured bitrate (bps)",
        ylabel="Mean SNR (dB)",
    )

    baseline_families = sorted(set(str(row["family"]) for row in baseline_rows))
    for family in baseline_families:
        family_rows = sorted(
            [row for row in baseline_rows if row["family"] == family],
            key=lambda row: row["compressed_bps"],
        )
        color = FAMILY_COLORS.get(family, COLOR_LIGHT)
        l1_points = [(sx_l1(row["compressed_bps"]), sy_l1(row["l1_mean"])) for row in family_rows]
        snr_points = [(sx_snr(row["compressed_bps"]), sy_snr(row["snr_mean_db"])) for row in family_rows]
        if len(family_rows) > 1:
            fig.polyline(l1_points, color=color, width=0.8, dash="4,4")
            fig.polyline(snr_points, color=color, width=0.8, dash="4,4")
        for l1_point, snr_point in zip(l1_points, snr_points):
            marker(fig, *l1_point, color, 3.5)
            marker(fig, *snr_point, color, 3.5)

    plotted_codings = tuple(
        coding
        for coding in (
            "fixed-theoretical",
            "zstd-packed10",
            "zstd-uint16",
            "rans-factorized",
        )
        if any(row["coding"] == coding for row in visible_entropy)
    )
    for coding in plotted_codings:
        coding_rows = sorted(
            [row for row in visible_entropy if row["coding"] == coding],
            key=lambda row: row["compressed_bps"],
        )
        color = RESULT_COLORS[coding]
        dash = "4,4" if coding == "fixed-theoretical" else ""
        l1_points = [(sx_l1(row["compressed_bps"]), sy_l1(row["l1_mean"])) for row in coding_rows]
        snr_points = [(sx_snr(row["compressed_bps"]), sy_snr(row["snr_mean_db"])) for row in coding_rows]
        fig.polyline(l1_points, color=color, width=2.0, dash=dash)
        fig.polyline(snr_points, color=color, width=2.0, dash=dash)
        for l1_point, snr_point in zip(l1_points, snr_points):
            marker(fig, *l1_point, color, 6.0)
            marker(fig, *snr_point, color, 6.0)

    legend_entries = [(RESULT_LABELS[coding], RESULT_COLORS[coding]) for coding in plotted_codings]
    legend_entries.extend((family, FAMILY_COLORS.get(family, COLOR_LIGHT)) for family in baseline_families)
    col_width = 190
    for idx, (label, color) in enumerate(legend_entries):
        col = idx % 4
        row_idx = idx // 4
        x = 90 + col * col_width
        y = 350 + row_idx * 18
        fig.rect(x, y - 9, 11, 8, fill=color)
        fig.text(x + 16, y - 1, label, size=8.0, color=COLOR_GRAY)
    fig.text(
        490,
        424,
        "Measured streams include per-window framing; classical codec baselines are dashed.",
        size=8.5,
        color=COLOR_GRAY,
        anchor="middle",
    )
    fig.render_svg(output_dir / "fig_seisdac_entropy_rate_distortion.svg")
    fig.render_pdf(output_dir / "fig_seisdac_entropy_rate_distortion.pdf")


def write_summary(
    rows: Sequence[Dict[str, float]],
    stats: Sequence[Dict[str, float]],
    info: Dict[str, float],
    output_dir: Path,
) -> None:
    full_nq = max(int(row["n_quantizers"]) for row in rows)
    full_rows = [row for row in rows if row["n_quantizers"] == full_nq]
    order = {coding: idx for idx, coding in enumerate(RESULT_ORDER)}
    full_rows.sort(key=lambda row: order[row["coding"]])
    best = min(
        (row for row in full_rows if bool(row["actual_stream"])),
        key=lambda row: row["compressed_bps"],
    )
    rans_full_row = next(
        (row for row in full_rows if row["coding"] == "rans-factorized"),
        None,
    )
    lines = [
        "=== SeisDAC entropy-coded bitrate ===",
        f"Dataset: {info['dataset']} ({info['split']} split)",
        f"Samples: {info['num_samples']}",
        f"Window: {info['window_length']} samples at {info['sample_rate']} Hz",
        f"RVQ frames per window: {full_rows[0]['frames_per_window']}",
        f"Code width: {full_rows[0]['bits_per_code']} bits",
        f"Packed/zstd header: {STREAM_HEADER.size} bytes per window",
        *(
            [f"Factorized-rANS header: {RANS_STREAM_HEADER.size} bytes per window"]
            if rans_full_row is not None
            else []
        ),
        "",
        f"Full-rate results (n_quantizers={full_nq}):",
    ]
    for row in full_rows:
        lines.append(
            f"  {RESULT_LABELS[row['coding']]:29s} "
            f"bps={row['compressed_bps']:7.1f} "
            f"ratio={row['compression_ratio_vs_float32']:5.2f}x "
            f"saving={row['savings_vs_fixed_percent']:+6.2f}%"
        )
    lines.extend(
        [
            "",
            f"Best measured method: {RESULT_LABELS[best['coding']]} at {best['compressed_bps']:.1f} bps",
            f"Waveform quality: L1={best['l1_mean']:.4f}, SNR={format_float(best['snr_mean_db'], 2)} dB",
            f"Marginal entropy estimate: {best['marginal_entropy_estimate_bps']:.1f} bps",
            f"First-order entropy estimate: {best['first_order_entropy_estimate_bps']:.1f} bps",
            *(
                [
                    "Learned factorized estimate: "
                    f"{rans_full_row['factorized_estimated_bps']:.1f} bps"
                ]
                if rans_full_row is not None
                else []
            ),
            "",
            "Notes:",
            "  Every measured stream was decoded and checked against the original RVQ indices.",
            "  Lossless entropy coding preserves reconstruction and picking outputs at the code level.",
            "  Byte rates include framing and one independently coded entropy payload per window.",
            "  Fixed-width theoretical rate is asymptotic; actual streams also include latent-frame padding.",
            "  Entropy estimates are fitted on this evaluation set; the first-order value is optimistic.",
            "  Longer continuous streams should amortize headers and may compress better than these windows.",
        ]
    )
    (output_dir / "seisdac_entropy_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    table_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Full-rate SeisDAC bitrate before and after lossless entropy coding on ETHZ development windows. Measured rates include independently framed per-window streams. Entropy coding preserves the RVQ indices and therefore does not alter reconstruction quality.}",
        r"\label{tab:seisdac_entropy}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Rate accounting & Bitrate (bps) & Ratio & Saving & L1 & SNR (dB) \\",
        r"\midrule",
    ]
    for row in full_rows:
        table_lines.append(
            f"{RESULT_LABELS[row['coding']]} & {row['compressed_bps']:.0f} & "
            f"{row['compression_ratio_vs_float32']:.1f}$\\times$ & "
            f"{row['savings_vs_fixed_percent']:+.1f}\\% & "
            f"{row['l1_mean']:.4f} & {format_float(row['snr_mean_db'], 2)} \\\\"
        )
    table_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\linewidth]{fig_seisdac_entropy_rate_distortion.pdf}",
            r"\caption{Rate--distortion curves after lossless entropy coding of SeisDAC RVQ indices. Classical codec curves are included as references.}",
            r"\label{fig:seisdac_entropy_rd}",
            r"\end{figure}",
        ]
    )
    (output_dir / "seisdac_entropy_tables_and_figure.tex").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure actual SeisDAC bitrate with reversible zstd and factorized-rANS coding."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_name", default="ETHZ")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline_csv", default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--seisdac_quantizers", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--zstd_level", type=int, default=9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--plot_max_bps", type=float, default=4000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    waveforms, info, model = load_waveforms(args)
    info["dataset"] = args.data_name
    info["split"] = args.split
    codes = extract_codes(
        waveforms,
        model,
        device=args.device,
        batch_size=args.batch_size,
    )
    duration_sec = waveforms[0].shape[-1] / float(info["sample_rate"])
    stats = compute_codebook_stats(
        codes,
        codebook_size=int(model.generator.codebook_size),
        duration_sec=duration_sec,
    )
    rows, sample_rows = evaluate_entropy_coding(
        waveforms,
        codes,
        stats,
        model,
        parse_int_list(args.seisdac_quantizers),
        device=args.device,
        batch_size=args.batch_size,
        zstd_level=args.zstd_level,
    )
    rans_rows, rans_sample_rows = evaluate_factorized_rans(
        codes,
        rows,
        model,
        original_length=int(waveforms[0].shape[-1]),
        sample_rate=int(info["sample_rate"]),
        n_channels=int(info["n_channels"]),
        batch_size=args.batch_size,
    )
    rows.extend(rans_rows)
    sample_rows.extend(rans_sample_rows)

    results_csv = output_dir / "seisdac_entropy_results.csv"
    samples_csv = output_dir / "seisdac_entropy_sample_rows.csv"
    stats_csv = output_dir / "seisdac_entropy_codebook_stats.csv"
    write_csv(rows, results_csv)
    write_sample_csv(sample_rows, samples_csv)
    write_codebook_csv(stats, stats_csv)
    metadata = {
        **info,
        "checkpoint": args.checkpoint,
        "zstd_level": args.zstd_level,
        "packed_zstd_header_bytes": STREAM_HEADER.size,
        "factorized_rans_header_bytes": RANS_STREAM_HEADER.size,
        "results": rows,
        "codebook_stats": stats,
    }
    (output_dir / "seisdac_entropy_results.json").write_text(
        json.dumps(json_safe(metadata), indent=2) + "\n", encoding="utf-8"
    )
    baseline_rows = load_baseline_rows(Path(args.baseline_csv), args.plot_max_bps)
    draw_figure(
        rows,
        baseline_rows,
        output_dir,
        plot_max_bps=args.plot_max_bps,
    )
    write_summary(rows, stats, info, output_dir)

    print(f"Saved results: {results_csv}")
    print(f"Saved per-window rates: {samples_csv}")
    print(f"Saved codebook statistics: {stats_csv}")
    print(f"Saved summary: {output_dir / 'seisdac_entropy_summary.txt'}")
    print(f"Saved figure: {output_dir / 'fig_seisdac_entropy_rate_distortion.pdf'}")


if __name__ == "__main__":
    main()
