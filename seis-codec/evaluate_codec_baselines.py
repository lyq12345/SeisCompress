"""Evaluate classical codec baselines and draw rate-distortion curves.

The evaluation uses the same SeisBench dev-window pipeline as ``evaluate.py``:
windowing, dtype conversion, and peak normalization are inherited from the
trained SeisDAC checkpoint. Metrics are therefore comparable to the existing
SeisDAC reconstruction results.
"""

import argparse
import csv
import io
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
import torch
from obspy import Stream, Trace, UTCDateTime, read as read_obspy

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - exercised only when optional deps are absent.
    zstd = None

try:
    import zfpy
except ImportError:  # pragma: no cover
    zfpy = None

try:
    from pysz import sz, szAlgorithm, szConfig, szErrorBoundMode
except ImportError:  # pragma: no cover
    sz = None
    szAlgorithm = None
    szConfig = None
    szErrorBoundMode = None

from evaluate import ensure_dataset_split
from make_paper_figures import (
    COLOR_AXIS,
    COLOR_GRAY,
    COLOR_LIGHT,
    Figure,
    draw_axes,
    draw_panel_title,
    parse_metrics_txt,
)
from train import SeisDACLightning


DEFAULT_CHECKPOINT = (
    "/data/seismic/seis-codec-logs/ethz_nogan_spectral/"
    "seislm_enc_peaknorm_latreg0p003_nogan_200ep_freeze/checkpoints/best-137-24288.ckpt"
)
DEFAULT_OURS_METRICS = "/data/seismic/seis-codec-eval/ethz_nogan_latreg0p003_best137/metrics.txt"
DEFAULT_OUTPUT_DIR = "/data/seismic/seis-codec-eval/codec_baselines_ethz"

CHANNEL_CODES = ("BHZ", "BHN", "BHE")
FAMILY_ORDER = (
    "SeisDAC",
    "ZFP",
    "SZ3",
    "zstd+quant",
    "zstd-float32",
    "MiniSEED Steim2",
)
FAMILY_COLORS = {
    "SeisDAC": "#111827",
    "ZFP": "#D97706",
    "SZ3": "#6A4C93",
    "zstd+quant": "#C44E52",
    "zstd-float32": "#6B7280",
    "MiniSEED Steim2": "#2C6B73",
}


@dataclass(frozen=True)
class CodecOutput:
    reconstructed: np.ndarray
    nbytes: int
    encode_sec: float
    decode_sec: float


@dataclass(frozen=True)
class CodecSpec:
    family: str
    codec: str
    setting: str
    sort_value: float
    run: Callable[[np.ndarray], CodecOutput]


def parse_float_list(text: str) -> List[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def compute_snr_np(reference: np.ndarray, estimate: np.ndarray) -> float:
    noise = reference - estimate
    signal_power = float(np.sum(reference.astype(np.float64) ** 2))
    noise_power = float(np.sum(noise.astype(np.float64) ** 2))
    if noise_power == 0:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise_power)


def finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return float("inf")
    return float(np.mean(finite))


def finite_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return float("nan")
    return float(np.std(finite))


def format_float(value: float, digits: int) -> str:
    if math.isinf(value):
        return r"$\infty$"
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def quantize_uniform(x: np.ndarray, bits: int) -> Tuple[np.ndarray, float]:
    if bits < 2:
        raise ValueError("Uniform quantization requires at least 2 bits.")
    scale = float((1 << (bits - 1)) - 1)
    q = np.rint(np.clip(x, -1.0, 1.0) * scale).astype(np.int32)
    q = np.clip(q, -int(scale), int(scale)).astype(np.int32, copy=False)
    return q, scale


def pack_quantized(q_signed: np.ndarray, bits: int) -> bytes:
    scale = (1 << (bits - 1)) - 1
    values = (q_signed.reshape(-1).astype(np.int32) + scale).astype(np.uint32)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    bit_matrix = ((values[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1), bitorder="big").tobytes()


def unpack_quantized(payload: bytes, bits: int, shape: Tuple[int, ...]) -> np.ndarray:
    n_values = int(np.prod(shape))
    raw = np.frombuffer(payload, dtype=np.uint8)
    bits_flat = np.unpackbits(raw, bitorder="big", count=n_values * bits)
    bit_matrix = bits_flat.reshape(n_values, bits).astype(np.uint32)
    weights = (1 << np.arange(bits - 1, -1, -1, dtype=np.uint32)).astype(np.uint32)
    values = bit_matrix @ weights
    scale = (1 << (bits - 1)) - 1
    q_signed = values.astype(np.int32) - scale
    return q_signed.reshape(shape)


def zstd_float32_codec(level: int) -> Callable[[np.ndarray], CodecOutput]:
    if zstd is None:
        raise RuntimeError("zstandard is not installed. Run: pip3 install zstandard")
    compressor = zstd.ZstdCompressor(level=level)
    decompressor = zstd.ZstdDecompressor()

    def run(x: np.ndarray) -> CodecOutput:
        x_le = np.ascontiguousarray(x.astype("<f4", copy=False))
        start = time.perf_counter()
        payload = compressor.compress(x_le.tobytes(order="C"))
        mid = time.perf_counter()
        decoded_payload = decompressor.decompress(payload)
        decoded = np.frombuffer(decoded_payload, dtype="<f4").reshape(x.shape).astype(np.float32, copy=True)
        end = time.perf_counter()
        return CodecOutput(decoded, len(payload), mid - start, end - mid)

    return run


def zstd_quantized_codec(bits: int, level: int) -> Callable[[np.ndarray], CodecOutput]:
    if zstd is None:
        raise RuntimeError("zstandard is not installed. Run: pip3 install zstandard")
    compressor = zstd.ZstdCompressor(level=level)
    decompressor = zstd.ZstdDecompressor()

    def run(x: np.ndarray) -> CodecOutput:
        q, scale = quantize_uniform(x, bits)
        packed = pack_quantized(q, bits)
        start = time.perf_counter()
        payload = compressor.compress(packed)
        mid = time.perf_counter()
        decoded_packed = decompressor.decompress(payload)
        q_decoded = unpack_quantized(decoded_packed, bits, x.shape)
        decoded = (q_decoded.astype(np.float32) / scale).astype(np.float32, copy=False)
        end = time.perf_counter()
        return CodecOutput(decoded, len(payload), mid - start, end - mid)

    return run


def miniseed_steim2_codec(bits: int, sample_rate: int, reclen: int) -> Callable[[np.ndarray], CodecOutput]:
    delta = 1.0 / float(sample_rate)
    starttime = UTCDateTime(0)
    channel_index = {channel: i for i, channel in enumerate(CHANNEL_CODES)}

    def run(x: np.ndarray) -> CodecOutput:
        q, scale = quantize_uniform(x, bits)
        stream = Stream()
        for channel_idx in range(q.shape[0]):
            trace = Trace(data=np.ascontiguousarray(q[channel_idx].astype(np.int32, copy=False)))
            trace.stats.delta = delta
            trace.stats.starttime = starttime
            trace.stats.network = "XX"
            trace.stats.station = "SDAC"
            trace.stats.location = ""
            trace.stats.channel = CHANNEL_CODES[channel_idx] if channel_idx < len(CHANNEL_CODES) else f"BH{channel_idx}"
            stream.append(trace)

        buffer = io.BytesIO()
        start = time.perf_counter()
        stream.write(buffer, format="MSEED", encoding="STEIM2", reclen=reclen)
        payload = buffer.getvalue()
        mid = time.perf_counter()
        decoded_stream = read_obspy(io.BytesIO(payload), format="MSEED")
        q_decoded = np.empty_like(q)
        for trace in decoded_stream:
            idx = channel_index.get(trace.stats.channel)
            if idx is None:
                continue
            q_decoded[idx] = trace.data[: q.shape[1]].astype(np.int32, copy=False)
        decoded = (q_decoded.astype(np.float32) / scale).astype(np.float32, copy=False)
        end = time.perf_counter()
        return CodecOutput(decoded, len(payload), mid - start, end - mid)

    return run


def zfp_rate_codec(rate: float) -> Callable[[np.ndarray], CodecOutput]:
    if zfpy is None:
        raise RuntimeError("zfpy is not installed. Run: pip3 install zfpy")

    def run(x: np.ndarray) -> CodecOutput:
        x32 = np.ascontiguousarray(x.astype(np.float32, copy=False))
        start = time.perf_counter()
        payload = zfpy.compress_numpy(x32, rate=rate)
        mid = time.perf_counter()
        decoded = zfpy.decompress_numpy(payload).astype(np.float32, copy=False)
        end = time.perf_counter()
        return CodecOutput(decoded.reshape(x.shape), len(payload), mid - start, end - mid)

    return run


def sz3_abs_codec(abs_bound: float) -> Callable[[np.ndarray], CodecOutput]:
    if sz is None:
        raise RuntimeError("pysz is not installed. Run: pip3 install pysz")

    def run(x: np.ndarray) -> CodecOutput:
        x32 = np.ascontiguousarray(x.astype(np.float32, copy=False))
        cfg = szConfig(x32.shape)
        cfg.errorBoundMode = szErrorBoundMode.ABS
        cfg.absErrorBound = abs_bound
        cfg.cmprAlgo = szAlgorithm.INTERP_LORENZO
        start = time.perf_counter()
        payload, _ratio = sz.compress(x32, cfg)
        mid = time.perf_counter()
        decoded, _cfg = sz.decompress(payload, np.float32, x32.shape)
        decoded = decoded.astype(np.float32, copy=False).reshape(x32.shape)
        end = time.perf_counter()
        return CodecOutput(decoded, int(payload.nbytes), mid - start, end - mid)

    return run


def build_specs(args: argparse.Namespace, sample_rate: int) -> List[CodecSpec]:
    specs: List[CodecSpec] = []
    quant_bits = parse_int_list(args.quant_bits)

    if zstd is not None:
        specs.append(
            CodecSpec(
                family="zstd-float32",
                codec="zstd",
                setting=f"float32 level={args.zstd_level}",
                sort_value=1e9,
                run=zstd_float32_codec(args.zstd_level),
            )
        )
        for bits in quant_bits:
            specs.append(
                CodecSpec(
                    family="zstd+quant",
                    codec="zstd",
                    setting=f"q{bits} level={args.zstd_level}",
                    sort_value=float(bits),
                    run=zstd_quantized_codec(bits, args.zstd_level),
                )
            )
    else:
        print("Skipping zstd: missing Python package. Install with: pip3 install zstandard")

    for bits in quant_bits:
        specs.append(
            CodecSpec(
                family="MiniSEED Steim2",
                codec="MiniSEED Steim2",
                setting=f"q{bits} reclen={args.miniseed_reclen}",
                sort_value=float(bits),
                run=miniseed_steim2_codec(bits, sample_rate, args.miniseed_reclen),
            )
        )

    if zfpy is not None:
        for rate in parse_float_list(args.zfp_rates):
            specs.append(
                CodecSpec(
                    family="ZFP",
                    codec="ZFP",
                    setting=f"rate={rate:g} bits/value",
                    sort_value=rate,
                    run=zfp_rate_codec(rate),
                )
            )
    else:
        print("Skipping ZFP: missing Python package. Install with: pip3 install zfpy")

    if sz is not None:
        for abs_bound in parse_float_list(args.sz3_abs_bounds):
            specs.append(
                CodecSpec(
                    family="SZ3",
                    codec="SZ3",
                    setting=f"abs={abs_bound:g}",
                    sort_value=abs_bound,
                    run=sz3_abs_codec(abs_bound),
                )
            )
    else:
        print("Skipping SZ3: missing Python package. Install with: pip3 install pysz")

    return specs


def load_waveforms(args: argparse.Namespace) -> Tuple[List[np.ndarray], Dict[str, float], SeisDACLightning]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading checkpoint for eval preprocessing: {args.checkpoint}")
    model = SeisDACLightning.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model.eval()

    dataset_cls = getattr(sbd, args.data_name)
    dataset = dataset_cls(
        sampling_rate=model.config.model.sample_rate,
        component_order="ZNE",
        dimension_order="NCW",
    )
    ensure_dataset_split(dataset)
    split_dataset = dataset.get_split(args.split)
    generator = sbg.GenericGenerator(split_dataset)
    generator.add_augmentations(model.get_val_augmentations())

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(generator), size=min(args.num_samples, len(generator)), replace=False)
    waveforms: List[np.ndarray] = []
    for idx in indices:
        sample = generator[int(idx)]
        x = np.asarray(sample["X"], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        waveforms.append(np.ascontiguousarray(x))

    info = {
        "sample_rate": int(model.config.model.sample_rate),
        "window_length": int(waveforms[0].shape[-1]),
        "n_channels": int(waveforms[0].shape[0]),
        "num_samples": int(len(waveforms)),
    }
    return waveforms, info, model


def compute_seisdac_bitrate(
    *,
    n_quantizers: int,
    codebook_size: int,
    hop_length: int,
    sample_rate: int,
    n_channels: int,
) -> Tuple[float, float]:
    bits_per_frame = n_quantizers * math.log2(codebook_size)
    compressed_bps = bits_per_frame * (sample_rate / hop_length)
    original_bps = n_channels * 32.0 * sample_rate
    return compressed_bps, original_bps / compressed_bps


@torch.no_grad()
def evaluate_seisdac_sweep(
    waveforms: Sequence[np.ndarray],
    model: SeisDACLightning,
    quantizers: Sequence[int],
    *,
    device: str,
    batch_size: int,
) -> List[Dict[str, float]]:
    generator = model.generator
    max_quantizers = int(generator.n_codebooks)
    requested = sorted(set(int(q) for q in quantizers if 1 <= int(q) <= max_quantizers))
    if not requested:
        print("Skipping SeisDAC sweep: no valid n_quantizers requested.")
        return []

    sample_rate = int(model.config.model.sample_rate)
    n_channels = int(model.config.model.in_channels)
    model.eval()
    model.to(device)

    batches = [
        np.stack(waveforms[start : start + batch_size], axis=0).astype(np.float32, copy=False)
        for start in range(0, len(waveforms), batch_size)
    ]
    rows: List[Dict[str, float]] = []
    for idx, n_quantizers in enumerate(requested, start=1):
        print(f"[SeisDAC {idx:02d}/{len(requested):02d}] n_quantizers={n_quantizers}")
        l1_values: List[float] = []
        snr_values: List[float] = []
        forward_times: List[float] = []
        for batch_np in batches:
            batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = generator(batch, sample_rate=sample_rate, n_quantizers=n_quantizers)
            fake = out["audio"]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()
            reconstructed = fake.detach().cpu().numpy().astype(np.float32, copy=False)
            forward_times.extend([(end - start) / len(batch_np)] * len(batch_np))
            for reference, estimate in zip(batch_np, reconstructed):
                l1_values.append(float(np.mean(np.abs(reference - estimate))))
                snr_values.append(compute_snr_np(reference, estimate))

        compressed_bps, compression_ratio = compute_seisdac_bitrate(
            n_quantizers=n_quantizers,
            codebook_size=int(generator.codebook_size),
            hop_length=int(generator.hop_length),
            sample_rate=sample_rate,
            n_channels=n_channels,
        )
        row = {
            "family": "SeisDAC",
            "codec": "SeisDAC",
            "setting": f"nq={n_quantizers}",
            "sort_value": float(n_quantizers),
            "samples": len(waveforms),
            "compressed_bytes_mean": float("nan"),
            "compressed_bytes_std": float("nan"),
            "compressed_bps": compressed_bps,
            "compression_ratio_vs_float32": compression_ratio,
            "l1_mean": float(np.mean(l1_values)),
            "l1_std": float(np.std(l1_values)),
            "snr_mean_db": finite_mean(snr_values),
            "snr_std_db": finite_std(snr_values),
            "snr_inf_count": int(np.sum(np.isposinf(np.asarray(snr_values, dtype=np.float64)))),
            "encode_ms_per_window": float(np.mean(forward_times) * 1000.0),
            "decode_ms_per_window": float("nan"),
        }
        rows.append(row)
        print(
            "    "
            f"bps={row['compressed_bps']:.1f}, "
            f"ratio={row['compression_ratio_vs_float32']:.2f}x, "
            f"L1={row['l1_mean']:.4f}, "
            f"SNR={format_float(row['snr_mean_db'], 2)} dB"
        )
    model.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def evaluate_specs(
    waveforms: Sequence[np.ndarray],
    specs: Sequence[CodecSpec],
    *,
    sample_rate: int,
    original_bps: float,
    sample_rows_path: Optional[Path],
) -> List[Dict[str, float]]:
    duration_sec = waveforms[0].shape[-1] / float(sample_rate)
    results: List[Dict[str, float]] = []
    sample_rows_file = None
    sample_writer = None
    if sample_rows_path is not None:
        sample_rows_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rows_file = sample_rows_path.open("w", newline="", encoding="utf-8")
        sample_writer = csv.DictWriter(
            sample_rows_file,
            fieldnames=[
                "family",
                "codec",
                "setting",
                "sample_index",
                "compressed_bytes",
                "compressed_bps",
                "l1",
                "snr_db",
            ],
        )
        sample_writer.writeheader()

    try:
        for spec_idx, spec in enumerate(specs, start=1):
            print(f"[{spec_idx:02d}/{len(specs):02d}] {spec.family} | {spec.setting}")
            l1_values: List[float] = []
            snr_values: List[float] = []
            byte_values: List[int] = []
            encode_times: List[float] = []
            decode_times: List[float] = []

            for sample_idx, x in enumerate(waveforms):
                output = spec.run(x)
                reconstructed = output.reconstructed
                l1 = float(np.mean(np.abs(reconstructed.astype(np.float32) - x)))
                snr = compute_snr_np(x, reconstructed)
                compressed_bps = output.nbytes * 8.0 / duration_sec

                l1_values.append(l1)
                snr_values.append(snr)
                byte_values.append(output.nbytes)
                encode_times.append(output.encode_sec)
                decode_times.append(output.decode_sec)

                if sample_writer is not None:
                    sample_writer.writerow(
                        {
                            "family": spec.family,
                            "codec": spec.codec,
                            "setting": spec.setting,
                            "sample_index": sample_idx,
                            "compressed_bytes": output.nbytes,
                            "compressed_bps": f"{compressed_bps:.6f}",
                            "l1": f"{l1:.8f}",
                            "snr_db": "inf" if math.isinf(snr) else f"{snr:.6f}",
                        }
                    )

            compressed_bps_mean = float(np.mean(byte_values) * 8.0 / duration_sec)
            row = {
                "family": spec.family,
                "codec": spec.codec,
                "setting": spec.setting,
                "sort_value": spec.sort_value,
                "samples": len(waveforms),
                "compressed_bytes_mean": float(np.mean(byte_values)),
                "compressed_bytes_std": float(np.std(byte_values)),
                "compressed_bps": compressed_bps_mean,
                "compression_ratio_vs_float32": original_bps / compressed_bps_mean,
                "l1_mean": float(np.mean(l1_values)),
                "l1_std": float(np.std(l1_values)),
                "snr_mean_db": finite_mean(snr_values),
                "snr_std_db": finite_std(snr_values),
                "snr_inf_count": int(np.sum(np.isposinf(np.asarray(snr_values, dtype=np.float64)))),
                "encode_ms_per_window": float(np.mean(encode_times) * 1000.0),
                "decode_ms_per_window": float(np.mean(decode_times) * 1000.0),
            }
            results.append(row)
            print(
                "    "
                f"bps={row['compressed_bps']:.1f}, "
                f"ratio={row['compression_ratio_vs_float32']:.2f}x, "
                f"L1={row['l1_mean']:.4f}, "
                f"SNR={format_float(row['snr_mean_db'], 2)} dB"
            )
    finally:
        if sample_rows_file is not None:
            sample_rows_file.close()

    return results


def append_ours_result(
    rows: List[Dict[str, float]],
    ours_metrics_path: Path,
    *,
    sample_rate: int,
    n_channels: int,
) -> None:
    if not ours_metrics_path.exists():
        print(f"Skipping SeisDAC reference point: missing {ours_metrics_path}")
        return
    metrics = parse_metrics_txt(ours_metrics_path)
    rows.append(
        {
            "family": "SeisDAC",
            "codec": "SeisDAC",
            "setting": "no-GAN latreg=3e-3",
            "sort_value": 0.0,
            "samples": int(metrics.get("samples", 0)),
            "compressed_bytes_mean": float("nan"),
            "compressed_bytes_std": float("nan"),
            "compressed_bps": metrics["compressed_bps"],
            "compression_ratio_vs_float32": metrics.get(
                "compression_ratio_vs_float32",
                n_channels * 32.0 * sample_rate / metrics["compressed_bps"],
            ),
            "l1_mean": metrics["l1_mean"],
            "l1_std": metrics["l1_std"],
            "snr_mean_db": metrics["snr_mean_db"],
            "snr_std_db": metrics["snr_std_db"],
            "snr_inf_count": 0,
            "encode_ms_per_window": float("nan"),
            "decode_ms_per_window": float("nan"),
        }
    )


def family_sort_key(row: Dict[str, float]) -> Tuple[int, float]:
    try:
        family_idx = FAMILY_ORDER.index(str(row["family"]))
    except ValueError:
        family_idx = len(FAMILY_ORDER)
    return family_idx, float(row.get("sort_value", 0.0))


def write_csv(rows: Sequence[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "codec",
        "setting",
        "samples",
        "compressed_bps",
        "compression_ratio_vs_float32",
        "l1_mean",
        "l1_std",
        "snr_mean_db",
        "snr_std_db",
        "snr_inf_count",
        "compressed_bytes_mean",
        "compressed_bytes_std",
        "encode_ms_per_window",
        "decode_ms_per_window",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=family_sort_key):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sanitize_json_value(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    return value


def write_json(rows: Sequence[Dict[str, float]], info: Dict[str, float], path: Path) -> None:
    payload = {
        "dataset": info["dataset"],
        "split": info["split"],
        "sample_rate": info["sample_rate"],
        "window_length": info["window_length"],
        "n_channels": info["n_channels"],
        "samples": info["num_samples"],
        "results": sorted(rows, key=family_sort_key),
    }
    path.write_text(json.dumps(sanitize_json_value(payload), indent=2) + "\n", encoding="utf-8")


def choose_matched_rows(rows: Sequence[Dict[str, float]], target_bps: float) -> List[Dict[str, float]]:
    selected: List[Dict[str, float]] = []
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        if family == "zstd-float32":
            selected.extend(sorted(family_rows, key=lambda row: row["compressed_bps"]))
            continue
        selected.append(min(family_rows, key=lambda row: abs(row["compressed_bps"] - target_bps)))
    return selected


def write_latex_tables(rows: Sequence[Dict[str, float]], output_dir: Path, target_bps: float) -> None:
    matched = choose_matched_rows(rows, target_bps)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Codec baselines nearest to the full-rate SeisDAC point on the ETHZ development split. SeisDAC rate-distortion points are produced by varying the number of residual vector quantizers at inference time. MiniSEED Steim2 and zstd rate-distortion points use uniform quantization before entropy coding; ZFP uses fixed-rate mode; SZ3 uses absolute error-bounded mode.}",
        r"\label{tab:codec_baseline_matched}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Codec & Setting & Bitrate (bps) & Ratio & L1 error & SNR (dB) \\",
        r"\midrule",
    ]
    for row in matched:
        lines.append(
            f"{row['family']} & {row['setting']} & "
            f"{row['compressed_bps']:.0f} & "
            f"{row['compression_ratio_vs_float32']:.1f}$\\times$ & "
            f"{row['l1_mean']:.4f} & "
            f"{format_float(row['snr_mean_db'], 2)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\linewidth]{fig_codec_rate_distortion.pdf}",
            r"\caption{Rate-distortion curves for classical codec baselines and the proposed SeisDAC model on ETHZ development windows. SeisDAC is swept by using fewer residual vector quantizers at inference time.}",
            r"\label{fig:codec_rate_distortion}",
            r"\end{figure}",
        ]
    )
    (output_dir / "codec_baseline_tables_and_figure.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def nice_upper(value: float, step: float) -> float:
    return max(step, math.ceil(value / step) * step)


def marker(fig: Figure, x: float, y: float, color: str, *, size: float = 6.0, stroke: str = "#FFFFFF") -> None:
    fig.rect(x - size / 2, y - size / 2, size, size, fill=color, stroke=stroke, stroke_width=0.7)


def draw_compact_legend(fig: Figure, x: float, y: float, entries: Sequence[Tuple[str, str]]) -> None:
    col_w = 150
    row_h = 16
    for idx, (label, color) in enumerate(entries):
        col = idx % 3
        row = idx // 3
        px = x + col * col_w
        py = y + row * row_h
        fig.rect(px, py - 9, 11, 8, fill=color)
        fig.text(px + 16, py - 1, label, size=8.2, color=COLOR_GRAY)


def finite_snr_plot_values(rows: Sequence[Dict[str, float]]) -> List[float]:
    return [float(row["snr_mean_db"]) for row in rows if math.isfinite(float(row["snr_mean_db"]))]


def draw_rate_distortion_figure(rows: Sequence[Dict[str, float]], output_dir: Path, *, original_bps: float) -> None:
    x_max = nice_upper(max(original_bps, max(float(row["compressed_bps"]) for row in rows)) * 1.02, 1000.0)
    l1_max = nice_upper(max(float(row["l1_mean"]) for row in rows) * 1.10, 0.02)
    finite_snr = finite_snr_plot_values(rows)
    snr_min = min(0.0, math.floor(min(finite_snr) / 10.0) * 10.0) if finite_snr else 0.0
    snr_max = nice_upper(max(finite_snr) + 5.0, 10.0) if finite_snr else 60.0
    snr_max = min(max(snr_max, 40.0), 100.0)

    x_ticks = [tick for tick in [0, 1000, 2000, 4000, 6000, 8000, 9600, 12000] if tick <= x_max]
    l1_ticks = [tick for tick in np.linspace(0, l1_max, 6)]
    snr_step = 10.0 if snr_max - snr_min <= 60 else 20.0
    snr_ticks = list(np.arange(snr_min, snr_max + 0.1, snr_step))

    fig = Figure(980, 430)
    draw_panel_title(fig, 42, 32, "A", "Rate-distortion: waveform error")
    draw_panel_title(fig, 528, 32, "B", "Rate-distortion: SNR")

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
        xlabel="Bitrate (bps)",
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
        xlabel="Bitrate (bps)",
        ylabel="Mean SNR (dB)",
    )

    for family in FAMILY_ORDER:
        family_rows = sorted(
            [row for row in rows if row["family"] == family],
            key=lambda row: float(row["compressed_bps"]),
        )
        if not family_rows:
            continue
        color = FAMILY_COLORS[family]
        l1_points = [(sx_l1(float(row["compressed_bps"])), sy_l1(float(row["l1_mean"]))) for row in family_rows]
        snr_points = [
            (
                sx_snr(float(row["compressed_bps"])),
                sy_snr(snr_max if math.isinf(float(row["snr_mean_db"])) else float(row["snr_mean_db"])),
            )
            for row in family_rows
        ]
        if len(l1_points) > 1:
            fig.polyline(l1_points, color=color, width=1.5)
            fig.polyline(snr_points, color=color, width=1.5)
        for point_idx, (row, (px_l1, py_l1), (px_snr, py_snr)) in enumerate(
            zip(family_rows, l1_points, snr_points)
        ):
            marker(fig, px_l1, py_l1, color, size=7.0 if family == "SeisDAC" else 5.5)
            marker(fig, px_snr, py_snr, color, size=7.0 if family == "SeisDAC" else 5.5)
            if family == "SeisDAC" and point_idx == len(family_rows) - 1:
                fig.text(px_l1 + 8, py_l1 - 8, "SeisDAC", size=8.5, weight="bold")
                fig.text(px_snr + 8, py_snr - 8, "SeisDAC", size=8.5, weight="bold")
            elif family == "zstd-float32":
                fig.text(px_l1 - 8, py_l1 - 8, "lossless", size=8, color=COLOR_GRAY, anchor="end")
                fig.text(px_snr - 8, py_snr + 12, "lossless", size=8, color=COLOR_GRAY, anchor="end")

    for sx in [sx_l1, sx_snr]:
        fig.line(sx(original_bps), 68, sx(original_bps), 298, color=COLOR_LIGHT, width=1.0, dash="4,4")
    draw_compact_legend(
        fig,
        260,
        360,
        [(family, FAMILY_COLORS[family]) for family in FAMILY_ORDER if any(row["family"] == family for row in rows)],
    )
    fig.text(490, 415, "Dashed vertical line marks uncompressed float32 ZNE bitrate.", size=9, color=COLOR_GRAY, anchor="middle")

    fig.render_svg(output_dir / "fig_codec_rate_distortion.svg")
    fig.render_pdf(output_dir / "fig_codec_rate_distortion.pdf")


def write_summary_txt(rows: Sequence[Dict[str, float]], info: Dict[str, float], path: Path, target_bps: float) -> None:
    matched = choose_matched_rows(rows, target_bps)
    lines = [
        "=== Codec baseline rate-distortion ===",
        f"Dataset: {info['dataset']} ({info['split']} split)",
        f"Samples: {info['num_samples']}",
        f"Sample rate: {info['sample_rate']} Hz",
        f"Window length: {info['window_length']} samples",
        "",
        "Closest points to SeisDAC bitrate:",
    ]
    for row in matched:
        lines.append(
            f"  {row['family']:18s} {row['setting']:24s} "
            f"bps={row['compressed_bps']:7.1f} "
            f"ratio={row['compression_ratio_vs_float32']:5.2f}x "
            f"L1={row['l1_mean']:.4f} "
            f"SNR={format_float(row['snr_mean_db'], 2)} dB"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "  MiniSEED Steim2 supports integer traces; the curve uses uniform quantization before Steim2.",
            "  zstd-float32 is lossless; zstd+quant uses bit-packed uniform quantization before zstd.",
            "  ZFP uses fixed-rate mode; SZ3 uses absolute error-bounded mode.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MiniSEED Steim2, zstd, ZFP, and SZ3 codec baselines.",
        epilog="Optional dependencies: pip3 install zstandard zfpy pysz",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ours_metrics", default=DEFAULT_OURS_METRICS)
    parser.add_argument("--data_name", default="ETHZ")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quant_bits", default="4,6,8,10,12,14,16")
    parser.add_argument("--zstd_level", type=int, default=9)
    parser.add_argument("--zfp_rates", default="1,1.5,2,2.5,3,3.75,4.5,6,8,12")
    parser.add_argument("--sz3_abs_bounds", default="0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--miniseed_reclen", type=int, default=4096)
    parser.add_argument("--seisdac_quantizers", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--skip_seisdac_sweep", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_sample_rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    waveforms, info, model = load_waveforms(args)
    info["dataset"] = args.data_name
    info["split"] = args.split
    sample_rate = int(info["sample_rate"])
    n_channels = int(info["n_channels"])
    original_bps = n_channels * 32.0 * sample_rate
    specs = build_specs(args, sample_rate)
    if not specs:
        raise RuntimeError("No codec baselines available. Install optional dependencies first.")

    sample_rows_path = output_dir / "codec_baseline_sample_rows.csv" if args.save_sample_rows else None
    rows = evaluate_specs(
        waveforms,
        specs,
        sample_rate=sample_rate,
        original_bps=original_bps,
        sample_rows_path=sample_rows_path,
    )
    if args.skip_seisdac_sweep:
        append_ours_result(rows, Path(args.ours_metrics), sample_rate=sample_rate, n_channels=n_channels)
    else:
        rows.extend(
            evaluate_seisdac_sweep(
                waveforms,
                model,
                parse_int_list(args.seisdac_quantizers),
                device=args.device,
                batch_size=args.batch_size,
            )
        )

    csv_path = output_dir / "codec_baseline_results.csv"
    json_path = output_dir / "codec_baseline_results.json"
    txt_path = output_dir / "codec_baseline_summary.txt"
    write_csv(rows, csv_path)
    write_json(rows, info, json_path)
    seisdac_rows = [row for row in rows if row["family"] == "SeisDAC"]
    target_bps = max((row["compressed_bps"] for row in seisdac_rows), default=1125.0)
    write_latex_tables(rows, output_dir, target_bps)
    draw_rate_distortion_figure(rows, output_dir, original_bps=original_bps)
    write_summary_txt(rows, info, txt_path, target_bps)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved summary: {txt_path}")
    print(f"Saved figure: {output_dir / 'fig_codec_rate_distortion.pdf'}")
    if sample_rows_path is not None:
        print(f"Saved sample rows: {sample_rows_path}")


if __name__ == "__main__":
    main()
