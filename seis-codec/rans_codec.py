"""Byte-aligned rANS coding for categorical RVQ entropy models."""

import hashlib
import struct
from typing import Tuple

import numpy as np


RANS_BYTE_L = 1 << 23
STREAM_MAGIC = b"SDR1"
FIRST_ORDER_STREAM_MAGIC = b"SDR2"
STREAM_VERSION = 1
STREAM_HEADER = struct.Struct("<4sBBBBIII8s")


def _validate_cdfs(cdfs: np.ndarray, precision: int) -> np.ndarray:
    cdfs = np.asarray(cdfs)
    if cdfs.ndim != 2 or cdfs.shape[1] < 3:
        raise ValueError(f"Expected CDFs [n_models, alphabet_size + 1], got {cdfs.shape}")
    if not 1 <= int(precision) <= 16:
        raise ValueError("rANS precision must be between 1 and 16 bits")
    total = 1 << int(precision)
    cdfs64 = cdfs.astype(np.int64, copy=False)
    if np.any(cdfs64[:, 0] != 0) or np.any(cdfs64[:, -1] != total):
        raise ValueError("Each CDF must start at zero and end at 2**precision")
    if np.any(np.diff(cdfs64, axis=1) <= 0):
        raise ValueError("rANS requires strictly positive symbol frequencies")
    return np.ascontiguousarray(cdfs64)


def rans_encode(
    symbols: np.ndarray,
    model_indexes: np.ndarray,
    cdfs: np.ndarray,
    *,
    precision: int,
) -> bytes:
    """Encode symbols whose categorical model is selected by ``model_indexes``."""
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    model_indexes = np.asarray(model_indexes, dtype=np.int64).reshape(-1)
    if symbols.shape != model_indexes.shape:
        raise ValueError("symbols and model_indexes must have the same shape")
    cdfs = _validate_cdfs(cdfs, precision)
    alphabet_size = cdfs.shape[1] - 1
    if symbols.size:
        if symbols.min() < 0 or symbols.max() >= alphabet_size:
            raise ValueError("Symbol outside rANS alphabet")
        if model_indexes.min() < 0 or model_indexes.max() >= cdfs.shape[0]:
            raise ValueError("Invalid rANS model index")

    return _rans_encode_prevalidated(symbols, model_indexes, cdfs, precision=precision)


def _rans_encode_prevalidated(
    symbols: np.ndarray,
    model_indexes: np.ndarray,
    cdfs: np.ndarray,
    *,
    precision: int,
) -> bytes:
    """Encode arrays after shape, range, and CDF validation by the caller."""
    state = RANS_BYTE_L
    emitted = bytearray()
    for symbol, model_idx in zip(symbols[::-1], model_indexes[::-1]):
        start = int(cdfs[model_idx, symbol])
        frequency = int(cdfs[model_idx, symbol + 1] - start)
        state_max = ((RANS_BYTE_L >> precision) << 8) * frequency
        while state >= state_max:
            emitted.append(state & 0xFF)
            state >>= 8
        state = ((state // frequency) << precision) + (state % frequency) + start

    if state >= (1 << 32):
        raise RuntimeError("rANS state exceeded 32 bits")
    return struct.pack("<I", state) + bytes(reversed(emitted))


def rans_decode(
    payload: bytes,
    model_indexes: np.ndarray,
    cdfs: np.ndarray,
    *,
    precision: int,
) -> np.ndarray:
    """Decode a payload produced by :func:`rans_encode`."""
    if len(payload) < 4:
        raise ValueError("Truncated rANS payload")
    model_indexes = np.asarray(model_indexes, dtype=np.int64).reshape(-1)
    cdfs = _validate_cdfs(cdfs, precision)
    if model_indexes.size and (
        model_indexes.min() < 0 or model_indexes.max() >= cdfs.shape[0]
    ):
        raise ValueError("Invalid rANS model index")

    state = struct.unpack_from("<I", payload)[0]
    position = 4
    mask = (1 << precision) - 1
    symbols = np.empty(model_indexes.size, dtype=np.int64)
    for output_idx, model_idx in enumerate(model_indexes):
        slot = state & mask
        cdf = cdfs[model_idx]
        symbol = int(np.searchsorted(cdf, slot, side="right") - 1)
        if symbol < 0 or symbol >= cdf.size - 1:
            raise ValueError("Invalid rANS state for supplied CDF")
        start = int(cdf[symbol])
        frequency = int(cdf[symbol + 1] - start)
        symbols[output_idx] = symbol
        state = frequency * (state >> precision) + slot - start
        while state < RANS_BYTE_L:
            if position >= len(payload):
                raise ValueError("Truncated rANS renormalization bytes")
            state = (state << 8) | payload[position]
            position += 1

    if position != len(payload) or state != RANS_BYTE_L:
        raise ValueError("rANS payload has trailing bytes or an inconsistent final state")
    return symbols


class FactorizedRansCodec:
    """Self-framed rANS stream using shared factorized CDF tables."""

    def __init__(self, cdfs: np.ndarray, *, precision: int = 16):
        self.precision = int(precision)
        self.cdfs = _validate_cdfs(cdfs, self.precision)
        self.n_codebooks = int(self.cdfs.shape[0])
        self.codebook_size = int(self.cdfs.shape[1] - 1)

    def _cdf_digest(self, n_quantizers: int) -> bytes:
        payload = np.ascontiguousarray(self.cdfs[:n_quantizers].astype("<u4")).tobytes()
        return hashlib.sha256(payload).digest()[:8]

    def encode(self, codes: np.ndarray, *, original_length: int) -> bytes:
        codes = np.ascontiguousarray(codes, dtype=np.int64)
        if codes.ndim != 2:
            raise ValueError(f"Expected codes [Q, T], got {codes.shape}")
        n_quantizers, n_frames = codes.shape
        if not 1 <= n_quantizers <= min(self.n_codebooks, 255):
            raise ValueError("Unsupported number of quantizers")
        # Time-major serialization makes the stream order explicit and is also
        # the order a future causal model can extend without changing framing.
        symbols = codes.T.reshape(-1)
        model_indexes = np.tile(np.arange(n_quantizers, dtype=np.int64), n_frames)
        payload = rans_encode(
            symbols,
            model_indexes,
            self.cdfs[:n_quantizers],
            precision=self.precision,
        )
        header = STREAM_HEADER.pack(
            STREAM_MAGIC,
            STREAM_VERSION,
            self.precision,
            n_quantizers,
            0,
            n_frames,
            int(original_length),
            len(payload),
            self._cdf_digest(n_quantizers),
        )
        return header + payload

    def decode(self, stream: bytes) -> Tuple[np.ndarray, int]:
        if len(stream) < STREAM_HEADER.size:
            raise ValueError("Truncated factorized-rANS stream header")
        (
            magic,
            version,
            precision,
            n_quantizers,
            _reserved,
            n_frames,
            original_length,
            payload_nbytes,
            cdf_digest,
        ) = STREAM_HEADER.unpack_from(stream)
        if magic != STREAM_MAGIC or version != STREAM_VERSION:
            raise ValueError("Unsupported factorized-rANS stream format")
        if precision != self.precision:
            raise ValueError("rANS precision does not match the entropy model")
        if not 1 <= n_quantizers <= self.n_codebooks:
            raise ValueError("Stream requests unavailable entropy models")
        if cdf_digest != self._cdf_digest(n_quantizers):
            raise ValueError("Stream was encoded with a different entropy-model CDF")
        payload = stream[STREAM_HEADER.size :]
        if len(payload) != payload_nbytes:
            raise ValueError("Truncated or trailing factorized-rANS payload")

        model_indexes = np.tile(np.arange(n_quantizers, dtype=np.int64), n_frames)
        symbols = rans_decode(
            payload,
            model_indexes,
            self.cdfs[:n_quantizers],
            precision=self.precision,
        )
        return symbols.reshape(n_frames, n_quantizers).T, int(original_length)


class FirstOrderRansCodec:
    """Self-framed causal rANS using the previous same-codebook symbol."""

    def __init__(
        self,
        marginal_cdfs: np.ndarray,
        conditional_cdfs: np.ndarray,
        *,
        precision: int = 16,
    ):
        self.precision = int(precision)
        marginal_cdfs = np.asarray(marginal_cdfs)
        conditional_cdfs = np.asarray(conditional_cdfs)
        if marginal_cdfs.ndim != 2:
            raise ValueError("Marginal CDFs must have shape [Q, K + 1]")
        n_codebooks = int(marginal_cdfs.shape[0])
        codebook_size = int(marginal_cdfs.shape[1] - 1)
        expected_conditional_shape = (
            n_codebooks,
            codebook_size,
            codebook_size + 1,
        )
        if conditional_cdfs.shape != expected_conditional_shape:
            raise ValueError(
                "Conditional CDFs must have shape "
                f"{expected_conditional_shape}, got {conditional_cdfs.shape}"
            )
        validated_marginal = _validate_cdfs(marginal_cdfs, self.precision)
        validated_conditional = _validate_cdfs(
            conditional_cdfs.reshape(-1, codebook_size + 1),
            self.precision,
        )
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.cdfs = np.concatenate(
            [validated_marginal, validated_conditional],
            axis=0,
        )
        self._digests = {}

    def _conditional_model_index(self, codebook: int, previous_symbol: int) -> int:
        return self.n_codebooks + codebook * self.codebook_size + previous_symbol

    def _cdf_digest(self, n_quantizers: int) -> bytes:
        if n_quantizers not in self._digests:
            digest = hashlib.sha256()
            marginal = self.cdfs[:n_quantizers]
            conditional = self.cdfs[
                self.n_codebooks : self.n_codebooks + n_quantizers * self.codebook_size
            ]
            digest.update(np.ascontiguousarray(marginal.astype("<u4")).tobytes())
            digest.update(np.ascontiguousarray(conditional.astype("<u4")).tobytes())
            self._digests[n_quantizers] = digest.digest()[:8]
        return self._digests[n_quantizers]

    def _model_indexes(self, codes: np.ndarray) -> np.ndarray:
        n_quantizers, n_frames = codes.shape
        indexes = np.empty((n_frames, n_quantizers), dtype=np.int64)
        indexes[0] = np.arange(n_quantizers, dtype=np.int64)
        if n_frames > 1:
            codebook_offsets = (
                self.n_codebooks
                + np.arange(n_quantizers, dtype=np.int64) * self.codebook_size
            )
            indexes[1:] = codes[:, :-1].T + codebook_offsets[None, :]
        return indexes.reshape(-1)

    def encode(self, codes: np.ndarray, *, original_length: int) -> bytes:
        codes = np.ascontiguousarray(codes, dtype=np.int64)
        if codes.ndim != 2:
            raise ValueError(f"Expected codes [Q, T], got {codes.shape}")
        n_quantizers, n_frames = codes.shape
        if not 1 <= n_quantizers <= min(self.n_codebooks, 255):
            raise ValueError("Unsupported number of quantizers")
        if n_frames < 1:
            raise ValueError("At least one latent frame is required")
        if codes.min() < 0 or codes.max() >= self.codebook_size:
            raise ValueError("Symbol outside rANS alphabet")

        symbols = codes.T.reshape(-1)
        payload = _rans_encode_prevalidated(
            symbols,
            self._model_indexes(codes),
            self.cdfs,
            precision=self.precision,
        )
        header = STREAM_HEADER.pack(
            FIRST_ORDER_STREAM_MAGIC,
            STREAM_VERSION,
            self.precision,
            n_quantizers,
            0,
            n_frames,
            int(original_length),
            len(payload),
            self._cdf_digest(n_quantizers),
        )
        return header + payload

    def decode(self, stream: bytes) -> Tuple[np.ndarray, int]:
        if len(stream) < STREAM_HEADER.size:
            raise ValueError("Truncated first-order-rANS stream header")
        (
            magic,
            version,
            precision,
            n_quantizers,
            _reserved,
            n_frames,
            original_length,
            payload_nbytes,
            cdf_digest,
        ) = STREAM_HEADER.unpack_from(stream)
        if magic != FIRST_ORDER_STREAM_MAGIC or version != STREAM_VERSION:
            raise ValueError("Unsupported first-order-rANS stream format")
        if precision != self.precision:
            raise ValueError("rANS precision does not match the entropy model")
        if not 1 <= n_quantizers <= self.n_codebooks or n_frames < 1:
            raise ValueError("Stream requests unavailable entropy models")
        if cdf_digest != self._cdf_digest(n_quantizers):
            raise ValueError("Stream was encoded with a different entropy-model CDF")
        payload = stream[STREAM_HEADER.size :]
        if len(payload) != payload_nbytes or len(payload) < 4:
            raise ValueError("Truncated or trailing first-order-rANS payload")

        state = struct.unpack_from("<I", payload)[0]
        position = 4
        mask = (1 << self.precision) - 1
        symbols = np.empty(n_frames * n_quantizers, dtype=np.int64)
        for output_idx in range(symbols.size):
            frame = output_idx // n_quantizers
            codebook = output_idx % n_quantizers
            if frame == 0:
                model_idx = codebook
            else:
                previous_symbol = int(symbols[output_idx - n_quantizers])
                model_idx = self._conditional_model_index(codebook, previous_symbol)
            cdf = self.cdfs[model_idx]
            slot = state & mask
            symbol = int(np.searchsorted(cdf, slot, side="right") - 1)
            if symbol < 0 or symbol >= self.codebook_size:
                raise ValueError("Invalid rANS state for supplied CDF")
            start = int(cdf[symbol])
            frequency = int(cdf[symbol + 1] - start)
            symbols[output_idx] = symbol
            state = frequency * (state >> self.precision) + slot - start
            while state < RANS_BYTE_L:
                if position >= len(payload):
                    raise ValueError("Truncated rANS renormalization bytes")
                state = (state << 8) | payload[position]
                position += 1

        if position != len(payload) or state != RANS_BYTE_L:
            raise ValueError("rANS payload has trailing bytes or an inconsistent final state")
        return symbols.reshape(n_frames, n_quantizers).T, int(original_length)
