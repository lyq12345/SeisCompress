"""Factorized categorical entropy model for residual-VQ code indices."""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def probabilities_to_quantized_cdf(
    probabilities: np.ndarray,
    *,
    precision: int = 16,
) -> np.ndarray:
    """Convert categorical PMFs to strictly positive integer CDFs.

    A frequency of at least one is reserved for every code so any valid RVQ
    index remains decodable, including codes absent from a calibration batch.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(f"Expected [n_codebooks, codebook_size], got {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("Probabilities must be finite and non-negative")

    n_codebooks, codebook_size = probabilities.shape
    if not 1 <= int(precision) <= 16:
        raise ValueError("CDF precision must be between 1 and 16 bits")
    total = 1 << int(precision)
    if codebook_size > total:
        raise ValueError(
            f"CDF precision {precision} cannot assign positive mass to {codebook_size} symbols"
        )

    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every categorical PMF must have positive total mass")
    probabilities = probabilities / row_sums

    # Reserve one count per symbol, then distribute the remaining counts using
    # largest remainders. This is deterministic and sums exactly to 2**precision.
    distributable = total - codebook_size
    scaled = probabilities * distributable
    base = np.floor(scaled).astype(np.int64)
    frequencies = base + 1
    fractions = scaled - base
    for row_idx in range(n_codebooks):
        remainder = int(total - frequencies[row_idx].sum())
        if remainder:
            order = np.argsort(-fractions[row_idx], kind="stable")
            frequencies[row_idx, order[:remainder]] += 1

    if np.any(frequencies <= 0) or np.any(frequencies.sum(axis=1) != total):
        raise RuntimeError("Failed to construct a valid quantized CDF")
    cdf = np.zeros((n_codebooks, codebook_size + 1), dtype=np.uint32)
    cdf[:, 1:] = np.cumsum(frequencies, axis=1, dtype=np.uint64).astype(np.uint32)
    return cdf


class FactorizedCategoricalEntropyModel(nn.Module):
    """One learned categorical prior per residual-VQ codebook.

    The model assumes symbols are independent over time and across codebooks,
    while allowing each codebook to have a different marginal distribution.
    """

    def __init__(
        self,
        n_codebooks: int,
        codebook_size: int,
        *,
        cdf_precision: int = 16,
    ):
        super().__init__()
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.cdf_precision = int(cdf_precision)
        if self.n_codebooks < 1 or self.codebook_size < 2:
            raise ValueError("Entropy model requires at least one codebook and two symbols")
        if not 1 <= self.cdf_precision <= 16:
            raise ValueError("cdf_precision must be between 1 and 16 bits")
        if self.codebook_size > (1 << self.cdf_precision):
            raise ValueError("cdf_precision is too small for the codebook")

        # Uniform initialization reproduces fixed-width coding before the prior
        # has observed data and avoids introducing an arbitrary code preference.
        self.logits = nn.Parameter(torch.zeros(self.n_codebooks, self.codebook_size))

    def log_probabilities(self, n_quantizers: Optional[int] = None) -> torch.Tensor:
        n_quantizers = self.n_codebooks if n_quantizers is None else int(n_quantizers)
        if not 1 <= n_quantizers <= self.n_codebooks:
            raise ValueError(f"n_quantizers must be in [1, {self.n_codebooks}]")
        return F.log_softmax(self.logits[:n_quantizers], dim=-1)

    def probabilities(self, n_quantizers: Optional[int] = None) -> torch.Tensor:
        return self.log_probabilities(n_quantizers).exp()

    @torch.no_grad()
    def calibrate_from_counts(
        self,
        counts: torch.Tensor,
        *,
        smoothing: float = 1.0,
    ) -> torch.Tensor:
        """Set each categorical PMF to its smoothed maximum-likelihood estimate.

        ``counts[q, k]`` is the number of occurrences of symbol ``k`` in
        codebook ``q``. Positive additive smoothing keeps every RVQ symbol
        encodable, including symbols absent from the calibration split.
        """
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be finite and positive")
        counts = torch.as_tensor(counts, dtype=torch.float64, device=self.logits.device)
        expected_shape = (self.n_codebooks, self.codebook_size)
        if tuple(counts.shape) != expected_shape:
            raise ValueError(f"Expected counts {expected_shape}, got {tuple(counts.shape)}")
        if not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Counts must be finite and non-negative")
        if (counts.sum(dim=-1) <= 0).any():
            raise ValueError("Every codebook must contain at least one observed symbol")

        smoothed = counts + float(smoothing)
        probabilities = smoothed / smoothed.sum(dim=-1, keepdim=True)
        self.logits.copy_(probabilities.log().to(dtype=self.logits.dtype))
        return probabilities

    def estimate_bits(self, codes: torch.Tensor) -> torch.Tensor:
        """Return exact factorized cross-entropy bits for each batch item."""
        if codes.ndim != 3:
            raise ValueError(f"Expected codes [B, Q, T], got {tuple(codes.shape)}")
        n_quantizers = int(codes.shape[1])
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.codebook_size):
            raise ValueError("RVQ code is outside the entropy model alphabet")
        log_probs = self.log_probabilities(n_quantizers)
        model_indexes = torch.arange(n_quantizers, device=codes.device).view(1, -1, 1)
        selected = log_probs[model_indexes, codes.long()]
        return -selected.sum(dim=(1, 2)) / math.log(2.0)

    @torch.no_grad()
    def quantized_cdf(self, n_quantizers: Optional[int] = None) -> np.ndarray:
        probabilities = self.probabilities(n_quantizers).detach().double().cpu().numpy()
        return probabilities_to_quantized_cdf(
            probabilities,
            precision=self.cdf_precision,
        )
