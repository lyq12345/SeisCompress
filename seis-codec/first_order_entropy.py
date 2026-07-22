"""Causal first-order categorical entropy model for residual-VQ codes."""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from factorized_entropy import probabilities_to_quantized_cdf


class FirstOrderCategoricalEntropyModel(nn.Module):
    """CDF tables for ``p(z[q, t] | z[q, t - 1])`` with marginal startup.

    The tables are calibrated rather than gradient-trained. Storing quantized
    CDFs directly ensures estimated rates use the same probabilities as rANS.
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

        uniform = np.full(
            (self.n_codebooks, self.codebook_size),
            1.0 / self.codebook_size,
            dtype=np.float64,
        )
        marginal_cdf = probabilities_to_quantized_cdf(
            uniform,
            precision=self.cdf_precision,
        )
        conditional_cdf = np.repeat(
            marginal_cdf[:, None, :],
            self.codebook_size,
            axis=1,
        )
        self.register_buffer(
            "marginal_cdf",
            torch.from_numpy(marginal_cdf.astype(np.int32, copy=False)),
        )
        self.register_buffer(
            "conditional_cdf",
            torch.from_numpy(conditional_cdf.astype(np.int32, copy=False)),
        )

    @torch.no_grad()
    def calibrate_from_counts(
        self,
        marginal_counts: torch.Tensor,
        transition_counts: torch.Tensor,
        *,
        marginal_smoothing: float = 1.0,
        backoff_concentration: float = 32.0,
    ) -> None:
        """Fit marginal and transition CDFs with marginal-distribution backoff."""
        if not math.isfinite(marginal_smoothing) or marginal_smoothing <= 0:
            raise ValueError("marginal_smoothing must be finite and positive")
        if not math.isfinite(backoff_concentration) or backoff_concentration <= 0:
            raise ValueError("backoff_concentration must be finite and positive")

        marginal_counts_np = np.asarray(
            torch.as_tensor(marginal_counts).cpu(),
            dtype=np.float64,
        )
        transition_counts_np = np.asarray(
            torch.as_tensor(transition_counts).cpu(),
            dtype=np.float64,
        )
        marginal_shape = (self.n_codebooks, self.codebook_size)
        transition_shape = (
            self.n_codebooks,
            self.codebook_size,
            self.codebook_size,
        )
        if marginal_counts_np.shape != marginal_shape:
            raise ValueError(
                f"Expected marginal counts {marginal_shape}, got {marginal_counts_np.shape}"
            )
        if transition_counts_np.shape != transition_shape:
            raise ValueError(
                f"Expected transition counts {transition_shape}, got {transition_counts_np.shape}"
            )
        if (
            not np.all(np.isfinite(marginal_counts_np))
            or not np.all(np.isfinite(transition_counts_np))
            or np.any(marginal_counts_np < 0)
            or np.any(transition_counts_np < 0)
        ):
            raise ValueError("Counts must be finite and non-negative")
        if np.any(marginal_counts_np.sum(axis=-1) <= 0):
            raise ValueError("Every codebook must contain at least one observed symbol")

        smoothed_marginal = marginal_counts_np + float(marginal_smoothing)
        marginal_probabilities = smoothed_marginal / smoothed_marginal.sum(
            axis=-1,
            keepdims=True,
        )
        context_totals = transition_counts_np.sum(axis=-1, keepdims=True)
        conditional_probabilities = (
            transition_counts_np
            + float(backoff_concentration) * marginal_probabilities[:, None, :]
        ) / (context_totals + float(backoff_concentration))

        marginal_cdf = probabilities_to_quantized_cdf(
            marginal_probabilities,
            precision=self.cdf_precision,
        )
        conditional_cdf = probabilities_to_quantized_cdf(
            conditional_probabilities.reshape(-1, self.codebook_size),
            precision=self.cdf_precision,
        ).reshape(
            self.n_codebooks,
            self.codebook_size,
            self.codebook_size + 1,
        )
        self.marginal_cdf.copy_(
            torch.from_numpy(marginal_cdf.astype(np.int32, copy=False)).to(
                self.marginal_cdf.device
            )
        )
        self.conditional_cdf.copy_(
            torch.from_numpy(conditional_cdf.astype(np.int32, copy=False)).to(
                self.conditional_cdf.device
            )
        )

    def estimate_bits(self, codes: torch.Tensor) -> torch.Tensor:
        """Return CDF-quantized first-order cross-entropy bits per batch item."""
        if codes.ndim != 3:
            raise ValueError(f"Expected codes [B, Q, T], got {tuple(codes.shape)}")
        n_quantizers = int(codes.shape[1])
        if not 1 <= n_quantizers <= self.n_codebooks:
            raise ValueError(f"n_quantizers must be in [1, {self.n_codebooks}]")
        if codes.shape[2] < 1:
            raise ValueError("At least one latent frame is required")
        if codes.numel() and (codes.min() < 0 or codes.max() >= self.codebook_size):
            raise ValueError("RVQ code is outside the entropy model alphabet")

        codes = codes.long()
        marginal_frequencies = (
            self.marginal_cdf[:n_quantizers, 1:]
            - self.marginal_cdf[:n_quantizers, :-1]
        ).to(device=codes.device, dtype=torch.float64)
        codebook_indexes = torch.arange(n_quantizers, device=codes.device).view(1, -1)
        first_frequencies = marginal_frequencies[
            codebook_indexes,
            codes[:, :, 0],
        ]
        bits = -torch.log2(first_frequencies).sum(dim=1)

        if codes.shape[2] > 1:
            conditional_frequencies = (
                self.conditional_cdf[:n_quantizers, :, 1:]
                - self.conditional_cdf[:n_quantizers, :, :-1]
            ).to(device=codes.device, dtype=torch.float64)
            previous = codes[:, :, :-1]
            current = codes[:, :, 1:]
            transition_frequencies = conditional_frequencies[
                codebook_indexes.unsqueeze(-1),
                previous,
                current,
            ]
            bits -= torch.log2(transition_frequencies).sum(dim=(1, 2))

        bits += codes.shape[1] * codes.shape[2] * float(self.cdf_precision)
        return bits.to(dtype=torch.float32)

    @torch.no_grad()
    def quantized_cdfs(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.marginal_cdf.detach().cpu().numpy().astype(np.uint32, copy=False),
            self.conditional_cdf.detach().cpu().numpy().astype(np.uint32, copy=False),
        )
