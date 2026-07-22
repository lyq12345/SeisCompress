from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from dac.model.dac import DAC, Encoder, Decoder, init_weights
from dac.nn.layers import WNConv1d

from factorized_entropy import FactorizedCategoricalEntropyModel
from first_order_entropy import FirstOrderCategoricalEntropyModel
from quantize_stable import ResidualVectorQuantizeStable
from rans_codec import (
    FIRST_ORDER_STREAM_MAGIC,
    STREAM_MAGIC,
    FactorizedRansCodec,
    FirstOrderRansCodec,
)
from seis_encoder import SeisLMEncoder


class SeisDAC(DAC):
    def __init__(
        self,
        in_channels=3,
        use_stable_quantizer=True,
        use_seislm_encoder=False,
        seislm_encoder_checkpoint="",
        freeze_seislm_extractor=False,
        use_entropy_model=False,
        use_first_order_entropy_model=False,
        entropy_temperature=0.1,
        entropy_cdf_precision=16,
        *args,
        **kwargs,
    ):
        quantizer_dropout = kwargs.get("quantizer_dropout", False)
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.use_stable_quantizer = bool(use_stable_quantizer)
        self.entropy_temperature = float(entropy_temperature)

        if use_stable_quantizer:
            self.quantizer = ResidualVectorQuantizeStable(
                input_dim=self.latent_dim,
                n_codebooks=self.n_codebooks,
                codebook_size=self.codebook_size,
                codebook_dim=self.codebook_dim,
                quantizer_dropout=quantizer_dropout,
            )
            self.quantizer.apply(init_weights)

        if use_seislm_encoder:
            # Plan A: pretrained SeisLM conv extractor + adapter (total 8x
            # downsampling, must match prod(decoder_rates)).
            self.encoder = SeisLMEncoder(
                latent_dim=self.latent_dim,
                checkpoint_path=seislm_encoder_checkpoint,
                freeze_extractor=freeze_seislm_extractor,
            )
        else:
            # Modify the encoder to accept `in_channels` instead of 1
            self.encoder.block[0] = WNConv1d(in_channels, self.encoder_dim, kernel_size=7, padding=3)

        # Modify the decoder to output `in_channels` instead of 1
        # The decoder.model is a nn.Sequential, the second to last layer is WNConv1d
        out_dim = self.decoder.model[-2].in_channels
        self.decoder.model[-2] = WNConv1d(out_dim, in_channels, kernel_size=7, padding=3)

        if use_seislm_encoder:
            # Conv geometry changed; recompute chunked-inference delay.
            self.delay = self.get_delay()

        if use_entropy_model and not self.use_stable_quantizer:
            raise ValueError("The differentiable entropy loss requires the stable quantizer")
        self.entropy_model = None
        if use_entropy_model:
            self.entropy_model = FactorizedCategoricalEntropyModel(
                n_codebooks=self.n_codebooks,
                codebook_size=self.codebook_size,
                cdf_precision=entropy_cdf_precision,
            )
        self.first_order_entropy_model = None
        if use_first_order_entropy_model:
            self.first_order_entropy_model = FirstOrderCategoricalEntropyModel(
                n_codebooks=self.n_codebooks,
                codebook_size=self.codebook_size,
                cdf_precision=entropy_cdf_precision,
            )

    def _entropy_log_probabilities(self):
        if self.entropy_model is None:
            return None
        return self.entropy_model.log_probabilities()

    def _quantize(self, z: torch.Tensor, n_quantizers: int = None):
        if self.entropy_model is not None:
            return self.quantizer(
                z,
                n_quantizers,
                entropy_log_probs=self._entropy_log_probabilities(),
                entropy_temperature=self.entropy_temperature,
            )
        result = self.quantizer(z, n_quantizers)
        if len(result) == 5:
            result = (*result, z.new_zeros(z.shape[0]))
        return result

    def factorized_rans_codec(self) -> FactorizedRansCodec:
        if self.entropy_model is None:
            raise RuntimeError("This SeisDAC checkpoint has no factorized entropy model")
        return FactorizedRansCodec(
            self.entropy_model.quantized_cdf(),
            precision=self.entropy_model.cdf_precision,
        )

    def first_order_rans_codec(self) -> FirstOrderRansCodec:
        if self.first_order_entropy_model is None:
            raise RuntimeError("This SeisDAC checkpoint has no first-order entropy model")
        marginal_cdf, conditional_cdf = self.first_order_entropy_model.quantized_cdfs()
        return FirstOrderRansCodec(
            marginal_cdf,
            conditional_cdf,
            precision=self.first_order_entropy_model.cdf_precision,
        )

    def _rans_codec_for_encoding(self):
        if self.first_order_entropy_model is not None:
            return self.first_order_rans_codec()
        return self.factorized_rans_codec()

    def _rans_codec_for_streams(self, streams: Sequence[bytes]):
        magics = {stream[:4] for stream in streams}
        if len(magics) != 1:
            raise ValueError("All streams in a batch must use the same entropy coder")
        magic = magics.pop()
        if magic == FIRST_ORDER_STREAM_MAGIC:
            return self.first_order_rans_codec()
        if magic == STREAM_MAGIC:
            return self.factorized_rans_codec()
        raise ValueError("Unsupported SeisDAC rANS stream magic")

    @torch.no_grad()
    def encode_to_rans(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ) -> Sequence[bytes]:
        """Encode a waveform batch into independently framed rANS streams."""
        if self.training:
            raise RuntimeError("Call model.eval() before entropy encoding")
        if audio_data.ndim != 3:
            raise ValueError(f"Expected audio [B, C, T], got {tuple(audio_data.shape)}")
        original_length = int(audio_data.shape[-1])
        prepared = self.preprocess(audio_data, sample_rate)
        encoder_latent = self.encoder(prepared)
        # Symbol extraction does not need the differentiable rate surrogate.
        _, codes, _, _, _, _ = self.quantizer(encoder_latent, n_quantizers)
        codec = self._rans_codec_for_encoding()
        codes_np = codes.detach().cpu().numpy().astype(np.int64, copy=False)
        return [
            codec.encode(sample_codes, original_length=original_length)
            for sample_codes in codes_np
        ]

    @torch.no_grad()
    def decode_from_rans(self, streams: Sequence[bytes]) -> torch.Tensor:
        """Decode rANS streams into a same-length waveform batch."""
        if self.training:
            raise RuntimeError("Call model.eval() before entropy decoding")
        if not streams:
            raise ValueError("At least one rANS stream is required")
        codec = self._rans_codec_for_streams(streams)
        decoded = [codec.decode(stream) for stream in streams]
        shapes = {codes.shape for codes, _ in decoded}
        lengths = {length for _, length in decoded}
        if len(shapes) != 1 or len(lengths) != 1:
            raise ValueError("All streams in a decoded batch must have matching shapes and lengths")
        codes_np = np.stack([codes for codes, _ in decoded], axis=0)
        device = next(self.parameters()).device
        codes = torch.from_numpy(codes_np).to(device=device, dtype=torch.long)
        z_q, _, _ = self.quantizer.from_codes(codes)
        original_length = lengths.pop()
        return self.decode(z_q)[..., :original_length]

    def encode(self, audio_data: torch.Tensor, n_quantizers: int = None):
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss, _ = self._quantize(z, n_quantizers)
        return z, codes, latents, commitment_loss, codebook_loss

    def forward(self, audio_data: torch.Tensor, sample_rate: int = None, n_quantizers: int = None):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        encoder_latent = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss, rate_bits = self._quantize(
            encoder_latent,
            n_quantizers,
        )

        x = self.decode(z)
        effective_sample_rate = self.sample_rate if sample_rate is None else sample_rate
        duration_sec = length / float(effective_sample_rate)
        rate_bps = rate_bits / duration_sec
        symbols_per_sample = max(1, int(codes.shape[1] * codes.shape[2]))
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "encoder_latent": encoder_latent,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
            "rate/estimated_bits": rate_bits,
            "rate/estimated_bps": rate_bps,
            "rate/estimated_kbps": rate_bps / 1000.0,
            "rate/bits_per_symbol": rate_bits / symbols_per_sample,
        }
