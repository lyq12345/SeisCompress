import torch
import torch.nn as nn
from dac.model.dac import DAC, Encoder, Decoder, init_weights
from dac.nn.layers import WNConv1d

from quantize_stable import ResidualVectorQuantizeStable
from seis_encoder import SeisLMEncoder


class SeisDAC(DAC):
    def __init__(
        self,
        in_channels=3,
        use_stable_quantizer=True,
        use_seislm_encoder=False,
        seislm_encoder_checkpoint="",
        freeze_seislm_extractor=False,
        *args,
        **kwargs,
    ):
        quantizer_dropout = kwargs.get("quantizer_dropout", False)
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels

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

    def encode(self, audio_data: torch.Tensor, n_quantizers: int = None):
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(z, n_quantizers)
        return z, codes, latents, commitment_loss, codebook_loss

    def forward(self, audio_data: torch.Tensor, sample_rate: int = None, n_quantizers: int = None):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        encoder_latent = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            encoder_latent, n_quantizers
        )

        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "encoder_latent": encoder_latent,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }
