import torch
import torch.nn as nn
from dac.model.dac import DAC, Encoder, Decoder
from dac.nn.layers import WNConv1d

class SeisDAC(DAC):
    def __init__(self, in_channels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        
        # Modify the encoder to accept `in_channels` instead of 1
        self.encoder.block[0] = WNConv1d(in_channels, self.encoder_dim, kernel_size=7, padding=3)
        
        # Modify the decoder to output `in_channels` instead of 1
        # The decoder.model is a nn.Sequential, the second to last layer is WNConv1d
        out_dim = self.decoder.model[-2].in_channels
        self.decoder.model[-2] = WNConv1d(out_dim, in_channels, kernel_size=7, padding=3)

    def encode(self, audio_data: torch.Tensor, n_quantizers: int = None):
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(z, n_quantizers)
        return z, codes, latents, commitment_loss, codebook_loss

    def forward(self, audio_data: torch.Tensor, sample_rate: int = None, n_quantizers: int = None):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )

        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }
