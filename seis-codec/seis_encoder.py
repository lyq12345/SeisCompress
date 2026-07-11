"""SeisLM-based encoder for SeisDAC (Plan A).

Replaces the DAC convolutional encoder with SeisLM's pretrained
Wav2Vec2FeatureEncoder (2 conv layers, LayerNorm + GELU, 4x downsampling,
256 channels), followed by a small trainable adapter that:
  - downsamples once more (stride 2) so the total factor matches the DAC
    decoder's 8x upsampling (decoder_rates [2, 2, 2]),
  - projects 256 -> latent_dim for the residual VQ,
  - normalizes the output scale (GroupNorm) so encoder latents stay bounded.

Neither seisLM nor descript-audio-codec sources are modified.
"""

from pathlib import Path
import sys

_SEISLM_SRC = Path(__file__).resolve().parents[1] / "seisLM"
if _SEISLM_SRC.exists() and str(_SEISLM_SRC) not in sys.path:
    sys.path.insert(0, str(_SEISLM_SRC))

import ml_collections
import torch
import torch.nn as nn

from dac.nn.layers import WNConv1d
from seisLM.model.foundation.conv_encoder import Wav2Vec2FeatureEncoder

_STATE_DICT_PREFIX = "model.wav2vec2.feature_extractor."

# Matches pretrained_seislm_base (epoch=39-step=1203000.ckpt).
_SEISLM_FE_CONFIG = dict(
    conv_dim=[256, 256],
    conv_kernel=[3, 3],
    conv_stride=[2, 2],
    conv_bias=True,
    input_dim=3,
    num_feat_extract_layers=2,
    feat_extract_norm="layer",
    use_rms_norm=False,
)


def load_seislm_feature_extractor(checkpoint_path: str) -> Wav2Vec2FeatureEncoder:
    """Build a Wav2Vec2FeatureEncoder and load pretrained SeisLM weights."""
    config = ml_collections.ConfigDict(_SEISLM_FE_CONFIG)
    extractor = Wav2Vec2FeatureEncoder(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        k[len(_STATE_DICT_PREFIX):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(_STATE_DICT_PREFIX)
    }
    missing, unexpected = extractor.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected
    print(
        f"Loaded SeisLM feature extractor from {checkpoint_path} "
        f"({len(state_dict)} tensors)"
    )
    return extractor


class SeisLMEncoder(nn.Module):
    """Pretrained SeisLM conv extractor + trainable adapter, total 8x downsampling."""

    def __init__(
        self,
        latent_dim: int = 512,
        checkpoint_path: str = "",
        freeze_extractor: bool = False,
    ):
        super().__init__()
        if checkpoint_path:
            self.feature_extractor = load_seislm_feature_extractor(checkpoint_path)
        else:
            print("WARNING: no SeisLM checkpoint given; feature extractor is randomly initialized.")
            self.feature_extractor = Wav2Vec2FeatureEncoder(
                ml_collections.ConfigDict(_SEISLM_FE_CONFIG)
            )

        if freeze_extractor:
            self.feature_extractor._freeze_parameters()  # pylint: disable=protected-access

        feature_dim = _SEISLM_FE_CONFIG["conv_dim"][-1]
        self.adapter = nn.Sequential(
            WNConv1d(feature_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            # Keeps latent scale bounded; addresses encoder-norm blowups.
            nn.GroupNorm(1, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=3, T) -> (B, latent_dim, ~T/8)
        features = self.feature_extractor(x)
        return self.adapter(features)
