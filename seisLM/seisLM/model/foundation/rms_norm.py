"""RMSNorm compatibility helpers."""

import torch
from torch import Tensor, nn


if hasattr(nn, "RMSNorm"):
  RMSNorm = nn.RMSNorm
else:

  class RMSNorm(nn.Module):
    """Fallback RMSNorm for torch versions that do not provide nn.RMSNorm."""

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float | None = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
      super().__init__()
      if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
      self.normalized_shape = tuple(normalized_shape)
      self.eps = eps
      self.elementwise_affine = elementwise_affine
      if elementwise_affine:
        self.weight = nn.Parameter(
          torch.empty(self.normalized_shape, device=device, dtype=dtype)
        )
      else:
        self.register_parameter("weight", None)
      self.reset_parameters()

    def reset_parameters(self) -> None:
      if self.elementwise_affine:
        nn.init.ones_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
      eps = self.eps
      if eps is None:
        eps = torch.finfo(x.dtype).eps
      output = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
      if self.weight is not None:
        output = output * self.weight
      return output
