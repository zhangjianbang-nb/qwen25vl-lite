"""Shared building blocks (norm layers).

Kept in one place so the text decoder and the vision tower cannot drift apart.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["RMSNorm"]


class RMSNorm(nn.Module):
    """RMS normalisation.

    Qwen2.5-VL uses the *plain* formulation -- ``eps`` is added inside the square root
    (``x * rsqrt(mean(x^2) + eps)``) -- not the zero-centred variant from Qwen2.5 text-only
    models.  Computing in float32 and casting back matches HF and avoids bf16 drift.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
