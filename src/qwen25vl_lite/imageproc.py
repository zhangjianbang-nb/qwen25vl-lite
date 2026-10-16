"""Image and video preprocessing that reproduces the HuggingFace Qwen2.5-VL processor.

The only preprocessing Qwen2.5-VL needs is:

1. ``smart_resize`` to a height/width that is a multiple of ``patch_size * merge_size``
   while staying inside a pixel budget;
2. rescale to ``[0, 1]`` and normalise with ``mean = std = 0.5`` (so the range is ``[-1, 1]``);
3. patchify into ``[tokens, in_chans * temporal_patch_size * patch_size * patch_size]``
   laid out so a ``Conv3d`` with ``kernel = stride = (T, P, P)`` consumes it directly.

Getting the patch *order* wrong is a silent failure -- the model still runs, the image is
just scrambled -- so the ordering is asserted against the HF processor in the test suite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

__all__ = [
    "ImageBudget",
    "smart_resize",
    "patchify",
    "image_grid_thw",
    "vision_tokens",
    "max_pixels_for_tokens",
    "process_image",
    "process_video",
]

# Qwen2.5-VL defaults (from the model card / processor_config.json).
OPENAI_MEAN = (0.5, 0.5, 0.5)
OPENAI_STD = (0.5, 0.5, 0.5)
DEFAULT_MIN_PIXELS = 4 * 28 * 28
DEFAULT_MAX_PIXELS = 16384 * 28 * 28


@dataclass
class ImageBudget:
    """Pixel and token clamps for one image."""

    min_pixels: int = DEFAULT_MIN_PIXELS
    max_pixels: int = DEFAULT_MAX_PIXELS
    patch_size: int = 14
    merge_size: int = 2

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    @property
    def max_vision_tokens(self) -> int:
        return self.max_pixels // (self.factor ** 2)

    @classmethod
    def from_tokens(cls, max_tokens: int, patch_size: int = 14, merge_size: int = 2, min_pixels: int = DEFAULT_MIN_PIXELS):
        """Build a budget whose ceiling is expressed in *vision tokens* instead of pixels."""
        factor = patch_size * merge_size
        return cls(min_pixels=min_pixels, max_pixels=int(max_tokens) * factor * factor,
                   patch_size=patch_size, merge_size=merge_size)


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """Resize dimensions so both are multiples of ``factor`` and the area is in budget.

    Mirrors ``qwen_vl_utils.smart_resize`` exactly, including the ``floor`` on shrink and
    ``ceil`` on grow -- using ``round`` in those branches changes the grid for some sizes.
    """
    if height < factor or width < factor:
        raise ValueError(f"image {height}x{width} is smaller than the minimum factor {factor}")
    if max_pixels < min_pixels:
        raise ValueError(f"max_pixels {max_pixels} < min_pixels {min_pixels}")

    h_bar = max(factor, int(round(height / factor)) * factor)
    w_bar = max(factor, int(round(width / factor)) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL image -> ``[C, H, W]`` float tensor in ``[-1, 1]``."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(OPENAI_MEAN, dtype=t.dtype).view(3, 1, 1)
    std = torch.tensor(OPENAI_STD, dtype=t.dtype).view(3, 1, 1)
    return (t - mean) / std


def patchify(frames: torch.Tensor, patch_size: int, temporal_patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Turn pixels into the flat patch matrix the vision tower's ``Conv3d`` expects.

    ``frames`` is either ``[C, H, W]`` (a still image, which is temporally duplicated to
    fill one temporal patch) or ``[F, C, H, W]`` (video frames, padded up to a multiple of
    ``temporal_patch_size``).

    Returns ``(pixel_values [tokens, C*T*P*P], grid_thw)``.
    """
    if frames.dim() == 3:
        c, h, w = frames.shape
        grid_t = 1
        x = frames.unsqueeze(0).expand(temporal_patch_size, c, h, w).contiguous()
    elif frames.dim() == 4:
        f, c, h, w = frames.shape
        if f % temporal_patch_size:
            pad = temporal_patch_size - (f % temporal_patch_size)
            x = torch.cat([frames, frames[-1:].expand(pad, c, h, w)], dim=0)
        else:
            x = frames
        grid_t = x.shape[0] // temporal_patch_size
    else:
        raise ValueError(f"frames must be [C,H,W] or [F,C,H,W], got {tuple(frames.shape)}")

    if h % patch_size or w % patch_size:
        raise ValueError(f"image {h}x{w} is not a multiple of patch_size {patch_size}")
    grid_h, grid_w = h // patch_size, w // patch_size
    t = temporal_patch_size

    # Layout (grid_t, grid_h, grid_w, C, T, P, P): exactly what
    #   hidden.view(-1, C, T, P, P)  +  Conv3d(kernel=stride=(T,P,P))
    # consumes.
    x = x.reshape(grid_t, t, c, grid_h, patch_size, grid_w, patch_size)
    x = x.permute(0, 3, 5, 2, 1, 4, 6).reshape(-1, c * t * patch_size * patch_size)
    return x.contiguous(), (grid_t, grid_h, grid_w)


def image_grid_thw(height: int, width: int, patch_size: int, temporal_patch_size: int) -> Tuple[int, int, int]:
    return (1, height // patch_size, width // patch_size)


def vision_tokens(grid_thw: Sequence[int], merge_size: int) -> int:
    """Number of language-model tokens an image of this grid produces."""
    t, h, w = (int(x) for x in grid_thw)
    return (t * h * w) // (merge_size ** 2)


def max_pixels_for_tokens(max_tokens: int, patch_size: int = 14, merge_size: int = 2) -> int:
    """Pixel ceiling equivalent to a vision-token ceiling."""
    factor = patch_size * merge_size
    return int(max_tokens) * factor * factor


def process_image(
    image: Image.Image,
    budget: Optional[ImageBudget] = None,
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> Dict[str, torch.Tensor]:
    """Full single-image pipeline -> ``{"pixel_values": [N, C*T*P*P], "grid_thw": [1, 3]}``."""
    budget = budget or ImageBudget(patch_size=patch_size, merge_size=merge_size)
    min_px = min_pixels if min_pixels is not None else budget.min_pixels
    max_px = max_pixels if max_pixels is not None else budget.max_pixels
    if max_px > budget.max_pixels:
        raise ValueError(f"requested max_pixels {max_px} exceeds the configured budget {budget.max_pixels}")

    width, height = image.size
    h_bar, w_bar = smart_resize(height, width, budget.factor, min_px, max_px)
    resized = image.convert("RGB").resize((w_bar, h_bar), resample=Image.BICUBIC)
    pixels, grid = patchify(_to_tensor(resized), patch_size, temporal_patch_size)
    return {
        "pixel_values": pixels,
        "grid_thw": torch.tensor([grid], dtype=torch.long),
    }


def process_video(
    frames: Sequence[Image.Image],
    budget: Optional[ImageBudget] = None,
    *,
    max_pixels: Optional[int] = None,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> Dict[str, torch.Tensor]:
    """Video pipeline.  All frames share one resized geometry, as in the HF processor."""
    if not frames:
        raise ValueError("video has no frames")
    budget = budget or ImageBudget(patch_size=patch_size, merge_size=merge_size)
    max_px = max_pixels if max_pixels is not None else budget.max_pixels

    width, height = frames[0].size
    h_bar, w_bar = smart_resize(height, width, budget.factor, budget.min_pixels, max_px)
    arr = np.stack(
        [np.asarray(f.convert("RGB").resize((w_bar, h_bar), resample=Image.BICUBIC), dtype=np.float32) for f in frames],
        axis=0,
    )
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous() / 255.0
    mean = torch.tensor(OPENAI_MEAN, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(OPENAI_STD, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    pixels, grid = patchify(x, patch_size, temporal_patch_size)
    return {
        "pixel_values": pixels,
        "grid_thw": torch.tensor([grid], dtype=torch.long),
    }
