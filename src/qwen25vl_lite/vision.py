"""Qwen2.5-VL vision tower (faithful port of ``Qwen2_5_VisionTransformerPretrainedModel``).

Three details drive correctness and are easy to get subtly wrong:

* **Patch layout.** ``patch_embed`` is a ``Conv3d`` with ``kernel == stride == (T, P, P)``;
  the incoming pixel matrix must already be in ``(grid_t, grid_h, grid_w, C, T, P, P)``
  order.  ``imageproc.patchify`` produces exactly that.
* **Merge-unit permutation.** Tokens are permuted in ``spatial_merge_unit`` blocks so that
  each ``m x m`` block is contiguous, window attention runs on the permuted order, and the
  merger output is permuted back with ``argsort(window_index)``.
* **Window vs full attention.** Only the layers listed in ``fullatt_block_indexes``
  (``[7, 15, 23, 31]``) attend globally; the rest attend inside ``window_size`` windows.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .attention import sdpa
from .config import VisionConfig
from .layers import RMSNorm
from .rope import apply_rotary_pos_emb_vision, build_inv_freq, vision_cos_sin, vision_rope_position_ids

__all__ = [
    "VisionPatchEmbed",
    "PatchMerger",
    "VisionAttention",
    "VisionBlock",
    "VisionTransformer",
    "get_vision_cu_seqlens",
    "get_window_index",
    "packed_attention",
]


class VisionPatchEmbed(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.temporal_patch_size = cfg.temporal_patch_size
        self.in_chans = cfg.in_chans
        self.embed_dim = cfg.hidden_size
        k = (cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        self.proj = nn.Conv3d(cfg.in_chans, cfg.hidden_size, kernel_size=k, stride=k, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target = self.proj.weight.dtype
        x = hidden_states.view(
            -1, self.in_chans, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(x.to(target)).view(-1, self.embed_dim)


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.dim = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        backends: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)  # [1, heads, seq, hd]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = packed_attention(q, k, v, cu_seqlens, scale=self.scaling, backends=backends)
        # packed_attention returns [1, heads, seq, head_dim]; ``proj`` expects
        # [seq, heads * head_dim], i.e. a *sequence-major* flattening.  Reshaping directly
        # would silently produce head-major order.
        return self.proj(out.transpose(1, 2).reshape(seq_len, -1))


def packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: Optional[float] = None,
    backends: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Non-causal attention over variable-length packed segments.

    Segments of equal length are batched into a single SDPA call; segments of different
    lengths are handled group-by-group.  This is exact (no padding, no block-diagonal
    mask) and avoids materialising an O(total^2) mask, which matters because a single
    16k-token image would need a 1 GiB mask.
    """
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    if not lengths:
        return q
    if len(lengths) == 1:
        return sdpa(q, k, v, scale=scale, backends=backends, enable_gqa=False)

    out = torch.empty_like(q)
    unique = sorted(set(lengths))
    for length in unique:
        starts = [int(cu_seqlens[i]) for i, L in enumerate(lengths) if L == length]
        if len(starts) == 1:
            s = starts[0]
            out[:, :, s : s + length, :] = sdpa(
                q[:, :, s : s + length, :], k[:, :, s : s + length, :], v[:, :, s : s + length, :],
                scale=scale, backends=backends, enable_gqa=False,
            )
            continue
        qs = torch.stack([q[0, :, s : s + length, :] for s in starts], dim=0)  # [n, heads, L, hd]
        ks = torch.stack([k[0, :, s : s + length, :] for s in starts], dim=0)
        vs = torch.stack([v[0, :, s : s + length, :] for s in starts], dim=0)
        o = sdpa(qs, ks, vs, scale=scale, backends=backends, enable_gqa=False)
        for i, s in enumerate(starts):
            out[0, :, s : s + length, :] = o[i]
    return out


class VisionMLP(nn.Module):
    """SwiGLU MLP with biases (the vision tower keeps them)."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(cfg.hidden_size, eps=1e-6)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)

    def forward(self, hidden_states, cos, sin, cu_seqlens, backends=None):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin, cu_seqlens, backends)
        return hidden_states + self.mlp(self.norm2(hidden_states))


def get_vision_cu_seqlens(grid_thw: torch.Tensor, merge_temporal: bool = False) -> torch.Tensor:
    """Cumulative segment boundaries for packed vision attention.

    With ``merge_temporal=False`` each *frame* is its own segment (``h*w`` per frame,
    ``t`` segments per entry) -- the Qwen2-VL/Qwen2.5-VL convention.
    """
    if merge_temporal:
        seqlens = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    else:
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)


def get_window_index(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Window-attention permutation and cumulative window boundaries.

    Returns ``(window_index, cu_window_seqlens)``.  ``window_index`` indexes *merge units*
    (not tokens); the caller expands it over ``spatial_merge_unit`` tokens.
    """
    window_index: List[torch.Tensor] = []
    cu_window_seqlens: List[int] = [0]
    window_index_id = 0
    vit_window = window_size // spatial_merge_size // patch_size
    merge_unit = spatial_merge_size ** 2

    for grid_t, grid_h, grid_w in grid_thw.tolist():
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        llm_h = grid_h // spatial_merge_size
        llm_w = grid_w // spatial_merge_size
        index = torch.arange(grid_t * llm_h * llm_w).reshape(grid_t, llm_h, llm_w)
        pad_h = vit_window - llm_h % vit_window
        pad_w = vit_window - llm_w % vit_window
        n_win_h = (llm_h + pad_h) // vit_window
        n_win_w = (llm_w + pad_w) // vit_window
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(grid_t, n_win_h, vit_window, n_win_w, vit_window)
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, n_win_h * n_win_w, vit_window, vit_window
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_tmp = seqlens.cumsum(0) * merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_tmp.tolist())
        window_index_id += grid_t * llm_h * llm_w

    cu = torch.unique_consecutive(torch.tensor(cu_window_seqlens, dtype=torch.int32))
    return torch.cat(window_index, dim=0), cu


class VisionTransformer(nn.Module):
    """The vision tower.  ``forward`` returns the *post-merger* language-model features."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.spatial_merge_unit = cfg.spatial_merge_size ** 2
        self.patch_size = cfg.patch_size
        self.window_size = cfg.window_size
        self.fullatt_block_indexes = tuple(cfg.fullatt_block_indexes)
        self.patch_embed = VisionPatchEmbed(cfg)
        self._inv_freq: Optional[torch.Tensor] = None
        self.blocks = nn.ModuleList([VisionBlock(cfg) for _ in range(cfg.depth)])
        self.merger = PatchMerger(cfg.out_hidden_size, cfg.hidden_size, cfg.spatial_merge_size)

    def rotary_inv_freq(self, device=None) -> torch.Tensor:
        """Lazily built vision RoPE frequencies (not a buffer -- see text.RotaryEmbedding)."""
        if self._inv_freq is None or self._inv_freq.device != torch.device(device or "cpu"):
            self._inv_freq = build_inv_freq(self.cfg.rotary_dim, 10_000.0, device=device)
        return self._inv_freq

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        backends: Optional[Sequence[str]] = None,
        return_intermediates: bool = False,
    ):
        """``pixel_values`` is ``[tokens, C*T*P*P]``; returns ``[merged_tokens, out_hidden]``.

        With ``return_intermediates=True`` a dict is returned carrying the window
        permutation, so an external token-merge/prune stage can undo it.
        """
        device = pixel_values.device
        position_ids = vision_rope_position_ids(grid_thw.to(device), self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw.to(device))
        window_index, cu_window_seqlens = get_window_index(
            grid_thw.to(device), self.spatial_merge_size, self.window_size, self.patch_size
        )

        hidden = self.patch_embed(pixel_values)
        seq_len = hidden.shape[0]
        unit = self.spatial_merge_unit
        hidden = hidden.reshape(seq_len // unit, unit, -1)[window_index].reshape(seq_len, -1)

        cos, sin = vision_cos_sin(position_ids, self.rotary_inv_freq(device), dtype=hidden.dtype)
        cos = cos.reshape(seq_len // unit, unit, -1)[window_index].reshape(seq_len, -1)
        sin = sin.reshape(seq_len // unit, unit, -1)[window_index].reshape(seq_len, -1)

        cu_seqlens = cu_seqlens.to(device)
        cu_window_seqlens = cu_window_seqlens.to(device)
        for layer_num, block in enumerate(self.blocks):
            cu_now = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            hidden = block(hidden, cos, sin, cu_now, backends)

        reverse = torch.argsort(window_index)
        merged = self.merger(hidden)[reverse, :]

        if return_intermediates:
            return {
                "last_hidden_state": hidden,
                "pooler_output": merged,
                "window_index": window_index,
                "reverse_index": reverse,
                "cu_seqlens": cu_seqlens,
                "cu_window_seqlens": cu_window_seqlens,
                "position_ids": position_ids,
            }
        return merged
