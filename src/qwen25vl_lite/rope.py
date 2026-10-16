"""Rotary position embeddings for Qwen2.5-VL.

Two distinct schemes live here:

* **MRoPE** for the text decoder.  ``mrope_section`` (``[16, 24, 24]`` for Qwen2.5-VL)
  splits the rotary channel dimension into temporal / height / width thirds.  The
  position tensor has shape ``[3, batch, seq]`` and the three slices are interleaved
  back into a single ``head_dim`` block.
* **2D RoPE** for the vision tower.  Position ids are ``[N, 2]`` (height, width) laid
  out block-major over ``merge x merge`` blocks; the resulting frequency vector is
  duplicated to fill ``head_dim``.

Everything here is written to match ``transformers.models.qwen2_5_vl`` element for
element -- ``tests/test_model_vs_hf.py`` asserts that with no tolerance, against the
version pinned in ``pyproject.toml`` (HF moves these internals between releases; see the
note in that file).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def build_inv_freq(dim: int, theta: float = 10_000.0, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Inverse frequencies for a rotary embedding of width ``dim``.

    ``dim`` is the *rotary* width, i.e. the number of channels actually rotated; the
    number of distinct frequencies produced is ``dim // 2``.
    """
    idx = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    return (1.0 / (theta ** (idx / dim))).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension onto the first (and negate it)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


# --------------------------------------------------------------------------------------
# MRoPE (text decoder)
# --------------------------------------------------------------------------------------


def mrope_cos_sin(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine/sine tables for MRoPE.

    ``position_ids`` has shape ``[3, batch, seq]``; the result is ``[3, batch, seq, head_dim]``
    where ``head_dim == 2 * inv_freq.numel()``.
    """
    if position_ids.dim() != 3 or position_ids.shape[0] != 3:
        raise ValueError(f"position_ids must be [3, batch, seq], got {tuple(position_ids.shape)}")
    n_axes, batch, _ = position_ids.shape
    inv = inv_freq.to(torch.float32)[None, None, :, None].expand(n_axes, batch, -1, 1)
    pid = position_ids[:, :, None, :].to(torch.float32)
    freqs = torch.matmul(inv, pid).transpose(2, 3)  # [3, batch, seq, dim//2]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: Sequence[int],
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply MRoPE to query/key tensors.

    ``cos``/``sin`` are ``[3, batch, seq, head_dim]``.  The channel layout follows the
    canonical Qwen implementation: ``mrope_section`` (which sums to ``head_dim // 2``) is
    **repeated as a list** -- ``[16, 24, 24] -> [16, 24, 24, 16, 24, 24]`` -- and the six
    slices are taken from axes ``0, 1, 2, 0, 1, 2`` in order.  Doubling each entry instead
    of repeating the list yields a different (wrong) interleaving while still producing a
    correctly shaped tensor; ``tests/test_mrope_matches_hf.py`` pins this down.
    """
    sections: List[int] = list(int(s) for s in mrope_section) * 2
    if sum(sections) != cos.shape[-1]:
        raise ValueError(
            f"mrope_section {tuple(mrope_section)} doubled to {sum(sections)} does not cover "
            f"head_dim {cos.shape[-1]}"
        )
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(sections, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(sections, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def vision_position_ids(
    start_position: int,
    grid_thw: Sequence[int],
    spatial_merge_size: int,
    temporal_merge_size: int = 1,
    time_interval: int = 1,
    device=None,
) -> torch.Tensor:
    """3D position ids for one image/video, offset by ``start_position``.

    Returns ``[3, tokens]`` ordered ``(t, h, w)``.  All three axes are offset by
    ``start_position``; the temporal axis is additionally scaled by ``time_interval``
    *before* the offset is applied, so ``t`` advances by ``time_interval`` per temporal
    patch while still continuing the running sequence position.
    """
    t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    llm_t = t // temporal_merge_size
    llm_h = h // spatial_merge_size
    llm_w = w // spatial_merge_size

    pos_t = torch.arange(llm_t, device=device) * time_interval
    pos_h = torch.arange(llm_h, device=device) + start_position
    pos_w = torch.arange(llm_w, device=device) + start_position

    t_grid, h_grid, w_grid = torch.meshgrid(pos_t, pos_h, pos_w, indexing="ij")
    ids = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
    ids[0] += start_position  # applied after the time_interval multiply
    return ids


def mm_token_type_ids(input_ids: torch.Tensor, image_token_id: int, video_token_id: int) -> torch.Tensor:
    """0 = text, 1 = image, 2 = video, matching the ``mm_token_type_ids`` HF expects."""
    out = torch.zeros_like(input_ids, dtype=torch.int32)
    out = torch.where(input_ids == image_token_id, torch.ones_like(out), out)
    out = torch.where(input_ids == video_token_id, torch.full_like(out, 2), out)
    return out


def get_rope_index(
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    *,
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
    tokens_per_second: float = 2.0,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute MRoPE position ids for a batch of text/image/video sequences.

    Returns ``(position_ids [3, batch, seq], rope_deltas [batch, 1])``.  ``rope_deltas``
    is the constant added to the running text position during incremental decoding:
    ``max(position) + 1 - seq_len``.

    The algorithm mirrors HF exactly: consecutive tokens of the same modality form one
    group; text groups advance the position cursor by their length, vision groups consume
    one grid from the corresponding iterator and advance the cursor by
    ``max(llm_h, llm_w)`` -- **not** by the number of vision tokens, because the 2D grid
    positions are laid out in an overlapping coordinate system.
    """
    import itertools

    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")

    batch = input_ids.shape[0]
    device = input_ids.device
    position_ids = torch.zeros(3, batch, input_ids.shape[1], dtype=input_ids.dtype, device=device)
    deltas: List[int] = []

    grids = {1: iter(image_grid_thw) if image_grid_thw is not None else None,
             2: iter(video_grid_thw) if video_grid_thw is not None else None}
    if second_per_grid_ts is not None:
        seconds = iter(second_per_grid_ts)
    else:
        seconds = None

    for b in range(batch):
        ids = input_ids[b]
        types = mm_token_type_ids(ids, image_token_id, video_token_id)
        if attention_mask is not None:
            keep = attention_mask[b].bool()
            ids = ids[keep]
            types = types[keep]

        groups = []
        for key, grp in itertools.groupby(enumerate(types.tolist()), lambda x: x[1]):
            items = list(grp)
            groups.append((key, items[0][0], items[-1][0] + 1))

        cursor = 0
        chunks: List[torch.Tensor] = []
        for modality, start, end in groups:
            if modality == 0:  # text
                text_len = end - start
                chunks.append(torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + cursor)
                cursor += text_len
            else:
                if grids[modality] is None:
                    raise ValueError(
                        f"sequence contains modality {modality} tokens but no matching grid_thw was provided"
                    )
                grid = next(grids[modality])
                if modality == 2:
                    if seconds is not None:
                        # Position ids are integer steps here, so a fractional seconds-per-grid
                        # value cannot be represented.  Truncating it silently would produce a
                        # plausible but wrong temporal axis, so it is refused instead.  The video
                        # path is not part of the verified scope (see README).
                        spg = float(next(seconds))
                        if spg != int(spg):
                            raise ValueError(
                                f"fractional second_per_grid_ts ({spg}) is not supported: the "
                                f"temporal interval would not be an integer position step"
                            )
                        interval = tokens_per_second * int(spg)
                    else:
                        interval = tokens_per_second
                else:
                    interval = 1
                chunks.append(
                    vision_position_ids(cursor, grid.tolist(), spatial_merge_size, 1, interval, device=device)
                )
                cursor += max(int(grid[1]), int(grid[2])) // spatial_merge_size

        if not chunks:
            raise ValueError("empty sequence: no tokens to position")
        positions = torch.cat(chunks, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, b, attention_mask[b].bool()] = positions.to(device)
        else:
            position_ids[:, b] = positions.to(device)
        deltas.append(int(positions.max().item()) + 1 - int(ids.numel()))

    return position_ids, torch.tensor(deltas, device=device, dtype=torch.long).unsqueeze(1)


def text_position_ids(seq_len: int, start_position: int, delta: int, device=None) -> torch.Tensor:
    """Position ids for the *new* tokens of an incremental decode step.

    During decode every new token is text, so all three MRoPE axes carry the same value:
    ``arange(start_position + delta, ... + seq_len)``.  ``start_position`` is the number of
    tokens already computed -- using ``0`` here instead would restart RoPE at the beginning
    of the sequence and silently corrupt every generated token.
    """
    ids = torch.arange(seq_len, device=device) + start_position + delta
    return ids.view(1, 1, -1).expand(3, 1, -1)


def text_position_ids_from_delta(seq_len: int, delta: int, device=None) -> torch.Tensor:
    """Backwards-compatible alias assuming ``start_position == 0``."""
    return text_position_ids(seq_len, 0, delta, device)


# --------------------------------------------------------------------------------------
# Vision 2D RoPE
# --------------------------------------------------------------------------------------


def vision_rotary_freqs(position_ids: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """``[N, 2]`` (h, w) position ids -> ``[N, rotary_dim]`` frequencies."""
    return (position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.to(torch.float32)).flatten(1)


def vision_cos_sin(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[N, 2]`` -> ``[N, head_dim]`` cosine/sine tables (the frequencies are duplicated)."""
    rot = vision_rotary_freqs(position_ids, inv_freq)
    emb = torch.cat((rot, rot), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def vision_rope_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Block-major ``(h, w)`` position ids for a batch of vision grids.

    Tokens are laid out as ``m x m`` merge blocks, so the ids must be permuted the same
    way the patch embeddings are reshaped in the vision tower.
    """
    device = grid_thw.device
    out: List[torch.Tensor] = []
    for t, h, w in grid_thw.tolist():
        h_ids, w_ids = torch.meshgrid(
            torch.arange(int(h), device=device), torch.arange(int(w), device=device), indexing="ij"
        )
        block = (int(h) // spatial_merge_size, spatial_merge_size, int(w) // spatial_merge_size, spatial_merge_size)
        h_ids = h_ids.reshape(block).transpose(1, 2).flatten()
        w_ids = w_ids.reshape(block).transpose(1, 2).flatten()
        out.append(torch.stack([h_ids, w_ids], dim=-1).repeat(int(t), 1))
    return torch.cat(out, dim=0)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """2D RoPE for the vision tower. ``q``/``k`` are ``[seq, heads, head_dim]``."""
    q_dtype, k_dtype = q.dtype, k.dtype
    qf, kf = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_out = (qf * cos) + (rotate_half(qf) * sin)
    k_out = (kf * cos) + (rotate_half(kf) * sin)
    return q_out.to(q_dtype), k_out.to(k_dtype)
