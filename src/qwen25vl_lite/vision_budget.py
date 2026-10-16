"""Vision token budgeting and reduction.

The external review of 2026-10-15 was blunt about this area: token pruning is **not**
lossless on Qwen2.5-VL, and FastV cannot be treated as a plug-in optimisation because the
vision tower mixes window attention (``window_size=112``) with global attention at
``fullatt_block_indexes``.  So the tiers below are defined as *strategies*, not as quality
promises, and the mechanisms are ordered by how safe they are:

1. **Resolution budget** (always safe): the image is resized so it yields at most
   ``max_vision_tokens`` language-model tokens.  This changes the input, not the model.
2. **Grid-preserving pooling** (structure-safe): average ``r x r`` neighbouring merge
   units.  The result is still a rectangular grid, so MRoPE position ids stay valid.
3. **ToME merge** (content-based, approximate): bipartite soft matching merges tokens by
   feature similarity.  Survivors no longer form a grid, so this path *returns explicit
   position ids* and callers must use them instead of ``get_rope_index``.
4. **FastV-style pruning** (approximate, opt-in, off by default): image tokens whose
   attention share at a chosen decoder layer is low are dropped for the remaining layers.
   Quality is task-dependent (OCR, small objects, dense scenes and counting are the known
   weak spots) and must be validated per task; see ``scripts/quality_gate.py``.

Every reduction returns the new token count so the caller can emit exactly that many
``image_token_id`` placeholders -- getting this wrong is a shape error, not a silent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .imageproc import ImageBudget, max_pixels_for_tokens, vision_tokens

__all__ = [
    "VisionTier",
    "TIERS",
    "tier_budget",
    "pool_merge",
    "tome_merge",
    "fastv_select",
    "apply_fastv",
    "summarise_tiers",
]


@dataclass
class VisionTier:
    """A named vision-processing strategy.

    Each tier applies **exactly one** lever, so a quality change can be attributed to it:

    * ``max_vision_tokens`` -- the resolution ceiling.  This is the primary lever, because
      vision tokens are what prefill time and KV pressure are actually made of.
    * ``fastv_ratio`` -- the fraction of image tokens to *drop* after ``fastv_layer`` of the
      language model, using attention received as the ranking signal.  ``0`` disables it.

    ``pool_merge`` and ``tome_merge`` are implemented in this module and are available to
    callers, but they are deliberately **not** wired into a tier.  Both replace trained
    feature geometry (a 2x2 pool averages merge units the checkpoint never averaged; ToME
    fuses tokens by similarity), so their quality cost is model-specific and has to be
    measured per model rather than assumed.  Putting one inside a named tier would hide that
    measurement behind a label.
    """

    name: str
    max_vision_tokens: int
    fastv_ratio: float = 0.0
    fastv_layer: int = 2
    description: str = ""

    def validate(self) -> None:
        if self.max_vision_tokens <= 0:
            raise ValueError("max_vision_tokens must be positive")
        if not 0.0 <= self.fastv_ratio < 1.0:
            raise ValueError("fastv_ratio must be in [0, 1)")


# Default budgets are expressed in vision tokens (post-merge), which is what actually costs
# prefill time.  A 1M-pixel image with patch 14 and merge 2 yields ~1275 tokens.
TIERS: Dict[str, VisionTier] = {
    "full": VisionTier(
        name="full",
        max_vision_tokens=1280,
        fastv_ratio=0.0,
        description="full dynamic resolution; the quality reference",
    ),
    "half": VisionTier(
        name="half",
        max_vision_tokens=640,
        fastv_ratio=0.0,
        description="half the vision-token ceiling (640 vs 1280); resolution is the only lever",
    ),
    "thumbnail": VisionTier(
        name="thumbnail",
        max_vision_tokens=256,
        fastv_ratio=0.0,
        description="a 256-token ceiling; small text and fine detail are expected to suffer",
    ),
    "pruned": VisionTier(
        name="pruned",
        max_vision_tokens=1280,
        fastv_ratio=0.5,
        fastv_layer=2,
        description="full resolution, then FastV drops the 50% lowest-attention image tokens after layer 2",
    ),
}


def tier_budget(
    tier: VisionTier,
    patch_size: int = 14,
    merge_size: int = 2,
    min_pixels: Optional[int] = None,
) -> ImageBudget:
    """Pixel budget for a tier, expressed so the *token* ceiling is what binds."""
    budget = ImageBudget.from_tokens(
        tier.max_vision_tokens,
        patch_size=patch_size,
        merge_size=merge_size,
        min_pixels=min_pixels if min_pixels is not None else 4 * patch_size * merge_size * patch_size * merge_size // 4,
    )
    if budget.max_pixels < budget.min_pixels:
        budget.max_pixels = budget.min_pixels
    return budget


def pool_merge(
    features: torch.Tensor,
    grid_thw: torch.Tensor,
    factor: int,
    merge_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average-pool ``factor x factor`` neighbouring merge units.

    ``features`` is ``[total_tokens, dim]`` in the order produced by the vision tower
    (unit-major), and ``grid_thw`` is in *patch* units.  Returns the pooled features and the
    new grid, which is again a valid rectangular grid -- that is the whole point of using
    pooling instead of content-based merging: MRoPE position ids stay correct without any
    extra bookkeeping.
    """
    if factor <= 1:
        return features, grid_thw
    outs: List[torch.Tensor] = []
    new_grids: List[List[int]] = []
    offset = 0
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        gh, gw = h // merge_size, w // merge_size
        n_units = t * gh * gw
        block = features[offset : offset + n_units]
        offset += n_units
        if gh % factor or gw % factor:
            raise ValueError(
                f"grid {gh}x{gw} merge units is not divisible by pool factor {factor}; "
                "choose a factor that divides both, or fall back to a resolution budget"
            )
        block = block.reshape(t, gh // factor, factor, gw // factor, factor, -1)
        pooled = block.mean(dim=(2, 4)).reshape(-1, features.shape[-1])
        outs.append(pooled)
        new_grids.append([t, (gh // factor) * merge_size, (gw // factor) * merge_size])
    return torch.cat(outs, dim=0), torch.tensor(new_grids, dtype=torch.long)


def _unit_positions(grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
    """``[total_units, 2]`` (h, w) coordinates of each merge unit, unit-major order."""
    out: List[torch.Tensor] = []
    for t, h, w in grid_thw.tolist():
        gh, gw = int(h) // merge_size, int(w) // merge_size
        rows, cols = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        pos = torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=-1)
        out.append(pos.repeat(int(t), 1))
    return torch.cat(out, dim=0)


def tome_merge(
    features: torch.Tensor,
    grid_thw: torch.Tensor,
    r: int,
    merge_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ToME-style bipartite soft matching: merge up to ``r`` units into one.

    Unlike :func:`pool_merge` this is *content based*: partners are chosen by feature
    similarity, not by position.  Consequences the caller must respect:

    * the survivors are no longer a rectangular grid, so the returned ``position_ids``
      (``[n_survivors, 2]``) replace the grid-derived ones -- MRoPE needs explicit ids here;
    * survivors are returned in ascending source order, so the caller can still map each
      survivor back to the token id it replaces.

    Returns ``(merged_features, position_ids, group_sizes)``.  ``group_sizes`` is the number
    of tokens each survivor absorbed; ToME's "proportional attention" divides the attention
    logit by ``log(group_size)``, which the caller applies if it wants that behaviour.

    This is an *approximation*: it is content-dependent, so two images with the same token
    count can lose different amounts of detail.  Validate it on your task before trusting it.
    """
    pos = _unit_positions(grid_thw, merge_size)
    n = features.shape[0]
    if r <= 1 or n <= r:
        return features, pos, torch.ones(n, dtype=torch.long)

    x = features.float()
    idx_a = torch.arange(0, n, 2)
    idx_b = torch.arange(1, n, 2)
    sim = torch.nn.functional.normalize(x[idx_b], dim=-1) @ torch.nn.functional.normalize(x[idx_a], dim=-1).t()
    best = sim.argmax(dim=-1)
    best_sim = sim.gather(1, best[:, None]).squeeze(1)

    # Every token starts as its own singleton group; merging only ever *fuses* groups, so no
    # token can be dropped.  (Dropping unmatched tokens would silently shrink the sequence.)
    sizes = torch.ones(n)
    dest = torch.arange(n)
    # Greedily fuse the most similar pairs first, respecting the cap of `r` per group.
    for j in torch.argsort(best_sim, descending=True).tolist():
        b_idx = int(idx_b[j])
        rep = int(idx_a[int(best[j])])
        if sizes[rep] + sizes[b_idx] > r:
            continue
        dest[b_idx] = rep
        sizes[rep] += sizes[b_idx]
        sizes[b_idx] = 0.0

    survivors = torch.nonzero(sizes > 0).flatten()
    merged = torch.stack([x[dest == int(s)].mean(dim=0) for s in survivors.tolist()], dim=0)
    return merged.to(features.dtype), pos[survivors], sizes[survivors].to(torch.long)


def fastv_select(
    probs: torch.Tensor,
    image_positions: Sequence[int],
    ratio: float,
) -> torch.Tensor:
    """Indices of image tokens to **keep** according to their attention share.

    ``probs`` is an attention-probability tensor ``[heads, q_len, k_len]`` (or with a leading
    batch dim of 1).  Following FastV, an image token's importance is the attention it
    receives from all query positions; tokens below the ``ratio`` quantile are dropped.
    """
    if probs.dim() == 4:
        probs = probs[0]
    if ratio <= 0:
        return torch.tensor(list(image_positions), dtype=torch.long)
    if ratio >= 1:
        raise ValueError("fastv ratio must be < 1; dropping every image token is never useful")

    share = probs.mean(dim=0).sum(dim=0)  # average over heads, sum over queries -> [k_len]
    img = torch.tensor(list(image_positions), dtype=torch.long)
    scores = share[img]
    n_keep = max(1, int(round(img.numel() * (1.0 - ratio))))
    order = torch.argsort(scores, descending=True)
    keep_sorted = img[order[:n_keep]]
    return torch.sort(keep_sorted).values


def apply_fastv(
    token_ids: Sequence[int],
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    image_positions: Sequence[int],
    keep_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
    """Drop the non-kept image tokens from a *dense* prefill.

    Returns ``(hidden, position_ids, new_token_ids, keep_mask)``.  Text tokens are always
    kept; only image tokens can be removed.  The caller is responsible for writing KV only
    for the retained slots, which is what keeps the paged cache consistent with the shorter
    sequence.
    """
    n = len(token_ids)
    keep_mask = torch.ones(n, dtype=torch.bool)
    drop = set(int(i) for i in image_positions) - set(int(i) for i in keep_indices)
    for i in drop:
        keep_mask[i] = False
    new_ids = [t for t, k in zip(token_ids, keep_mask.tolist()) if k]
    # position_ids is [3, batch, seq]; index the *last* dim so the batch axis survives.
    return hidden[keep_mask], position_ids[..., keep_mask], new_ids, keep_mask


def summarise_tiers(tiers: Optional[Dict[str, VisionTier]] = None) -> str:
    """Human-readable table used by the CLI."""
    tiers = tiers or TIERS
    lines = [f"{'tier':<10} {'max_vision_tokens':>18} {'fastv_ratio':>11}  description"]
    for tier in tiers.values():
        lines.append(
            f"{tier.name:<10} {tier.max_vision_tokens:>18} {tier.fastv_ratio:>11}  {tier.description}"
        )
    return "\n".join(lines)
