"""Sampling: penalties, top-k/top-p, and reproducible per-sequence RNG.

Kept separate from the scheduler so the numerics can be tested without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

__all__ = ["SamplingParams", "apply_penalties", "sample_token", "sample_batch"]


@dataclass
class SamplingParams:
    """Decoding configuration.  ``temperature == 0`` means greedy."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 512
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Sequence[str] = field(default_factory=tuple)
    stop_token_ids: Sequence[int] = field(default_factory=tuple)
    ignore_eos: bool = False

    def validate(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


def apply_penalties(logits: torch.Tensor, prev_ids: Sequence[int], params: SamplingParams) -> torch.Tensor:
    """Apply repetition / presence / frequency penalties in place on a copy."""
    if not prev_ids:
        return logits
    if params.repetition_penalty == 1.0 and params.presence_penalty == 0.0 and params.frequency_penalty == 0.0:
        return logits

    out = logits.clone()
    ids = torch.tensor(list(prev_ids), dtype=torch.long, device=logits.device)
    counts = torch.bincount(ids, minlength=logits.shape[-1]).to(logits.dtype)

    if params.repetition_penalty != 1.0:
        # Positive logits are divided, negative ones multiplied (the CTRL convention).
        seen = counts > 0
        pos = out[seen] > 0
        out[seen] = torch.where(pos, out[seen] / params.repetition_penalty, out[seen] * params.repetition_penalty)
    if params.presence_penalty:
        out = out - params.presence_penalty * (counts > 0).to(logits.dtype)
    if params.frequency_penalty:
        out = out - params.frequency_penalty * counts
    return out


def sample_token(logits: torch.Tensor, params: SamplingParams, generator: Optional[torch.Generator] = None) -> int:
    """Sample one token id from a ``[vocab]`` logit vector."""
    if params.temperature <= 0:
        return int(torch.argmax(logits).item())

    scores = logits.float() / params.temperature

    if params.top_k and params.top_k > 0:
        k = min(params.top_k, scores.numel())
        kth = torch.topk(scores, k).values[-1]
        scores = torch.where(scores < kth, torch.full_like(scores, float("-inf")), scores)

    if params.min_p > 0:
        probs = torch.softmax(scores, dim=-1)
        cutoff = params.min_p * probs.max()
        scores = torch.where(probs < cutoff, torch.full_like(scores, float("-inf")), scores)

    if params.top_p < 1.0:
        sorted_scores, order = torch.sort(scores, descending=True)
        probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # Keep the smallest prefix whose cumulative mass reaches top_p (always at least one).
        keep = cumulative <= params.top_p
        keep[0] = True
        if not bool(keep.any()):
            keep[0] = True
        filtered = torch.full_like(sorted_scores, float("-inf"))
        filtered[keep] = sorted_scores[keep]
        scores = torch.empty_like(scores).scatter_(0, order, filtered)

    probs = torch.softmax(scores, dim=-1)
    if generator is not None:
        return int(torch.multinomial(probs, 1, generator=generator).item())
    return int(torch.multinomial(probs, 1).item())


def sample_batch(
    logits: torch.Tensor,
    params_list: Sequence[SamplingParams],
    prev_ids: Sequence[Sequence[int]],
    generators: Optional[Sequence[Optional[torch.Generator]]] = None,
) -> List[int]:
    """Sample one token for each row of a ``[batch, vocab]`` logit matrix."""
    out: List[int] = []
    for i, params in enumerate(params_list):
        lg = apply_penalties(logits[i], prev_ids[i], params)
        gen = generators[i] if generators else None
        out.append(sample_token(lg, params, gen))
    return out
