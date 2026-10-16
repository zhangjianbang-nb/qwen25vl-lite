"""Attention backends, runtime probing, and a reference paged-attention path.

The engine never *assumes* which SDPA backend will be selected -- the review of
2026-10-15 was explicit that SM version alone does not determine this.  Instead
:func:`probe_backends` actually runs a small SDPA call inside each backend's context
manager and records whether it succeeded, and the chosen backend is reported to the user
and written to logs.  ``--attn-backend`` pins a backend when you want reproducible numbers.

The paged path is implemented twice: a gather-based reference that is exact and runs
anywhere (used in tests and on CPU), and an optional compiled kernel (see ``native.py``).
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "BackendReport",
    "probe_backends",
    "pick_backends",
    "sdpa",
    "repeat_kv",
    "gather_paged_kv",
    "paged_attention",
    "build_chunk_mask",
    "build_decode_mask",
    "attention_with_probs",
]


def _sdpa_kernel_ctx(backends: Sequence):
    """Context manager pinning SDPA to ``backends`` (torch>=2.1 API)."""
    return torch.nn.attention.sdpa_kernel(list(backends))


def _backend_enum(name: str):
    B = torch.nn.attention.SDPBackend
    return {
        "flash": B.FLASH_ATTENTION,
        "mem_efficient": B.EFFICIENT_ATTENTION,
        "math": B.MATH,
        "cudnn": B.CUDNN_ATTENTION,
    }[name]


@dataclass
class BackendReport:
    """Result of an empirical SDPA backend probe."""

    device: str
    dtype: str
    head_dim: int
    probed: Dict[str, str] = field(default_factory=dict)
    chosen: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        ok = [k for k, v in self.probed.items() if v == "ok"]
        bad = {k: v for k, v in self.probed.items() if v != "ok"}
        lines = [
            f"SDPA probe on {self.device}/{self.dtype}, head_dim={self.head_dim}",
            f"  available: {', '.join(ok) if ok else '(none)'}",
        ]
        for k, v in bad.items():
            lines.append(f"  unavailable: {k} -> {v}")
        lines.append(f"  preference order: {' > '.join(self.chosen)}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def probe_backends(
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    head_dim: int = 128,
    seq_len: int = 64,
    num_heads: int = 4,
    num_kv_heads: int = 2,
) -> BackendReport:
    """Empirically determine which SDPA backends work for this shape/dtype/device.

    Each backend is exercised with a real ``scaled_dot_product_attention`` call.  A
    backend that raises (or is absent on this build) is recorded with its error message
    rather than silently assumed unavailable.
    """
    report = BackendReport(device=device, dtype=str(dtype).replace("torch.", ""), head_dim=head_dim)
    q = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

    candidates = ("math",) if device == "cpu" else ("flash", "mem_efficient", "cudnn", "math")
    if device == "cpu":
        report.notes.append(
            "CPU builds only implement the math kernel; the fused backends are reported as "
            "'not applicable' rather than 'available', because torch's context manager can "
            "silently ignore them on this device."
        )
    for name in ("flash", "mem_efficient", "cudnn", "math"):
        if name not in candidates:
            report.probed[name] = "not applicable on this device"
            continue
        try:
            with _sdpa_kernel_ctx([_backend_enum(name)]):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(num_heads != num_kv_heads))
            if not torch.isfinite(out).all():
                report.probed[name] = "ok but produced non-finite output"
            else:
                report.probed[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - we want the reason verbatim
            report.probed[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"

    report.chosen = pick_backends(report.probed, device)
    if head_dim % 8 != 0:
        report.notes.append(
            f"head_dim={head_dim} is not a multiple of 8; fused kernels commonly reject such shapes, "
            "so the math fallback is expected to be selected."
        )
    if num_heads != num_kv_heads and report.probed.get("mem_efficient", "").startswith("TypeError"):
        report.notes.append("mem_efficient rejected GQA on this torch build; the engine will repeat K/V instead.")
    return report


def pick_backends(probed: Dict[str, str], device: str) -> List[str]:
    """Ordered preference list restricted to backends that actually worked."""
    if device == "cpu":
        order = ["math"]
    else:
        # Flash first, then the fused memory-efficient kernel, then cuDNN, then math.
        order = ["flash", "mem_efficient", "cudnn", "math"]
    chosen = [b for b in order if probed.get(b) == "ok"]
    return chosen or ["math"]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[batch, kv_heads, seq, head_dim]`` -> ``[batch, kv_heads*n_rep, seq, head_dim]``."""
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backends: Optional[Sequence[str]] = None,
    enable_gqa: bool = True,
    num_kv_groups: int = 1,
) -> torch.Tensor:
    """Scaled dot-product attention with an explicitly chosen backend list.

    Falls back to repeating K/V when the selected backends reject the GQA layout, so a
    backend change can never silently produce a shape error.
    """
    kw = {"is_causal": is_causal} if attn_mask is None else {"attn_mask": attn_mask}
    if scale is not None:
        kw["scale"] = scale

    attempts: List[dict] = []
    if enable_gqa and num_kv_groups > 1:
        attempts.append({**kw, "enable_gqa": True})
    attempts.append(kw)
    attempts.append({**kw, "_kv_repeated": True})

    last_exc: Optional[Exception] = None
    for attempt in attempts:
        kk, vv = k, v
        if attempt.pop("_kv_repeated", False):
            kk, vv = repeat_kv(k, num_kv_groups), repeat_kv(v, num_kv_groups)
        try:
            ctx = _sdpa_kernel_ctx([_backend_enum(b) for b in backends]) if backends else contextlib.nullcontext()
            with ctx:
                return F.scaled_dot_product_attention(q, kk, vv, **attempt)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    raise RuntimeError(f"all SDPA attempts failed; last error: {last_exc}")


# --------------------------------------------------------------------------------------
# Paged KV
# --------------------------------------------------------------------------------------


def gather_paged_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Gather paged KV into dense ``[batch, kv_heads, max_len, head_dim]`` tensors.

    ``k_cache``/``v_cache`` are ``[num_blocks, block_size, kv_heads, head_dim]``;
    ``block_tables`` is ``[batch, max_blocks]`` of physical block ids (``-1`` = unused).
    """
    batch, max_blocks = block_tables.shape
    kv_heads, head_dim = k_cache.shape[2], k_cache.shape[3]
    max_len = int(context_lens.max().item())
    max_len = max(max_len, 1)

    k_out = k_cache.new_zeros(batch, kv_heads, max_len, head_dim)
    v_out = v_cache.new_zeros(batch, kv_heads, max_len, head_dim)
    for b in range(batch):
        n = int(context_lens[b].item())
        if n == 0:
            continue
        nblocks = (n + block_size - 1) // block_size
        ids = block_tables[b, :nblocks].tolist()
        if any(i < 0 for i in ids):
            raise ValueError(f"sequence {b} references an unallocated block: {ids}")
        kb = k_cache[ids].reshape(-1, kv_heads, head_dim)[:n]
        vb = v_cache[ids].reshape(-1, kv_heads, head_dim)[:n]
        k_out[b, :, :n] = kb.transpose(0, 1)
        v_out[b, :, :n] = vb.transpose(0, 1)
    return k_out, v_out


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    *,
    scale: Optional[float] = None,
    num_kv_groups: int = 1,
    backends: Optional[Sequence[str]] = None,
    query_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference paged attention (decode: 1 query per sequence; prefill: many).

    Exact and dependency-free.  ``q`` is ``[batch, heads, q_len, head_dim]``; keys/values
    are addressed through ``block_tables``.  When ``query_lens > 1`` the queries are the
    *tail* of the context, so the mask is causal with an offset of
    ``context_len - query_len``.
    """
    k, v = gather_paged_kv(k_cache, v_cache, block_tables, context_lens, block_size)
    heads, head_dim = q.shape[1], q.shape[3]

    if query_lens is None:
        # Every query attends to the whole context: no causal mask needed.
        return sdpa(
            q, k, v, scale=scale, backends=backends, enable_gqa=True, num_kv_groups=num_kv_groups
        )

    out = q.new_zeros(q.shape)
    max_ctx = k.shape[2]
    for b in range(q.shape[0]):
        n = int(query_lens[b].item())
        ctx = int(context_lens[b].item())
        if n == 0:
            continue
        qi = q[b : b + 1, :, :n, :]
        ki = k[b : b + 1, :, :ctx, :]
        vi = v[b : b + 1, :, :ctx, :]
        offset = ctx - n
        mask = torch.full((n, ctx), float("-inf"), device=q.device, dtype=q.dtype)
        for i in range(n):
            mask[i, : offset + i + 1] = 0.0
        out[b : b + 1, :, :n, :] = sdpa(
            qi,
            ki,
            vi,
            attn_mask=mask,
            scale=scale,
            backends=backends,
            enable_gqa=True,
            num_kv_groups=num_kv_groups,
        )
    return out


def build_chunk_mask(
    query_lens: torch.Tensor,
    context_lens: torch.Tensor,
    device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Causal mask for a chunked-prefill batch with heterogeneous context lengths.

    Returns ``[batch, 1, max_q, max_ctx]`` of additive ``0 / -inf`` values.  The offset for
    sequence ``b`` is ``context_len - query_len``, which is what makes the *absolute*
    position of a chunked query correct rather than restarting at 0.
    """
    batch = int(query_lens.shape[0])
    max_q = int(query_lens.max().item())
    max_ctx = int(context_lens.max().item())
    mask = torch.full((batch, 1, max_q, max_ctx), float("-inf"), device=device, dtype=dtype)
    for b in range(batch):
        n = int(query_lens[b].item())
        ctx = int(context_lens[b].item())
        offset = ctx - n
        for i in range(n):
            mask[b, 0, i, : offset + i + 1] = 0.0
    return mask


def build_decode_mask(
    context_lens: torch.Tensor,
    device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Length mask for a batched decode step.

    Every sequence has exactly one query, but the gathered KV is padded to the longest
    context in the batch.  Without this mask the short sequences would attend to zero-filled
    slots belonging to other sequences -- a wrong answer that still has the right shape.
    """
    batch = int(context_lens.shape[0])
    max_ctx = max(int(context_lens.max().item()), 1)
    mask = torch.full((batch, 1, 1, max_ctx), float("-inf"), device=device, dtype=dtype)
    for b in range(batch):
        mask[b, 0, 0, : int(context_lens[b].item())] = 0.0
    return mask


def attention_with_probs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    num_kv_groups: int = 1,
    query_offset: int = 0,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Explicit softmax attention, returning ``(output, probabilities)``.

    Only used where the weights themselves are the point (FastV-style pruning measures the
    attention share received by each image token).  Never on the hot path: materialising
    ``[batch, heads, q, k]`` is what the fused kernels exist to avoid.
    """
    head_dim = q.shape[-1]
    scale = scale if scale is not None else head_dim ** -0.5
    kk = repeat_kv(k, num_kv_groups) if num_kv_groups > 1 else k
    vv = repeat_kv(v, num_kv_groups) if num_kv_groups > 1 else v
    scores = torch.matmul(q.float(), kk.float().transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask
    elif scores.shape[-1] > scores.shape[-2]:
        # Causal with an offset: query i (absolute position query_offset + i) may attend to
        # keys 0..query_offset+i.
        n, ctx = scores.shape[-2], scores.shape[-1]
        idx = torch.arange(n, device=q.device).view(-1, 1) + query_offset
        allowed = torch.arange(ctx, device=q.device).view(1, -1) <= idx
        scores = scores.masked_fill(~allowed, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vv.float()).to(q.dtype), probs.to(q.dtype)

