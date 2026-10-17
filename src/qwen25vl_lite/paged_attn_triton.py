"""Fused paged decode attention, as a Triton flash-decoding kernel.

The reference path gathers every layer's whole context into dense tensors and hands them to
SDPA with an additive length mask.  That is exact, but on a decode step (one query per
sequence) it is badly matched to the hardware: measured on an RTX 5060 with
Qwen2.5-VL-3B, one layer costs 50 us at a 512-token context and 183 us at 2,048 tokens,
against a memory bound of 5-22 us.  The gather is a second cost on top (0.15 ms at 512
tokens, 0.54 ms at 2,048, per step).

This kernel reads the paged cache directly: it walks the block table, skips every block past
``context_lens``, and runs an online softmax so no dense gather and no mask tensor exist at
all.  It is flash-decoding: the context is split across programs, each producing a partial
``(max, sum, accumulator)``, and a second small kernel combines the splits.

This module is **opt-in** and never changes the default path.  A wrong attention kernel
produces fluent output rather than an error, so it ships with
:func:`verify_against_reference`, which reproduces exactly what ``text.py`` does on the
reference path -- gather, build the length mask, SDPA -- and compares.  The reference stays
the definition of correct.

Guards, all checked by :func:`can_use` before the kernel is selected:

* decode only: exactly one query per sequence (chunked prefill keeps the reference path),
* ``head_dim`` is 64, 128 or 256 (it is a ``tl.arange`` width),
* ``block_size`` is a power of two,
* every sequence has a non-empty context,
* fp16 or bf16 on CUDA.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "HAS_TRITON",
    "can_use",
    "paged_decode_attention",
    "verify_against_reference",
    "set_paged_attn_kernel",
    "paged_attn_kernel_enabled",
]

try:  # Triton is optional: the engine must import and run without it.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001 - any import failure means "no kernel available"
    HAS_TRITON = False


SUPPORTED_HEAD_DIMS = (64, 128, 256)

_ENABLED = False


def set_paged_attn_kernel(enabled: bool) -> None:
    """Turn the fused decode-attention kernel on or off (opt-in, global like the int4 one)."""
    global _ENABLED
    _ENABLED = bool(enabled)


def paged_attn_kernel_enabled() -> bool:
    return _ENABLED and HAS_TRITON


if HAS_TRITON:

    @triton.jit
    def _paged_decode_attn(
        q_ptr, k_ptr, v_ptr, bt_ptr, ctx_ptr,
        acc_ptr, m_ptr, l_ptr,
        stride_qb, stride_qh,
        stride_kb, stride_bt,
        H, KVH, GROUPS, SPLITS,
        SCALE,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        split = pid % SPLITS
        bh = pid // SPLITS
        h = bh % H
        b = bh // H
        kv_h = h // GROUPS

        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(q_ptr + b * stride_qb + h * stride_qh + offs_d).to(tl.float32)

        ctx = tl.maximum(tl.load(ctx_ptr + b), 1)
        n_blocks = tl.cdiv(ctx, BLOCK_N)
        per = tl.cdiv(n_blocks, SPLITS)
        start = split * per
        end = tl.minimum(start + per, n_blocks)

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        offs_n = tl.arange(0, BLOCK_N)
        kv_stride = KVH * BLOCK_D
        for blk in range(start, end):
            block_id = tl.load(bt_ptr + b * stride_bt + blk)
            k = tl.load(k_ptr + block_id * stride_kb + offs_n[:, None] * kv_stride
                        + kv_h * BLOCK_D + offs_d[None, :]).to(tl.float32)
            v = tl.load(v_ptr + block_id * stride_kb + offs_n[:, None] * kv_stride
                        + kv_h * BLOCK_D + offs_d[None, :]).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1) * SCALE
            pos = blk * BLOCK_N + offs_n
            s = tl.where(pos < ctx, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        tl.store(acc_ptr + pid * BLOCK_D + offs_d, acc)
        tl.store(m_ptr + pid, m_i)
        tl.store(l_ptr + pid, l_i)

    @triton.jit
    def _combine_splits(
        acc_ptr, m_ptr, l_ptr, out_ptr,
        stride_ob,
        H, GROUPS, SPLITS,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)          # one program per (batch, head)
        h = pid % H
        b = pid // H
        offs_d = tl.arange(0, BLOCK_D)

        # Running max over the splits, then rescale each split's accumulator onto it.
        M = float("-inf")
        for i in range(SPLITS):
            M = tl.maximum(M, tl.load(m_ptr + (pid * SPLITS + i)))
        denom = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for i in range(SPLITS):
            mi = tl.load(m_ptr + (pid * SPLITS + i))
            li = tl.load(l_ptr + (pid * SPLITS + i))
            w = tl.exp(mi - M)
            denom += li * w
            acc += tl.load(acc_ptr + (pid * SPLITS + i) * BLOCK_D + offs_d) * w
        tl.store(out_ptr + b * stride_ob + h * BLOCK_D + offs_d, acc / denom)


def _split_count(batch: int, heads: int) -> int:
    """Enough programs to fill the card, as a power of two to limit recompiles."""
    want = 128
    s = max(1, (want + batch * heads - 1) // (batch * heads))
    return min(8, 1 << (s - 1).bit_length())


def can_use(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
            context_lens: torch.Tensor) -> bool:
    """Whether the fused decode-attention kernel supports this call.

    Every check here is static metadata.  Nothing may read a device value: this function is
    called inside a CUDA-graph capture, where a device-to-host copy (``context_lens.min()``)
    is illegal.  A zero-length context is handled in the kernel instead.
    """
    if not HAS_TRITON:
        return False
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.dim() != 4 or q.shape[2] != 1:
        return False
    d = int(q.shape[3])
    if d not in SUPPORTED_HEAD_DIMS:
        return False
    if k_cache.shape[3] != d or v_cache.shape[3] != d:
        return False
    bs = int(k_cache.shape[1])
    if bs < 1 or (bs & (bs - 1)) != 0:
        return False
    if int(k_cache.shape[2]) != int(v_cache.shape[2]):
        return False
    if int(q.shape[1]) % int(k_cache.shape[2]) != 0:
        return False
    if context_lens.numel() == 0:
        return False
    return True


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """``[batch, heads, 1, head_dim]`` queries against the paged cache, no gather, no mask.

    Semantics match the reference path in ``text.py`` exactly: query ``(b, h)`` attends to
    positions ``[0, context_lens[b])`` of sequence ``b`` and nothing else.

    ``q`` is used through its own strides rather than copied: the decode query is a transposed
    view whose head dimension is contiguous, which is all the kernel needs.  On a CPU-bound
    step the two allocations here and the strides matter -- a per-layer ``contiguous()`` copy
    and four separate scratch tensors cost more than the kernel saves.
    """
    if not can_use(q, k_cache, v_cache, context_lens):
        raise ValueError("paged_decode_attention called on an unsupported configuration")

    batch, heads, _, head_dim = q.shape
    kv_heads = int(k_cache.shape[2])
    block_size = int(k_cache.shape[1])
    groups = heads // kv_heads
    splits = _split_count(batch, heads)
    total = batch * heads * splits

    # One allocation for all three partial buffers: (acc, m, l) laid out back to back.
    scratch = torch.empty(total * head_dim + 2 * total, device=q.device, dtype=torch.float32)
    acc = scratch[: total * head_dim].view(total, head_dim)
    m = scratch[total * head_dim: total * head_dim + total]
    l = scratch[total * head_dim + total:]
    out = torch.empty(batch, heads, head_dim, device=q.device, dtype=q.dtype)

    _paged_decode_attn[(total,)](
        q, k_cache, v_cache, block_tables, context_lens,
        acc, m, l,
        q.stride(0), q.stride(1),
        k_cache.stride(0), block_tables.stride(0),
        heads, kv_heads, groups, splits,
        float(scale),
        BLOCK_N=block_size, BLOCK_D=head_dim,
        num_warps=4, num_stages=3,
    )
    _combine_splits[(batch * heads,)](
        acc, m, l, out,
        out.stride(0),
        heads, groups, splits,
        BLOCK_D=head_dim,
        num_warps=4,
    )
    return out.unsqueeze(2)


def verify_against_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    scale: Optional[float] = None,
    atol: float = 2e-2,
) -> float:
    """Worst *relative* error between the fused kernel and the path ``text.py`` runs eagerly.

    The reference is rebuilt here the same way the eager path builds it -- gather the paged
    KV, build the length mask, run SDPA -- rather than by calling
    :func:`attention.paged_attention`, whose decode branch passes no mask and would therefore
    attend to padded slots this kernel deliberately skips.
    """
    from .attention import build_decode_mask, gather_paged_kv, sdpa

    if not can_use(q, k_cache, v_cache, context_lens):
        raise ValueError("verify_against_reference needs a configuration can_use() accepts")

    heads, head_dim = int(q.shape[1]), int(q.shape[3])
    groups = heads // int(k_cache.shape[2])
    scale = scale if scale is not None else head_dim ** -0.5
    block_size = int(k_cache.shape[1])

    k, v = gather_paged_kv(k_cache, v_cache, block_tables, context_lens, block_size,
                           max_len=block_tables.shape[1] * block_size)
    mask = build_decode_mask(context_lens, q.device, q.dtype,
                             max_len=block_tables.shape[1] * block_size)
    ref = sdpa(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True, num_kv_groups=groups)
    got = paged_decode_attention(q, k_cache, v_cache, block_tables, context_lens, scale=scale)

    err = (got.float() - ref.float()).abs()
    mag = ref.float().abs().max().clamp_min(1e-6)
    return (err.max() / mag).item()
