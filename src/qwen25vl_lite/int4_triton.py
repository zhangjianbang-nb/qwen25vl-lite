"""Fused INT4 groupwise dequantisation + GEMM, as Triton kernels.

The reference path materialises a dense fp16 copy of every weight on every decode step
(:meth:`QuantLinear.dense_weight`), then runs ``F.linear``.  Measured on an RTX 5060 with
Qwen2.5-VL-3B INT4/group-128, that dequantisation is ~92% of a decode step: a 2048x11008
projection costs 1.83 ms, of which 1.71 ms is the unpack.  A fused kernel reads the packed
nibbles once, applies the scale and zero, and feeds the GEMM directly, so the fp16 copy is
never written or read back.

Two kernels, chosen by row count:

* ``_int4_groupwise_gemv`` for a single activation row (the decode step).  ``tl.dot`` needs at
  least 16 rows, so the tiled kernel computes 16 and discards 15; on a memory-bound step a
  broadcast-multiply-and-reduce does only the useful work and is ~13% faster (8.4 -> 7.3 ms
  across one step's 252 projections).
* ``_int4_groupwise_gemm`` for anything wider (prefill), where the tensor cores are the right
  tool.

This module is **opt-in** and never changes the default path.  A fused kernel that is subtly
wrong produces plausible output, so it ships with :func:`verify_against_reference`, which is
exercised by the tests and can be run at load time (``--int4-kernel`` prints its worst error)
over row counts 1, 8 and 64, so both kernels are covered.  The reference path stays the
definition of correct.

Layout assumptions, all checked by :func:`can_use` before the kernel is selected:

* the weight is ``int4_groupwise`` with ``zeros`` present (asymmetric),
* ``group_size`` is one of 32/64/128/256 and divides the input width,
* there is no AWQ ``rescale`` vector,
* the activation is fp16 or bf16 on CUDA.

Anything else falls back to the reference path, so enabling the kernel cannot change the
result of a configuration it does not support.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "HAS_TRITON",
    "can_use",
    "fused_linear",
    "verify_against_reference",
    "SUPPORTED_GROUP_SIZES",
]

try:  # Triton is optional: the engine must import and run without it.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001 - any import failure means "no kernel available"
    HAS_TRITON = False


# Block widths are chosen so that BK == group_size: the group index is then constant inside a
# k-block, which lets scales and zeros load as a [BN] vector and broadcast instead of being
# gathered per element.  That change alone was worth ~8x on the prototype.
SUPPORTED_GROUP_SIZES = (32, 64, 128, 256)


if HAS_TRITON:

    @triton.jit
    def _int4_groupwise_gemm(
        x_ptr, w_ptr, s_ptr, z_ptr, bias_ptr, out_ptr,
        M, N, K,
        HAS_BIAS: tl.constexpr,
        G: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BN)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_b = tl.arange(0, BK // 2)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            x = tl.load(x_ptr + offs_m[:, None] * K + ks[None, :],
                        mask=offs_m[:, None] < M, other=0.0)

            # pack_int4 stores the even index in the low nibble and the odd index in the high.
            b = tl.load(w_ptr + offs_n[:, None] * (K // 2) + (k0 // 2 + offs_b)[None, :],
                        mask=offs_n[:, None] < N, other=0)
            b = b.to(tl.int32)
            q = tl.interleave(b & 0xF, (b >> 4) & 0xF)   # [BN, BK], even = low nibble

            g0 = k0 // G
            soff = offs_n * (K // G) + g0
            s = tl.load(s_ptr + soff, mask=offs_n < N, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + soff, mask=offs_n < N, other=0.0).to(tl.float32)

            w = ((q.to(tl.float32) - z[:, None]) * s[:, None]).to(x.dtype)
            acc += tl.dot(x, tl.trans(w))

        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :].to(tl.float32)

        offs_m2 = offs_m[:, None]
        offs_n2 = offs_n[None, :]
        tl.store(out_ptr + offs_m2 * N + offs_n2, acc.to(out_ptr.dtype.element_ty),
                 mask=(offs_m2 < M) & (offs_n2 < N))

    @triton.jit
    def _int4_groupwise_gemv(
        x_ptr, w_ptr, s_ptr, z_ptr, bias_ptr, out_ptr,
        N, K,
        HAS_BIAS: tl.constexpr,
        G: tl.constexpr,
        BN: tl.constexpr, BK: tl.constexpr,
    ):
        """Decode shape (M == 1): one row of activations, and no ``tl.dot``.

        ``tl.dot`` requires at least 16 rows, so the tiled kernel computes 16 and discards 15 --
        16x the useful work on a step that is memory-bound.  A broadcast-multiply-and-reduce
        does only the useful work, and measured on the whole step's 252 projections it is
        ~1.0 ms faster (8.4 -> 7.3 ms, 186 -> 202 GB/s).

        The scale and zero are applied once per output column instead of once per weight, using
        ``sum_k x_k*(q-z)*s == s*(sum_k x_k*q - z*sum_k x_k)``.  ``q`` is an integer in [0, 15],
        so ``sum_k x_k*q`` needs no rescaling of the packed value itself.
        """
        pid = tl.program_id(0)
        offs_n = pid * BN + tl.arange(0, BN)
        offs_b = tl.arange(0, BK // 2)

        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            x = tl.load(x_ptr + ks).to(tl.float32)
            b = tl.load(w_ptr + offs_n[:, None] * (K // 2) + (k0 // 2 + offs_b)[None, :],
                        mask=offs_n[:, None] < N, other=0)
            b = b.to(tl.int32)
            q = tl.interleave(b & 0xF, (b >> 4) & 0xF).to(tl.float32)

            g0 = k0 // G
            soff = offs_n * (K // G) + g0
            s = tl.load(s_ptr + soff, mask=offs_n < N, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + soff, mask=offs_n < N, other=0.0).to(tl.float32)

            acc += (tl.sum(q * x[None, :], axis=1) - z * tl.sum(x, axis=0)) * s

        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=offs_n < N)


def _tile_for(m: int, n: int) -> tuple:
    """(BM, BN, num_stages, num_warps) for a given row count and output width.

    Decode is M == 1, so the grid has only ``N / BN`` programs: a narrow BN is what keeps
    enough CTAs to saturate memory bandwidth.  BN=128 left the 256-row k/v projections with
    two CTAs and the 2048-row q/o projections with sixteen, which ran at ~105 GB/s.

    Sweeping BM/BN/stages/warps over the four shapes a decoder actually uses at M=1
    (11008x2048, 2048x11008, 2048x2048, 256x2048) gives a wide BN=32 with two stages for the
    wide FFN and a narrow BN=16 with three stages for the rest -- 1.2-1.4x per projection
    over one fixed policy, or 7.71 -> 5.95 ms per token across 36 layers.  Tile shape does
    not change the K-loop order, so this is a performance choice only.

    M == 1 does not use this: it takes the GEMV path below, which has no BM at all.
    """
    if m <= 16:
        if n >= 4096:
            return 16, 32, 2, 2
        if n <= 512:
            return 16, 16, 3, 4
        return 16, 16, 3, 2
    if m <= 64:
        return 64, 64, 2, 4
    return 128, 64, 2, 4


def _gemv_tile_for(n: int) -> tuple:
    """(BN, num_stages, num_warps) for the M == 1 GEMV path.

    Measured over a whole step's 252 projections, interleaved against the tiled kernel to
    cancel clock drift: BN=32/stages=3/warps=4 and BN=16/stages=3/warps=2 both land at ~7.3 ms
    (202 GB/s) against 8.4 ms (186 GB/s) for the tiled kernel.  The narrow BN is kept for the
    256-row k/v projections, which need the extra CTAs.
    """
    if n <= 512:
        return 16, 3, 2
    return 32, 3, 4


def can_use(x: torch.Tensor, qt) -> bool:
    """Whether the fused kernel supports this activation and quantised weight."""
    if not HAS_TRITON:
        return False
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if x.dim() < 2:
        return False
    if int(qt.bits) != 4:
        return False
    if qt.zeros is None or qt.rescale is not None:
        return False
    g = int(qt.group_size or 0)
    if g not in SUPPORTED_GROUP_SIZES:
        return False
    k = int(qt.shape[1])
    if k % g != 0:
        return False
    # The kernel indexes packed bytes with K // 2, so a zero-padded input width must not be
    # present; padded_in_features == k is the condition for that.
    if int(qt.meta.get("padded_in_features", k)) != k:
        return False
    if qt.data.dtype != torch.uint8 or not qt.data.is_contiguous():
        return False
    return True


def fused_linear(x: torch.Tensor, qt, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``F.linear(x, dequantize(qt), bias)`` without materialising the dense weight.

    ``x`` may be any shape ending in the input width; the leading dimensions are flattened
    and restored, matching ``F.linear``.

    A single activation row (the decode step) takes the GEMV kernel, which does not need
    ``tl.dot`` and therefore does not waste 15 of every 16 rows.  Anything wider -- prefill --
    takes the tiled kernel, where tensor cores are the right tool.
    """
    if not can_use(x, qt):
        raise ValueError("fused_linear called on an unsupported configuration; guard with can_use()")

    n, k = int(qt.shape[0]), int(qt.shape[1])
    x2 = x.reshape(-1, k).contiguous()
    m = x2.shape[0]
    bk = int(qt.group_size)
    bias_arg = bias if bias is not None else x2

    if m == 1:
        bn, stages, warps = _gemv_tile_for(n)
        out = torch.empty((1, n), device=x.device, dtype=x.dtype)
        _int4_groupwise_gemv[(triton.cdiv(n, bn),)](
            x2, qt.data, qt.scales, qt.zeros, bias_arg, out, n, k,
            HAS_BIAS=bias is not None,
            G=bk, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages,
        )
        return out.reshape(*x.shape[:-1], n)

    bm, bn, stages, warps = _tile_for(m, n)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _int4_groupwise_gemm[grid](
        x2, qt.data, qt.scales, qt.zeros, bias_arg, out, m, n, k,
        HAS_BIAS=bias is not None,
        G=bk, BM=bm, BN=bn, BK=bk,
        num_warps=warps, num_stages=stages,
    )
    return out.reshape(*x.shape[:-1], n)


def verify_against_reference(
    qt,
    x: Optional[torch.Tensor] = None,
    atol: float = 2e-2,
    dtype: torch.dtype = torch.float16,
) -> float:
    """Worst *relative* error between the fused kernel and the reference dequantise + GEMM.

    The reference path is the definition of correct; this is the self-check the design doc
    asks for before a fused kernel is trusted.

    The error is measured relative to the reference's own magnitude, because the reference is
    computed in ``dtype`` too.  bf16 carries 8 mantissa bits, so its own rounding noise is
    ~1e-2 of the output magnitude: an absolute threshold rejects a kernel whose difference is
    *smaller than one bf16 ulp*.  A genuinely wrong kernel -- a swapped nibble order, a wrong
    scale axis -- differs by order 1, so the relative form still catches it.  ``atol`` is
    therefore read as a relative bound.  Raises if the configuration is unsupported, so a
    silent fallback cannot masquerade as a passing check.

    With ``x=None`` the probe covers several row counts (1, 8, 64), because row count 1 takes
    the GEMV kernel and anything wider takes the tiled one; checking only one of them would
    leave the other unverified.  The probe input is seeded, so the startup decision is
    reproducible rather than a coin flip.
    """
    n, k = int(qt.shape[0]), int(qt.shape[1])

    if x is not None:
        probes = [x]
    else:
        gen = torch.Generator(device=str(qt.data.device)).manual_seed(0)
        probes = [torch.randn(m, k, device=qt.data.device, dtype=dtype, generator=gen)
                  for m in (1, 8, 64)]

    worst = 0.0
    for probe in probes:
        if not can_use(probe, qt):
            raise ValueError("verify_against_reference needs a configuration can_use() accepts")
        ref = torch.nn.functional.linear(probe, qt.dequantize(dtype, probe.device))
        got = fused_linear(probe, qt)
        err = (got.float() - ref.float()).abs()
        scale = ref.float().abs().max().clamp_min(1e-6)
        worst = max(worst, (err.max() / scale).item())
    return worst
