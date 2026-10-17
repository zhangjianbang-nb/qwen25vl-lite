"""The fused paged decode-attention kernel: opt-in, and numerically equal to the eager path.

The eager path in ``text.py`` -- gather the paged KV, build the length mask, run SDPA -- is
the definition of correct.  These tests run the guards on CPU and, when a CUDA device and
Triton are present, the kernel itself against that reference.
"""

from __future__ import annotations

import pytest
import torch

from qwen25vl_lite import paged_attn_triton
from qwen25vl_lite.paged_attn_triton import (
    can_use,
    paged_attn_kernel_enabled,
    paged_decode_attention,
    set_paged_attn_kernel,
    verify_against_reference,
)

NEEDS_GPU = not (torch.cuda.is_available() and paged_attn_triton.HAS_TRITON)
GPU_REASON = "needs CUDA and Triton"


@pytest.fixture(autouse=True)
def _reset_kernel_flag():
    """The flag is process-global; never let one test's choice leak into the next."""
    set_paged_attn_kernel(False)
    yield
    set_paged_attn_kernel(False)


def _case(batch, heads, kv_heads, head_dim, block_size, lens, dtype, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    max_blocks = max((n + block_size - 1) // block_size for n in lens)
    num_blocks = max(max_blocks + 4, 8)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k_cache = torch.randn(num_blocks, block_size, kv_heads, head_dim, device=device,
                          dtype=dtype, generator=torch.Generator(device=device).manual_seed(seed))
    v_cache = torch.randn(num_blocks, block_size, kv_heads, head_dim, device=device,
                          dtype=dtype, generator=torch.Generator(device=device).manual_seed(seed + 1))
    block_tables = torch.full((batch, max_blocks), -1, dtype=torch.long, device=device)
    for b in range(batch):
        n_blocks = (lens[b] + block_size - 1) // block_size
        ids = torch.randperm(num_blocks, generator=gen)[:n_blocks]
        block_tables[b, :n_blocks] = ids.to(device)
    context_lens = torch.tensor(lens, dtype=torch.long, device=device)
    q = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype,
                    generator=torch.Generator(device=device).manual_seed(seed + 2))
    return q, k_cache, v_cache, block_tables, context_lens


def test_kernel_is_off_by_default():
    assert paged_attn_kernel_enabled() is False


def test_can_use_rejects_a_cpu_query():
    q = torch.zeros(1, 4, 1, 128, dtype=torch.float16)
    k = torch.zeros(8, 16, 2, 128, dtype=torch.float16)
    ctx = torch.tensor([17])
    assert can_use(q, k, k, ctx) is False


@pytest.mark.skipif(NEEDS_GPU, reason=GPU_REASON)
def test_can_use_rejects_unsupported_shapes():
    q, k, v, bt, ctx = _case(1, 28, 2, 128, 16, [64], torch.bfloat16)
    assert can_use(q, k, v, ctx) is True

    # A non-power-of-two block size is not a tl.arange width.
    assert can_use(q, torch.zeros(8, 24, 2, 128, device="cuda", dtype=torch.bfloat16), v, ctx) is False
    # head_dim outside {64, 128, 256}.
    assert can_use(torch.zeros(1, 28, 1, 96, device="cuda", dtype=torch.bfloat16), k, v, ctx) is False
    # fp32 activations: the kernel is fp16/bf16 only.
    assert can_use(q.float(), k, v, ctx) is False
    # More than one query per sequence is prefill, which the reference path covers.
    assert can_use(torch.zeros(1, 28, 4, 128, device="cuda", dtype=torch.bfloat16), k, v, ctx) is False


@pytest.mark.skipif(NEEDS_GPU, reason=GPU_REASON)
@pytest.mark.parametrize("case", [
    (1, 28, 2, 128, 16, [512], torch.bfloat16),
    (1, 28, 2, 128, 16, [2048], torch.bfloat16),
    (1, 28, 2, 128, 16, [17], torch.bfloat16),
    (4, 28, 2, 128, 16, [33, 512, 2000, 15], torch.bfloat16),
    (1, 4, 2, 128, 16, [512], torch.float16),
    (1, 16, 2, 64, 16, [300], torch.bfloat16),
    (1, 28, 2, 128, 32, [600], torch.bfloat16),
    (2, 28, 2, 256, 16, [900, 100], torch.bfloat16),
])
def test_fused_matches_the_eager_path(case):
    """The worst relative error must stay at bf16 rounding noise, not at order 1."""
    batch, heads, kv_heads, head_dim, block_size, lens, dtype = case
    q, k_cache, v_cache, block_tables, context_lens = _case(
        batch, heads, kv_heads, head_dim, block_size, lens, dtype)
    err = verify_against_reference(q, k_cache, v_cache, block_tables, context_lens)
    assert err < 2e-2, f"{case}: worst relative error {err:.3e}"


@pytest.mark.skipif(NEEDS_GPU, reason=GPU_REASON)
def test_padded_block_table_tail_does_not_change_the_result():
    """The engine pads a bucketed block table with -1; the kernel must never read past context."""
    q, k_cache, v_cache, block_tables, context_lens = _case(1, 28, 2, 128, 16, [500], torch.bfloat16)
    scale = 128 ** -0.5
    exact = paged_decode_attention(q, k_cache, v_cache, block_tables, context_lens, scale=scale)

    # Widen the table with unusable ids: they address a real block, so a kernel that walked
    # past context_lens would silently attend to another sequence's data.
    padded = torch.cat([block_tables, torch.zeros(1, 8, dtype=torch.long, device="cuda")], dim=1)
    widened = paged_decode_attention(q, k_cache, v_cache, padded, context_lens, scale=scale)
    assert torch.equal(exact, widened)


@pytest.mark.skipif(NEEDS_GPU, reason=GPU_REASON)
def test_verify_refuses_an_unsupported_configuration():
    """A silent fallback would let a misconfiguration masquerade as a passing check."""
    q, k_cache, v_cache, block_tables, context_lens = _case(1, 28, 2, 128, 16, [64], torch.float32)
    with pytest.raises(ValueError):
        verify_against_reference(q, k_cache, v_cache, block_tables, context_lens)
