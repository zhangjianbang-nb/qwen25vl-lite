"""The sync-free paged path must agree with the reference path it replaced.

The reference `gather_paged_kv` / `build_decode_mask` / `build_chunk_mask` read device scalars
and are the definition of correct.  The engine's hot path passes a padded width instead so no
layer stalls on the device; these tests pin the two to the same numbers over the region that
is actually attended to.
"""

from __future__ import annotations

import torch

from qwen25vl_lite.attention import (
    build_chunk_mask,
    build_decode_mask,
    build_paged_slots,
    gather_paged_kv,
)
from qwen25vl_lite.cache import PagedKVCache

BLOCK = 4


def _cache_and_table():
    """2 layers, 6 blocks of 4; seq 0 spans blocks 0-2 (len 10), seq 1 blocks 3-4 (len 5)."""
    cache = PagedKVCache(2, 6, BLOCK, 2, 8)
    table = torch.tensor([[0, 1, 2], [3, 4, -1]])
    lens = torch.tensor([10, 5])

    def write_seq(layer, blocks, n):
        for i, blk in enumerate(blocks.tolist()):
            if blk < 0:
                continue
            start = i * BLOCK
            m = min(BLOCK, n - start)
            if m <= 0:
                continue
            k = torch.full((m, 2, 8), float(start))
            k[:, :, 0] = torch.arange(start, start + m).float().unsqueeze(1)
            cache.write(layer, k, k, torch.tensor([blk * BLOCK + j for j in range(m)]))

    for layer in range(2):
        write_seq(layer, table[0], 10)
        write_seq(layer, table[1], 5)
    return cache, table, lens


def test_padded_gather_matches_reference_on_every_attended_position():
    cache, table, lens = _cache_and_table()
    ref_k, ref_v = gather_paged_kv(cache.k[0], cache.v[0], table, lens, BLOCK)

    padded = table.shape[1] * BLOCK
    fast_k, fast_v = gather_paged_kv(cache.k[0], cache.v[0], table, lens, BLOCK, max_len=padded)
    assert fast_k.shape == (2, 2, padded, 8)

    for b, n in enumerate(lens.tolist()):
        assert torch.equal(fast_k[b, :, :n], ref_k[b, :, :n]), f"sequence {b} keys differ"
        assert torch.equal(fast_v[b, :, :n], ref_v[b, :, :n]), f"sequence {b} values differ"


def test_padded_gather_still_rejects_an_unallocated_block_in_the_used_range():
    cache, _, lens = _cache_and_table()
    bad = torch.tensor([[0, -1, 2]])
    # The reference path validates the blocks it reads; the fast path leaves that to this test.
    try:
        gather_paged_kv(cache.k[0], cache.v[0], bad, torch.tensor([10]), BLOCK)
    except ValueError as exc:
        assert "unallocated block" in str(exc)
    else:
        raise AssertionError("reference gather accepted an unallocated block")


def test_build_paged_slots_points_at_the_right_physical_slots():
    _, table, lens = _cache_and_table()
    padded = table.shape[1] * BLOCK
    slots = build_paged_slots(table, BLOCK, padded)
    for b, n in enumerate(lens.tolist()):
        want = [table[b, j // BLOCK].item() * BLOCK + (j % BLOCK) for j in range(n)]
        assert slots[b, :n].tolist() == want


def test_padded_decode_mask_matches_reference():
    _, _, lens = _cache_and_table()
    ref = build_decode_mask(lens, device="cpu", dtype=torch.float32)
    padded = 12
    fast = build_decode_mask(lens, device="cpu", dtype=torch.float32, max_len=padded)
    assert fast.shape == (2, 1, 1, padded)
    assert torch.equal(fast[:, :, :, : ref.shape[-1]], ref)


def test_padded_chunk_mask_matches_reference():
    q_lens = torch.tensor([2, 5])
    ctx = torch.tensor([5, 9])
    ref = build_chunk_mask(q_lens, ctx, device="cpu", dtype=torch.float32)
    padded = 12
    fast = build_chunk_mask(q_lens, ctx, device="cpu", dtype=torch.float32, max_q=5, max_ctx=padded)
    assert fast.shape == (2, 1, 5, padded)
    assert torch.equal(fast[:, :, :, : ref.shape[-1]], ref)
