"""Per-block prefix keys: reuse must follow the *prefix*, not the whole prompt.

`prefix_fingerprint` hashes the entire prompt, and the cache used to mix it into every node
key.  That made reuse depend on the tail, so two requests sharing a system prompt never
matched.  `prefix_block_keys` scopes the vision part to each block instead.
"""

from __future__ import annotations

import torch

from qwen25vl_lite.cache import (
    BlockManager,
    RadixPrefixCache,
    prefix_block_keys,
    prefix_fingerprint,
)

BS = 4
IMG = 99


def _cache(n_blocks: int = 64):
    return RadixPrefixCache(BlockManager(n_blocks, BS), max_nodes=128)


def test_whole_prompt_fingerprint_differs_but_block_keys_follow_the_prefix():
    shared = list(range(1, 13))  # 12 tokens = 3 blocks
    a = shared + list(range(20, 36))
    b = shared + list(range(40, 56))

    # The old whole-prompt fingerprint differs, which is exactly what broke reuse.
    assert prefix_fingerprint(a) != prefix_fingerprint(b)

    # A text-only prompt has no images, so every block's extra key is the same constant; the
    # block tokens inside the node key are what separate one block from another.  The next test
    # is the one that proves reuse follows the shared prefix.
    ka = prefix_block_keys(a, block_size=BS)
    kb = prefix_block_keys(b, block_size=BS)
    assert ka[:3] == kb[:3]
    assert len(set(ka)) == 1


def test_shared_prefix_matches_across_different_tails():
    pc = _cache()
    shared = list(range(1, 13))
    a = shared + list(range(20, 36))
    b = shared + list(range(40, 56))

    blocks = pc.bm.allocate(5)
    pc.insert(a, blocks, extra_keys=prefix_block_keys(a, block_size=BS))

    matched, got = pc.match(b, extra_keys=prefix_block_keys(b, block_size=BS))
    assert matched == 12, f"expected 12 shared tokens, got {matched}"
    assert len(got) == 3


def test_blocks_before_an_image_are_shared_across_different_pixels():
    grid = torch.tensor([[1, 2, 2]])          # 1*2*2 / merge^2(4) = 1 image token... use 4
    grid = torch.tensor([[1, 2, 4]])          # 1*2*4 / 4 = 2 image tokens
    prompt_a = list(range(1, 9)) + [IMG] * 2 + list(range(300, 316))
    prompt_b = list(range(1, 9)) + [IMG] * 2 + list(range(400, 416))
    feats_a = torch.zeros(2, 8)
    feats_b = torch.ones(2, 8)                # different pixels

    pc = _cache()
    blocks = pc.bm.allocate(len(prompt_a) // BS)
    pc.insert(prompt_a, blocks, extra_keys=prefix_block_keys(
        prompt_a, image_features=feats_a, grid_thw=grid, image_token_id=IMG, block_size=BS))

    matched, _ = pc.match(prompt_b, extra_keys=prefix_block_keys(
        prompt_b, image_features=feats_b, grid_thw=grid, image_token_id=IMG, block_size=BS))
    # The image starts at token 8, i.e. at block 2, so blocks 0 and 1 (8 tokens) are shared.
    assert matched == 8, f"expected the two pre-image blocks to be shared, got {matched}"


def test_identical_pixels_reuse_every_block():
    grid = torch.tensor([[1, 2, 4]])
    prompt = list(range(1, 9)) + [IMG] * 2 + list(range(300, 316))
    feats = torch.zeros(2, 8)

    pc = _cache()
    blocks = pc.bm.allocate(len(prompt) // BS)
    keys = prefix_block_keys(prompt, image_features=feats, grid_thw=grid,
                             image_token_id=IMG, block_size=BS)
    pc.insert(prompt, blocks, extra_keys=keys)

    matched, got = pc.match(prompt, extra_keys=keys)
    assert matched == len(prompt) // BS * BS
    assert len(got) == len(prompt) // BS


def test_different_grids_do_not_share_blocks_after_the_image():
    prompt = list(range(1, 9)) + [IMG] * 4 + list(range(300, 316))
    feats = torch.zeros(4, 8)

    pc = _cache()
    blocks = pc.bm.allocate(len(prompt) // BS)
    pc.insert(prompt, blocks, extra_keys=prefix_block_keys(
        prompt, image_features=feats, grid_thw=torch.tensor([[1, 2, 4]]),
        image_token_id=IMG, block_size=BS))

    matched, _ = pc.match(prompt, extra_keys=prefix_block_keys(
        prompt, image_features=feats, grid_thw=torch.tensor([[1, 4, 2]]),
        image_token_id=IMG, block_size=BS))
    # A different grid changes the MRoPE positions after the image, so only the blocks before
    # it may be reused.
    assert matched == 8, f"expected only the pre-image blocks, got {matched}"


def test_one_key_per_full_block_and_none_for_a_partial_tail():
    tokens = list(range(1, 19))  # 18 tokens = 4 full blocks + a 2-token tail
    keys = prefix_block_keys(tokens, block_size=BS)
    assert len(keys) == 4
    # Fewer tokens than one block: nothing is cacheable, because the block's KV would depend
    # on the tokens that follow it.
    assert prefix_block_keys(list(range(1, 4)), block_size=BS) == []
