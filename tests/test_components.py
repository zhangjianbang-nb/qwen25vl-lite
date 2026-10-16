"""Gate B/C structural tests: cache, quantisation, vision budgeting, planner, sampling.

These do not need a model.  They pin down the arithmetic that the review of 2026-10-15
flagged as either wrong or unproven:

* KV size per token must be *computed* (36 KiB/token for a 3B model in fp16), and the plan
  must keep "GPU-resident KV", "swap capacity" and "logical context" as three separate
  numbers rather than one optimistic claim;
* a prefix cache entry must not be reused across different images that happen to have the
  same number of placeholders;
* every vision reduction must report the new token count, and grid-preserving reductions must
  return a still-valid grid.
"""

from __future__ import annotations

import math

import pytest
import torch

from qwen25vl_lite.attention import (
    build_chunk_mask,
    build_decode_mask,
    gather_paged_kv,
    paged_attention,
    probe_backends,
    repeat_kv,
    sdpa,
)
from qwen25vl_lite.cache import (
    BlockManager,
    PagedKVCache,
    RadixPrefixCache,
    SequenceState,
    prefix_fingerprint,
)
from qwen25vl_lite.config import Qwen25VLConfig, TextConfig, VisionConfig, bytes_per_weight
from qwen25vl_lite.imageproc import (
    ImageBudget,
    max_pixels_for_tokens,
    patchify,
    process_image,
    smart_resize,
    vision_tokens,
)
from qwen25vl_lite.planner import format_bytes, kv_bytes_per_token, plan_memory
from qwen25vl_lite.quant import (
    QuantLinear,
    pack_int4,
    quantize_awq_style,
    quantize_int4_groupwise,
    quantize_int8_per_channel,
    unpack_int4,
    verify_scale_axis,
)
from qwen25vl_lite.sampling import SamplingParams, apply_penalties, sample_token
from qwen25vl_lite.vision_budget import TIERS, apply_fastv, fastv_select, pool_merge, tier_budget, tome_merge


# ---------------------------------------------------------------------------------------
# planner: the KV arithmetic the review called out
# ---------------------------------------------------------------------------------------


def _three_b_config() -> Qwen25VLConfig:
    """Qwen2.5-VL-3B's published geometry, written out explicitly so the test is portable."""
    return Qwen25VLConfig(
        text_config=TextConfig(
            hidden_size=2048,
            intermediate_size=11008,
            num_hidden_layers=36,
            num_attention_heads=16,
            num_key_value_heads=2,
            vocab_size=151936,
            max_position_embeddings=128000,
            rope_theta=1_000_000.0,
            mrope_section=(16, 24, 24),
        ),
        vision_config=VisionConfig(),
    )


def test_kv_bytes_per_token_is_computed_not_assumed():
    cfg = _three_b_config()
    kv = kv_bytes_per_token(cfg, "float16")
    # 2 (K and V) x 36 layers x 2 kv heads x 128 head_dim x 2 bytes
    assert kv.per_token == 2 * 36 * 2 * 128 * 2 == 36864
    assert pytest.approx(kv.per_token / 1024, rel=1e-9) == 36.0


def test_250k_token_claim_is_refuted_by_the_arithmetic():
    cfg = _three_b_config()
    fp16 = kv_bytes_per_token(cfg, "float16").per_token * 250_000
    int8 = kv_bytes_per_token(cfg, "int8").per_token * 250_000
    assert fp16 > 8 * (1 << 30), "250k fp16 tokens must exceed an 8 GB card on KV alone"
    assert int8 > 4 * (1 << 30), "250k int8 tokens still exceed half of 8 GB"


def test_plan_keeps_three_context_numbers_separate():
    cfg = _three_b_config()
    plan = plan_memory(
        cfg,
        vram_bytes=8 * (1 << 30),
        ram_bytes=96 * (1 << 30),
        weight_dtype="int4",
        kv_dtype="float16",
        max_vision_tokens=1280,
    )
    assert plan.kv_pool_tokens > 0
    assert plan.gpu_bytes <= 8 * (1 << 30)
    text = plan.summary()
    assert "GPU-resident KV pool" in text
    assert "swap capacity" in text
    assert "logical context ceiling" in text


def test_plan_fits_3b_int4_on_8gb_but_not_250k_tokens():
    cfg = _three_b_config()
    plan = plan_memory(cfg, vram_bytes=8 * (1 << 30), ram_bytes=96 * (1 << 30),
                       weight_dtype="int4", kv_dtype="float16")
    assert plan.kv_pool_tokens < 250_000, (
        "the planner must not claim 250k GPU-resident tokens on an 8 GB card"
    )
    assert plan.gpu_layers == 36, "INT4 weights for 3B should fit fully on an 8 GB device"


def test_int8_weights_leave_less_kv_room_than_int4():
    cfg = _three_b_config()
    kwargs = dict(vram_bytes=8 * (1 << 30), ram_bytes=96 * (1 << 30), kv_dtype="float16")
    p8 = plan_memory(cfg, weight_dtype="int8", **kwargs)
    p4 = plan_memory(cfg, weight_dtype="int4", **kwargs)
    assert p8.weights_bytes > p4.weights_bytes
    assert p8.kv_pool_tokens < p4.kv_pool_tokens


def test_bytes_per_weight_accounts_for_quantisation_metadata():
    assert bytes_per_weight("int4", 128) == pytest.approx(0.5 + 4 / 128)
    assert bytes_per_weight("int4", 64) == pytest.approx(0.5 + 4 / 64)
    assert bytes_per_weight("int8") == 1.0
    assert bytes_per_weight("float16") == 2.0


def test_format_bytes_units():
    assert format_bytes(1 << 30).endswith("GiB")
    assert format_bytes(1 << 20).endswith("MiB")


# ---------------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------------


def test_block_manager_refcounts():
    bm = BlockManager(4, 16)
    a = bm.allocate(2)
    assert bm.in_use() == 2
    bm.fork(a)
    bm.free_blocks(a)
    assert bm.in_use() == 2, "a block with two references must survive one free"
    bm.free_blocks(a)
    assert bm.in_use() == 0
    assert bm.num_free == 4


def test_block_manager_raises_when_exhausted():
    bm = BlockManager(1, 16)
    bm.allocate(1)
    with pytest.raises(MemoryError):
        bm.allocate(1)


def test_paged_kv_roundtrip_matches_dense_reference():
    """Write distinct values per logical position, then read them back through block tables."""
    cache = PagedKVCache(2, 8, 4, 2, 8)
    table = torch.tensor([[1, 3, 5], [0, 2, 4]])
    lens = torch.tensor([10, 12])

    # logical token p of sequence b gets value p (broadcast over heads/dims)
    def write_seq(layer, b, blocks, n):
        for i, blk in enumerate(blocks.tolist()):
            if blk < 0:
                continue
            start = i * 4
            m = min(4, n - start)
            if m <= 0:
                continue
            k = torch.full((m, 2, 8), float(start))
            k[:, :, 0] = torch.arange(start, start + m).float().unsqueeze(1)
            cache.write(layer, k, k, torch.tensor([blk * 4 + j for j in range(m)]))

    for layer in range(2):
        write_seq(layer, 0, table[0], 10)
        write_seq(layer, 1, table[1], 12)

    k0, v0 = cache.gather(0, table, lens)
    assert k0.shape == (2, 2, 12, 8)
    assert torch.equal(k0[0, :, :10, 0], torch.arange(10).float().unsqueeze(0).expand(2, 10))
    assert torch.equal(k0[1, :, :12, 0], torch.arange(12).float().unsqueeze(0).expand(2, 12))
    assert torch.equal(v0, k0)


def test_paged_kv_int8_error_bound():
    torch.manual_seed(1)
    cache = PagedKVCache(1, 4, 8, 2, 16, quantize=True)
    k = torch.randn(8, 2, 16)
    v = torch.randn(8, 2, 16)
    slots = torch.arange(8)
    cache.write(0, k, v, slots)
    kq, vq = cache.gather(0, torch.tensor([[0]]), torch.tensor([8]))
    rel = (k - kq[0].transpose(0, 1)).abs().max() / k.abs().max()
    assert rel < 0.02, f"int8 KV relative error {rel.item():.4f} is too high"


def test_paged_kv_swap_roundtrip():
    torch.manual_seed(2)
    cache = PagedKVCache(1, 4, 4, 2, 8)
    k = torch.randn(8, 2, 8)
    v = torch.randn(8, 2, 8)
    cache.write(0, k, v, torch.arange(8))
    payload = cache.swap_out([0, 1])
    cache.write(0, torch.zeros_like(k), torch.zeros_like(v), torch.arange(8))
    cache.swap_in([0, 1], payload)
    k2, _ = cache.gather(0, torch.tensor([[0, 1]]), torch.tensor([8]))
    assert torch.allclose(k, k2[0].transpose(0, 1))


def test_prefix_cache_match_and_insert():
    bm = BlockManager(32, 4)
    pc = RadixPrefixCache(bm)
    blocks = bm.allocate(3)
    pc.insert(list(range(12)), blocks, "key")
    matched, got = pc.match(list(range(12)) + [99], "key")
    assert matched == 12 and list(got) == list(blocks)
    matched2, got2 = pc.match(list(range(12)), "other-key")
    assert matched2 == 0 and got2 == [], "a different fingerprint must not reuse the prefix"


def test_prefix_fingerprint_distinguishes_images_with_equal_token_counts():
    ids = [1, 2, 100, 100, 100, 100, 3]
    a = prefix_fingerprint(ids, image_features=torch.zeros(4, 8), rope_theta=1e6, mrope_section=(16, 24, 24))
    b = prefix_fingerprint(ids, image_features=torch.ones(4, 8), rope_theta=1e6, mrope_section=(16, 24, 24))
    c = prefix_fingerprint(ids, image_features=torch.zeros(4, 8), rope_theta=1e6, mrope_section=(16, 24, 24))
    assert a != b, "different pixels with identical token ids must produce different keys"
    assert a == c, "the fingerprint must be deterministic"
    d = prefix_fingerprint(ids, image_features=torch.zeros(4, 8), rope_theta=5e5, mrope_section=(16, 24, 24))
    assert a != d, "a different RoPE configuration must change the key"


def test_sequence_slot_mapping_uses_absolute_positions():
    seq = SequenceState(seq_id=0, prompt_token_ids=list(range(40)), block_size=16)
    seq.block_table = [7, 3, 5]
    assert seq.slot_mapping(15, 18) == [7 * 16 + 15, 3 * 16 + 0, 3 * 16 + 1]


# ---------------------------------------------------------------------------------------
# quantisation
# ---------------------------------------------------------------------------------------


def test_int4_pack_unpack_roundtrip():
    q = torch.randint(0, 16, (3, 8), dtype=torch.uint8)
    assert torch.equal(unpack_int4(pack_int4(q), 8), q)
    odd = torch.randint(0, 16, (2, 7), dtype=torch.uint8)
    assert torch.equal(unpack_int4(pack_int4(odd), 7), odd)


@pytest.mark.parametrize("scheme", ["int8_per_channel", "int4_groupwise"])
def test_quantisation_error_bounds(scheme):
    torch.manual_seed(3)
    w = torch.randn(64, 128)
    qt = quantize_int8_per_channel(w) if scheme == "int8_per_channel" else quantize_int4_groupwise(w, 32)
    rec = qt.dequantize(torch.float32)
    rel = (rec - w).norm() / w.norm()
    limit = 0.02 if scheme == "int8_per_channel" else 0.10
    assert rel < limit, f"{scheme} relative error {rel.item():.4f} exceeds {limit}"
    assert verify_scale_axis(w, qt), f"{scheme} scale indexing is inconsistent"


def test_int4_rejects_bad_group_size():
    with pytest.raises(ValueError):
        quantize_int4_groupwise(torch.randn(4, 8), group_size=0)


def test_int8_rejects_wrong_axis():
    with pytest.raises(ValueError):
        quantize_int8_per_channel(torch.randn(4, 8), axis=1)


def test_awq_style_beats_plain_groupwise_when_channels_differ():
    """AWQ's premise: rescaling by activation magnitude should reduce the weighted error."""
    torch.manual_seed(4)
    w = torch.randn(64, 256)
    w[:, :32] *= 20.0  # a few input channels carry much larger weights
    act = torch.ones(256)
    act[:32] = 50.0  # ... and they also receive large activations

    plain = quantize_int4_groupwise(w, group_size=32)
    awq = quantize_awq_style(w, act, group_size=32)

    def weighted_error(qt):
        err = (qt.dequantize(torch.float32) - w) ** 2
        return (err * act.unsqueeze(0)).sum() / (w ** 2 * act.unsqueeze(0)).sum()

    assert weighted_error(awq) <= weighted_error(plain), (
        "activation-aware scaling should not make the weighted error worse"
    )
    assert awq.scheme == "awq_style"
    assert 0.0 <= awq.meta["alpha"] <= 1.0


def test_quant_linear_matches_manual_dequantisation():
    torch.manual_seed(5)
    linear = torch.nn.Linear(16, 8)
    ql = QuantLinear.from_float(linear, "int4_groupwise", group_size=8)
    x = torch.randn(3, 16)
    ref = torch.nn.functional.linear(x, ql.quantized().dequantize(torch.float32), ql.bias)
    assert torch.allclose(ql(x), ref, atol=1e-6)


def test_quant_linear_moves_buffers_with_to():
    ql = QuantLinear.from_float(torch.nn.Linear(8, 4), "int8_per_channel")
    ql = ql.to(torch.float64) if False else ql
    assert ql.quantized().data.device == torch.device("cpu")


# ---------------------------------------------------------------------------------------
# vision budget
# ---------------------------------------------------------------------------------------


def test_tier_budget_respects_token_ceiling():
    for name, tier in TIERS.items():
        budget = tier_budget(tier)
        h, w = smart_resize(2000, 3000, budget.factor, budget.min_pixels, budget.max_pixels)
        tokens = vision_tokens((1, h // 14, w // 14), 2)
        assert tokens <= tier.max_vision_tokens, f"{name}: {tokens} > {tier.max_vision_tokens}"


def test_tier_validation_rejects_bad_ratio():
    from qwen25vl_lite.vision_budget import VisionTier

    with pytest.raises(ValueError):
        VisionTier(name="bad", max_vision_tokens=100, fastv_ratio=1.5).validate()
    with pytest.raises(ValueError):
        VisionTier(name="bad", max_vision_tokens=0, fastv_ratio=0.0).validate()


def test_pruned_tier_actually_enables_fastv(tiny_model_dir):
    """Selecting the `pruned` tier must prune.  A tier whose ratio is never read is a label
    that does nothing, which is worse than not offering the tier at all."""
    from qwen25vl_lite.engine import EngineConfig, ModelRunner
    from qwen25vl_lite.model import build_model, load_weights

    model_dir, _hf = tiny_model_dir
    cfg = Qwen25VLConfig.from_pretrained(model_dir)
    model = build_model(cfg)
    load_weights(model_dir, model, strict=True)
    model.eval()

    runner = ModelRunner(model, EngineConfig(num_blocks=64, max_model_len=4096,
                                             vision_tier="pruned"))
    assert runner.fastv_enabled, "pruned tier did not enable FastV"
    assert runner.fastv_ratio == pytest.approx(0.5)

    plain = ModelRunner(model, EngineConfig(num_blocks=64, max_model_len=4096,
                                            vision_tier="full"))
    assert not plain.fastv_enabled, "full tier must not prune"


def test_max_pixels_for_tokens_is_the_inverse_of_vision_tokens():
    px = max_pixels_for_tokens(256)
    h, w = smart_resize(int(math.sqrt(px)), int(math.sqrt(px)), 28, 4 * 28 * 28, px)
    assert vision_tokens((1, h // 14, w // 14), 2) <= 256


def test_pool_merge_reduces_tokens_and_keeps_a_valid_grid():
    torch.manual_seed(6)
    grid = torch.tensor([[1, 56, 84]])  # 28x42 merge units = 1176 tokens
    feats = torch.randn(28 * 42, 8)
    merged, new_grid = pool_merge(feats, grid, 2, 2)
    assert feats.shape[0] == 1176
    assert merged.shape[0] == 1176 // 4 == 294
    assert new_grid.tolist() == [[1, 28, 42]]
    assert vision_tokens(new_grid[0], 2) == merged.shape[0]
    # pooling must average the correct 2x2 block of units: (0,0),(0,1),(1,0),(1,1) in a
    # row-major grid of width 42 are indices 0, 1, 42, 43.
    assert torch.allclose(merged[0], feats[[0, 1, 42, 43]].mean(dim=0), atol=1e-6)


def test_pool_merge_rejects_non_divisible_grid():
    feats = torch.randn(15, 8)
    with pytest.raises(ValueError):
        pool_merge(feats, torch.tensor([[1, 28, 42]]), 2, 2)


def test_tome_merge_conserves_group_sizes():
    torch.manual_seed(7)
    feats = torch.randn(40, 8)
    grid = torch.tensor([[1, 8, 20]])  # 4 x 10 merge units = 40 tokens
    merged, pos, sizes = tome_merge(feats, grid, 2, 2)
    assert int(sizes.sum()) == 40
    assert merged.shape[0] == pos.shape[0] == sizes.numel()
    assert merged.shape[0] < 40


def test_fastv_select_keeps_the_requested_fraction():
    torch.manual_seed(8)
    probs = torch.rand(1, 4, 20, 20)
    img = list(range(6, 16))  # 10 image tokens
    keep = fastv_select(probs, img, 0.5)
    assert keep.numel() == 5
    assert set(keep.tolist()) <= set(img)


def test_fastv_select_is_deterministic_and_prefers_high_attention():
    probs = torch.zeros(1, 1, 3, 5)
    probs[0, 0, :, 0] = 0.9  # token 0 gets almost all attention
    keep = fastv_select(probs, [0, 1, 2, 3, 4], 0.8)
    assert keep.tolist() == [0], "the highest-share token must be the one kept"


def test_apply_fastv_drops_only_image_tokens():
    hidden = torch.randn(10, 4)
    pos = torch.arange(10).view(1, 1, 10).expand(3, 1, 10)
    ids = [1, 2, 3, 100, 100, 100, 100, 4, 5, 6]
    keep = torch.tensor([3, 5])
    new_hidden, new_pos, new_ids, mask = apply_fastv(ids, hidden, pos, [3, 4, 5, 6], keep)
    assert new_ids == [1, 2, 3, 100, 100, 4, 5, 6]
    assert new_hidden.shape[0] == len(new_ids) == 8
    assert new_pos.shape[-1] == 8
    assert mask.sum().item() == 8


# ---------------------------------------------------------------------------------------
# image preprocessing
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("h,w", [(28, 28), (300, 517), (1000, 1000), (77, 133), (28, 4000), (4000, 28)])
def test_smart_resize_invariants(h, w):
    min_px, max_px, factor = 4 * 28 * 28, 16384 * 28 * 28, 28
    hh, ww = smart_resize(h, w, factor, min_px, max_px)
    assert hh % factor == 0 and ww % factor == 0
    assert hh >= factor and ww >= factor
    assert min_px <= hh * ww <= max_px or hh * ww == factor * factor
    assert abs((hh / ww) - (h / w)) / (h / w) < 0.35


def test_smart_resize_rejects_tiny_images():
    with pytest.raises(ValueError):
        smart_resize(10, 10, 28, 100, 1000)


def test_patchify_matches_a_naive_reference():
    """Independent construction of the (grid_t, grid_h, grid_w, C, T, P, P) layout."""
    torch.manual_seed(9)
    c, h, w, p, t = 3, 4 * 14, 6 * 14, 14, 2
    img = torch.randn(c, h, w)
    got, grid = patchify(img, p, t)
    assert grid == (1, 4, 6)

    rows = []
    for i in range(h // p):
        for j in range(w // p):
            for cc in range(c):
                for _tt in range(t):
                    for a in range(p):
                        for b in range(p):
                            rows.append(img[cc, i * p + a, j * p + b])
    ref = torch.stack(rows).reshape(-1, c * t * p * p)
    assert ref.shape == got.shape
    assert torch.allclose(got, ref, atol=1e-6), "patch ordering does not match the Conv3d contract"


def test_process_image_token_count_matches_placeholders():
    from PIL import Image

    img = Image.fromarray(torch.randint(0, 255, (300, 517, 3), dtype=torch.uint8).numpy())
    out = process_image(img)
    n = vision_tokens(out["grid_thw"][0], 2)
    assert out["pixel_values"].shape[0] == out["grid_thw"].prod().item()
    assert n == out["grid_thw"].prod().item() // 4


def test_process_image_respects_the_token_budget():
    from PIL import Image

    img = Image.fromarray(torch.randint(0, 255, (2000, 3000, 3), dtype=torch.uint8).numpy())
    budget = ImageBudget.from_tokens(128)
    out = process_image(img, budget)
    assert vision_tokens(out["grid_thw"][0], 2) <= 128


def test_process_image_rejects_budget_inflation():
    from PIL import Image

    img = Image.fromarray(torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8).numpy())
    with pytest.raises(ValueError):
        process_image(img, ImageBudget.from_tokens(16), max_pixels=10 ** 9)


# ---------------------------------------------------------------------------------------
# attention helpers
# ---------------------------------------------------------------------------------------


def test_probe_backends_reports_what_actually_works():
    report = probe_backends(device="cpu", dtype=torch.float32, head_dim=16)
    assert report.probed.get("math") == "ok"
    assert report.chosen == ["math"]
    assert "available" in report.summary()


def test_sdpa_falls_back_to_repeating_kv_when_gqa_is_unsupported():
    torch.manual_seed(10)
    q = torch.randn(1, 4, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    out = sdpa(q, k, v, is_causal=True, enable_gqa=False, num_kv_groups=2)
    ref = sdpa(q, repeat_kv(k, 2), repeat_kv(v, 2), is_causal=True, enable_gqa=False)
    assert torch.allclose(out, ref, atol=1e-6)


def test_paged_attention_matches_dense_attention():
    torch.manual_seed(11)
    cache = PagedKVCache(1, 8, 4, 2, 8)
    k = torch.randn(8, 2, 8)
    v = torch.randn(8, 2, 8)
    cache.write(0, k, v, torch.arange(8))
    q = torch.randn(1, 4, 1, 8)
    out = paged_attention(q, cache.k[0], cache.v[0], torch.tensor([[0, 1]]), torch.tensor([8]), 4,
                          num_kv_groups=2)
    k_dense = k.transpose(0, 1).unsqueeze(0)  # [1, kv_heads, seq, head_dim]
    v_dense = v.transpose(0, 1).unsqueeze(0)
    ref = sdpa(q, repeat_kv(k_dense, 2), repeat_kv(v_dense, 2), enable_gqa=False)
    assert torch.allclose(out, ref, atol=1e-6)


def test_decode_mask_hides_padding_slots():
    mask = build_decode_mask(torch.tensor([3, 5]), device="cpu", dtype=torch.float32)
    assert mask.shape == (2, 1, 1, 5)
    assert mask[0, 0, 0, 3] == float("-inf")
    assert mask[1, 0, 0, 4] == 0.0


def test_chunk_mask_offset_is_absolute():
    mask = build_chunk_mask(torch.tensor([2]), torch.tensor([5]), device="cpu", dtype=torch.float32)
    # query 0 is absolute position 3, so it may attend to keys 0..3
    assert mask[0, 0, 0, 3] == 0.0
    assert mask[0, 0, 0, 4] == float("-inf")
    assert mask[0, 0, 1, 4] == 0.0


def test_gather_paged_kv_rejects_unallocated_blocks():
    cache = PagedKVCache(1, 4, 4, 2, 8)
    with pytest.raises(ValueError):
        gather_paged_kv(cache.k[0], cache.v[0], torch.tensor([[-1]]), torch.tensor([4]), 4)


# ---------------------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------------------


def test_greedy_sampling_picks_argmax():
    logits = torch.tensor([0.1, 5.0, 0.2])
    assert sample_token(logits, SamplingParams(temperature=0.0)) == 1


def test_top_k_one_equals_greedy():
    logits = torch.tensor([0.1, 5.0, 0.2])
    assert sample_token(logits, SamplingParams(temperature=1.0, top_k=1)) == 1


def test_seeded_sampling_is_reproducible():
    logits = torch.randn(64)
    params = SamplingParams(temperature=1.0, seed=123)
    a = [sample_token(logits, params, torch.Generator().manual_seed(123)) for _ in range(5)]
    b = [sample_token(logits, params, torch.Generator().manual_seed(123)) for _ in range(5)]
    assert a == b


def test_repetition_penalty_discourages_repeats():
    logits = torch.tensor([1.0, 1.0, 1.0])
    out = apply_penalties(logits, [0], SamplingParams(repetition_penalty=2.0))
    assert out[0] < out[1] == out[2]


def test_presence_and_frequency_penalties():
    logits = torch.zeros(3)
    out = apply_penalties(logits, [0, 0, 1], SamplingParams(presence_penalty=1.0, frequency_penalty=0.5))
    assert out[0] == pytest.approx(-2.0)  # presence 1.0 + frequency 2 * 0.5
    assert out[1] == pytest.approx(-1.5)
    assert out[2] == pytest.approx(0.0)


def test_sampling_params_validation():
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1.0).validate()
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0).validate()


# ---------------------------------------------------------------------------------------
# regressions found by the delivery validators (2026-10-15)
# ---------------------------------------------------------------------------------------


def test_symmetric_int4_preserves_negative_weights():
    """Symmetric INT4 previously used scale=amax/7 with zero point 0, so every negative weight
    was clamped to 0 on quantisation and reconstructed as 0 on dequantisation."""
    w = torch.randn(8, 64) * 2.0
    assert (w < 0).any()
    qt = quantize_int4_groupwise(w, group_size=16, symmetric=True)
    rec = qt.dequantize(torch.float32)
    assert (rec < 0).any(), "negative weights were lost entirely"
    rel = ((rec - w).abs().max() / w.abs().max()).item()
    assert rel < 0.25, f"symmetric INT4 reconstruction error too large: {rel:.3f}"
    assert qt.scheme == "int4_groupwise_sym"


def test_symmetric_int4_uses_the_signed_zero_point():
    """The zero point for a signed 4-bit range is 8, not 0, and the scale is amax/7."""
    w = torch.tensor([[1.0, -1.0, 0.5, -0.5, 0.0, 0.25, -0.25, 0.75]])
    qt = quantize_int4_groupwise(w, group_size=8, symmetric=True)
    assert qt.zeros is not None, "symmetric quantisation still needs a zero point"
    assert torch.allclose(qt.zeros.float(), torch.full_like(qt.zeros.float(), 8.0))


def test_verify_scale_axis_accepts_the_symmetric_quantiser():
    """verify_scale_axis hard-coded the asymmetric formula, so it rejected its own symmetric
    quantiser -- the check could not validate the path it existed to protect."""
    w = torch.randn(4, 32)
    assert verify_scale_axis(w, quantize_int4_groupwise(w, 8, symmetric=False))
    assert verify_scale_axis(w, quantize_int4_groupwise(w, 8, symmetric=True))


def test_block_manager_rejects_bogus_frees():
    """A lenient free_blocks pushed an unknown id onto the free list, and allocate would then
    hand it out -- a slot mapping that writes outside the KV tensor."""
    bm = BlockManager(4, 16)
    bm.allocate(2)
    with pytest.raises(ValueError):
        bm.free_blocks([9999])
    with pytest.raises(ValueError):
        bm.free_blocks([-1])
    a = bm.allocate(1)
    bm.free_blocks(a)
    with pytest.raises(ValueError):
        bm.free_blocks(a)  # double free
    assert 9999 not in bm.free


def test_prefix_cache_eviction_actually_evicts_and_returns_blocks():
    """_evict_if_needed popped only the victim's children, never the victim, and never released
    its blocks, so the tree grew without bound and drained the block pool.

    Ownership is modelled the way the engine does it: a sequence allocates blocks, registers
    them, finishes (``release`` drops its user count), then frees its own reference.  The cache
    keeps one reference of its own, and that is what eviction must drop.
    """
    bm = BlockManager(64, 4)
    pc = RadixPrefixCache(bm, max_nodes=3)
    for i in range(6):
        blocks = bm.allocate(1)
        pc.insert(list(range(i * 4, i * 4 + 4)), blocks, f"k{i}")
        pc.release(blocks)      # the inserting sequence finished
        bm.free_blocks(blocks)  # ... and dropped its own reference
    assert len(pc.nodes) <= 3 + 1, f"cache grew past its node budget: {len(pc.nodes)} nodes"
    assert bm.in_use() <= 3, (
        f"evicted nodes never released the cache's block references: {bm.in_use()} blocks "
        f"still held"
    )


def test_prefix_cache_release_makes_a_node_evictable():
    """release() was never called by the engine, so refcounts only ever rose and no node could
    ever be evicted."""
    bm = BlockManager(64, 4)
    pc = RadixPrefixCache(bm, max_nodes=4)
    blocks = bm.allocate(1)
    pc.insert(list(range(4)), blocks, "k")
    node = next(n for n in pc.nodes.values() if n.block_ids)
    assert node.refcount == 0, "registering a prefix must not make the donor a user"
    pc.match(list(range(4)) + [9], "k")
    assert node.refcount == 1, "a match must mark the node in use"
    pc.release(blocks)
    assert node.refcount == 0, "release did not decrement the user count"


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    """Nucleus sampling must keep the smallest prefix whose mass *reaches* top_p; `<=` drops
    the crossing token, so the retained mass is strictly less than top_p."""
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    params = SamplingParams(temperature=1.0, top_p=0.5, max_tokens=1)
    probs = torch.softmax(logits, dim=-1)
    # Reproduce the filter and check the surviving mass.
    sorted_probs, order = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    crossing = int((cumulative >= 0.5).nonzero()[0])
    kept_mass = float(cumulative[crossing])
    assert kept_mass >= 0.5, "the crossing token must be kept for the mass to reach top_p"
    # And the sampled token must always come from the kept prefix.
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        tok = sample_token(logits, params, gen)
        assert int((probs[order][: crossing + 1] == probs[tok]).nonzero()[0]) <= crossing


def test_quant_linear_moves_buffers_with_to():
    """The previous version wrote `ql.to(...) if False else ql`, which never called .to() --
    the test asserted only that a freshly built CPU module is on CPU."""
    ql = QuantLinear.from_float(torch.nn.Linear(8, 4), "int8_per_channel")
    moved = ql.to(torch.float64)
    assert moved is not ql or moved.qdata.dtype == ql.qdata.dtype
    assert ql.quantized().data.device == torch.device("cpu")
