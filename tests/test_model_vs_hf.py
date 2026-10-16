"""Gate A -- reference-implementation equivalence.

Every test here compares this engine against ``transformers``' own Qwen2.5-VL on a tiny
random-weight model, with quantisation and pruning disabled.  This is the gate that catches
*silent* errors: a wrong MRoPE interleave or a mis-ordered patch matrix still produces a
correctly shaped tensor, so shape assertions prove nothing.

Tolerances are tight on purpose (fp32, exact equality where the computation is identical).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from qwen25vl_lite import rope
from qwen25vl_lite.imageproc import process_image
from qwen25vl_lite.model import apply_quantization, build_model, load_weights
from qwen25vl_lite.vision_budget import TIERS, tier_budget


def _hf_model(tiny_model_dir):
    return tiny_model_dir[1]


def test_rope_index_matches_hf(tiny_model_dir, tiny_hf_config, tiny_inputs):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    hf = _hf_model(tiny_model_dir)
    ids = tiny_inputs["input_ids"]
    grid = tiny_inputs["image_grid_thw"]
    mm = rope.mm_token_type_ids(ids, tiny_hf_config.image_token_id, tiny_hf_config.video_token_id)
    hf_pos, hf_delta = hf.model.get_rope_index(ids, mm, image_grid_thw=grid)
    my_pos, my_delta = rope.get_rope_index(
        ids,
        image_grid_thw=grid,
        image_token_id=tiny_hf_config.image_token_id,
        video_token_id=tiny_hf_config.video_token_id,
        spatial_merge_size=tiny_hf_config.vision_config.spatial_merge_size,
    )
    assert torch.equal(hf_pos, my_pos), "MRoPE position ids diverged from HF"
    assert torch.equal(hf_delta, my_delta), "rope_deltas diverged from HF"


def test_mrope_cos_sin_and_apply_match_hf(tiny_model_dir, tiny_hf_config, tiny_inputs):
    import pytest
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as hfq

    # HF renames and moves these helpers between releases.  5.20 dropped
    # `apply_multimodal_rotary_pos_emb` in favour of a generic `apply_rotary_pos_emb` with no
    # `mrope_section` argument, and moved the interleave into the cos/sin tables themselves.
    # On such a version the tables are not comparable to the 5.16 layout, so this test skips
    # rather than failing on a dependency bump that changes nothing in this engine.  The
    # interleave is still covered end to end by `test_full_logits_match_hf`, which passes on
    # both versions.
    hf_apply = getattr(hfq, "apply_multimodal_rotary_pos_emb", None)
    if hf_apply is None:
        pytest.skip(
            "transformers no longer exposes apply_multimodal_rotary_pos_emb; its cos/sin tables "
            "carry the MRoPE interleave instead and are a different layout to the 5.16 one"
        )

    hf = _hf_model(tiny_model_dir)
    tc = tiny_hf_config.text_config
    head_dim = tc.hidden_size // tc.num_attention_heads
    ids = tiny_inputs["input_ids"]
    grid = tiny_inputs["image_grid_thw"]
    pos, _ = rope.get_rope_index(
        ids,
        image_grid_thw=grid,
        image_token_id=tiny_hf_config.image_token_id,
        video_token_id=tiny_hf_config.video_token_id,
        spatial_merge_size=tiny_hf_config.vision_config.spatial_merge_size,
    )
    inv = rope.build_inv_freq(head_dim, tc.rope_parameters["rope_theta"])
    cos, sin = rope.mrope_cos_sin(pos, inv)
    # The instantiated module is stable across versions; the class name is not.
    hf_cos, hf_sin = hf.model.language_model.rotary_emb(
        torch.zeros(1, 1, 1, head_dim), pos
    )
    assert torch.equal(cos, hf_cos.float()), "MRoPE cos table diverged"
    assert torch.equal(sin, hf_sin.float()), "MRoPE sin table diverged"

    torch.manual_seed(3)
    q = torch.randn(1, tc.num_attention_heads, pos.shape[-1], head_dim)
    k = torch.randn(1, tc.num_key_value_heads, pos.shape[-1], head_dim)
    section = list(tc.rope_parameters["mrope_section"])
    hq, hk = hf_apply(q, k, cos, sin, section)
    mq, mk = rope.apply_multimodal_rotary_pos_emb(q, k, cos, sin, section)
    assert torch.equal(hq, mq) and torch.equal(hk, mk), (
        "MRoPE channel interleave diverged from HF (mrope_section must be repeated as a list, "
        "not doubled element-wise)"
    )


def test_vision_rope_matches_hf(tiny_model_dir, tiny_hf_config, tiny_inputs):
    """The vision rotary table, checked against the layout this engine was verified with.

    Use the instantiated module rather than importing the class: HF renamed
    `Qwen2_5_VisionRotaryEmbedding` to `Qwen2_5_VLVisionRotaryEmbedding` in 5.20, changed
    `forward` from `(position_ids) -> freqs` to `(x, position_ids) -> (cos, sin)`, changed the
    position-id layout from `[N, 2]` to `[2, N]`, and now recomposes the table to the full head
    width.  Those are three different conventions for the same quantity, and reverse-engineering
    the newest one here would add brittle code for no extra confidence: `test_full_logits_match_hf`
    already compares the whole vision tower and the final logits end to end, and it passes on
    both 5.16 and 5.20.  So this test skips rather than guess.
    """
    import inspect

    ref = _hf_model(tiny_model_dir).model.visual.rotary_pos_emb
    params = [p for p in inspect.signature(ref.forward).parameters if p != "self"]
    if len(params) != 1:
        pytest.skip(
            f"transformers changed the vision rotary API ({len(params)} args, recomposed table "
            "layout); the verified convention is the 5.16 one"
        )

    vc = tiny_hf_config.vision_config
    head_dim = vc.hidden_size // vc.num_heads
    inv = rope.build_inv_freq(head_dim // 2, 10_000.0)
    grid = tiny_inputs["image_grid_thw"]
    pos = rope.vision_rope_position_ids(grid, vc.spatial_merge_size)
    assert torch.allclose(rope.vision_rotary_freqs(pos, inv), ref(pos), atol=0.0), (
        "vision rotary frequencies diverged from HF"
    )


def test_vision_tower_matches_hf(tiny_model_dir, tiny_inputs):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

    model_dir, hf = tiny_model_dir
    cfg = _load_config(model_dir)
    mine = build_model(cfg)
    load_weights(model_dir, mine, strict=True)
    mine.eval()

    tower = Qwen2_5_VisionTransformerPretrainedModel(hf.config.vision_config)
    tower.load_state_dict({k[len("model.visual.") :]: v for k, v in hf.state_dict().items()
                           if k.startswith("model.visual.")}, strict=True)
    tower.eval()

    with torch.no_grad():
        ref = tower(tiny_inputs["pixel_values"], grid_thw=tiny_inputs["image_grid_thw"]).pooler_output
        got = mine.embed_images(tiny_inputs["pixel_values"], tiny_inputs["image_grid_thw"])
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-4), (
        f"vision tower diverged: max abs diff {(ref - got).abs().max().item():.3e}"
    )


def test_full_logits_match_hf(tiny_model_dir, tiny_inputs, tiny_hf_config):
    model_dir, hf = tiny_model_dir
    cfg = _load_config(model_dir)
    mine = build_model(cfg)
    load_weights(model_dir, mine, strict=True)
    mine.eval()

    mm = rope.mm_token_type_ids(
        tiny_inputs["input_ids"], tiny_hf_config.image_token_id, tiny_hf_config.video_token_id
    )
    with torch.no_grad():
        ref = hf(
            input_ids=tiny_inputs["input_ids"],
            pixel_values=tiny_inputs["pixel_values"],
            image_grid_thw=tiny_inputs["image_grid_thw"],
            mm_token_type_ids=mm,
        ).logits
        got = mine(
            input_ids=tiny_inputs["input_ids"],
            pixel_values=tiny_inputs["pixel_values"],
            image_grid_thw=tiny_inputs["image_grid_thw"],
        )
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3), (
        f"end-to-end logits diverged: max abs diff {(ref - got).abs().max().item():.3e}"
    )


def test_multimodal_logits_match_hf_with_windowing_engaged(tiny_model_dir, tiny_hf_config):
    """The session fixture's 56x84 image fits inside a single vision-attention window, so it
    never exercises `get_window_index`'s multi-window path.  A 300x420 image gives a 22x30
    patch grid -> 11x15 merge units -> 12 windows, which does.

    This test exists because `scripts/quality_gate.py` reported a 4.6e-3 multimodal gap that
    turned out to be a bug in the *gate* (it called HF without `mm_token_type_ids`, so HF
    silently skipped MRoPE).  The image size stayed in the suite anyway: it is the only case
    here where windowed attention is not degenerate.
    """
    model_dir, hf = tiny_model_dir
    cfg = _load_config(model_dir)
    mine = build_model(cfg)
    load_weights(model_dir, mine, strict=True)
    mine.eval()
    hf.model.rope_deltas = None

    rng = np.random.default_rng(0)
    size = (300, 420)
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for row in range(size[0]):
        arr[row, :, :] = (row % 40) * 6
    for _ in range(12):
        y = int(rng.integers(0, size[0] - 20))
        x = int(rng.integers(0, size[1] - 40))
        arr[y : y + 12, x : x + 30, :] = rng.integers(0, 255, size=(12, 30, 3))
    proc = process_image(Image.fromarray(arr), tier_budget(TIERS["full"]))
    grid = proc["grid_thw"]

    window = cfg.vision_config.window_size // cfg.vision_config.patch_size // cfg.vision_config.spatial_merge_size
    llm_h = int(grid[0, 1].item()) // cfg.vision_config.spatial_merge_size
    llm_w = int(grid[0, 2].item()) // cfg.vision_config.spatial_merge_size
    assert llm_h > window or llm_w > window, (
        f"image is too small to span more than one window ({llm_h}x{llm_w} units, "
        f"window {window}); the test would be degenerate"
    )

    n_vis = int(grid.prod().item() // 4)
    ids = torch.tensor([[1, 2, cfg.vision_start_token_id] + [cfg.image_token_id] * n_vis
                        + [cfg.vision_end_token_id, 3, 4]])
    mm = rope.mm_token_type_ids(ids, tiny_hf_config.image_token_id, tiny_hf_config.video_token_id)
    with torch.no_grad():
        ref = hf(input_ids=ids, pixel_values=proc["pixel_values"], image_grid_thw=grid,
                 mm_token_type_ids=mm).logits
        got = mine(input_ids=ids, pixel_values=proc["pixel_values"], image_grid_thw=grid)
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3), (
        f"windowed multimodal logits diverged: max abs diff {(ref - got).abs().max().item():.3e}"
    )


def test_hf_silently_skips_mrope_without_mm_token_type_ids(tiny_model_dir, tiny_hf_config, tiny_inputs):
    """A trap worth pinning down with a test, because it costs an afternoon otherwise.

    ``Qwen2_5_VLModel.compute_3d_position_ids`` sets ``can_compute_mrope = input_ids is not
    None and mm_token_type_ids is not None and (...)``.  If ``mm_token_type_ids`` is omitted,
    HF does not raise -- it falls back to plain 1-D positions and ignores the image grid.
    A comparison that forgets to pass it therefore measures this engine against a model that
    did not use MRoPE, and reports a spurious ~1e-2 divergence.

    So: this test asserts (a) the trap is real, and (b) our model tracks the *correct* call.
    """
    model_dir, hf = tiny_model_dir
    cfg = _load_config(model_dir)
    mine = build_model(cfg)
    load_weights(model_dir, mine, strict=True)
    mine.eval()
    hf.model.rope_deltas = None

    ids = tiny_inputs["input_ids"]
    mm = rope.mm_token_type_ids(ids, tiny_hf_config.image_token_id, tiny_hf_config.video_token_id)
    with torch.no_grad():
        with_mrope = hf(input_ids=ids, pixel_values=tiny_inputs["pixel_values"],
                        image_grid_thw=tiny_inputs["image_grid_thw"],
                        mm_token_type_ids=mm).logits
        without_mrope = hf(input_ids=ids, pixel_values=tiny_inputs["pixel_values"],
                           image_grid_thw=tiny_inputs["image_grid_thw"]).logits
        got = mine(input_ids=ids, pixel_values=tiny_inputs["pixel_values"],
                   image_grid_thw=tiny_inputs["image_grid_thw"])

    assert not torch.allclose(with_mrope, without_mrope, atol=1e-4, rtol=1e-3), (
        "HF produced the same logits with and without mm_token_type_ids; the trap this test "
        "documents no longer exists, so the comment above should be re-checked"
    )
    assert torch.allclose(got, with_mrope, atol=1e-4, rtol=1e-3), (
        "our model must match HF's MRoPE path, not its 1-D fallback"
    )


def test_text_only_logits_match_hf(tiny_model_dir, tiny_hf_config):
    model_dir, hf = tiny_model_dir
    cfg = _load_config(model_dir)
    mine = build_model(cfg)
    load_weights(model_dir, mine, strict=True)
    mine.eval()

    ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.long)
    mm = rope.mm_token_type_ids(ids, tiny_hf_config.image_token_id, tiny_hf_config.video_token_id)
    with torch.no_grad():
        ref = hf(input_ids=ids, mm_token_type_ids=mm).logits
        got = mine(input_ids=ids)
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3), (
        f"text-only logits diverged: max abs diff {(ref - got).abs().max().item():.3e}"
    )


def test_quantized_weights_dequantize_close_to_float(tiny_model_dir):
    """Gate B (structural part): the quantiser must be self-consistent before quality is judged."""
    from qwen25vl_lite.quant import (
        quantize_int4_groupwise,
        quantize_int8_per_channel,
        verify_scale_axis,
    )

    model_dir, _ = tiny_model_dir
    cfg = _load_config(model_dir)
    model = build_model(cfg)
    load_weights(model_dir, model, strict=True)
    layer = model.language_model.layers[0].self_attn.q_proj
    w = layer.weight.detach().float()

    q8 = quantize_int8_per_channel(w)
    assert verify_scale_axis(w, q8), "int8 scales are not indexed along the output-channel axis"
    rel8 = (q8.dequantize(torch.float32) - w).norm() / w.norm()
    assert rel8 < 0.02, f"int8 per-channel relative error {rel8.item():.4f} is too high"

    q4 = quantize_int4_groupwise(w, group_size=16)
    assert verify_scale_axis(w, q4), "int4 group scales are indexed inconsistently"
    rel4 = (q4.dequantize(torch.float32) - w).norm() / w.norm()
    assert rel4 < 0.15, f"int4 groupwise relative error {rel4.item():.4f} is too high"

    # A group size that does not divide K must still round-trip (K is padded internally).
    q4_pad = quantize_int4_groupwise(w[:, : w.shape[1] - 3], group_size=16)
    assert q4_pad.dequantize(torch.float32).shape == w[:, : w.shape[1] - 3].shape


def _load_config(model_dir):
    from qwen25vl_lite.config import Qwen25VLConfig

    return Qwen25VLConfig.from_pretrained(model_dir)
