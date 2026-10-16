"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import os
import sys

import pytest
import torch

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _tiny_configs():
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )

    vision = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_chans=3,
        out_hidden_size=64,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        window_size=112,
        fullatt_block_indexes=[1],
        tokens_per_second=2,
        hidden_act="silu",
    )
    text = Qwen2_5_VLTextConfig(
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        hidden_act="silu",
        use_sliding_window=False,
        sliding_window=32768,
        tie_word_embeddings=True,
    )
    cfg = Qwen2_5_VLConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=98,
        vision_end_token_id=99,
        eos_token_id=1,
        bos_token_id=0,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    return cfg


@pytest.fixture(scope="session")
def tiny_hf_config():
    """A tiny random-weight config whose head_dim is 16 (mrope_section sums to 8)."""
    return _tiny_configs()


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, tiny_hf_config):
    """Save a randomly initialised HF model to disk so the streaming loader can read it."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(0)
    hf_model = Qwen2_5_VLForConditionalGeneration(tiny_hf_config)
    hf_model = hf_model.to(torch.float32)
    hf_model.eval()
    d = tmp_path_factory.mktemp("tiny-qwen25vl")
    hf_model.save_pretrained(str(d), safe_serialization=True)
    return str(d), hf_model


@pytest.fixture(scope="session")
def tiny_inputs(tiny_hf_config):
    """A mixed text+image batch: ids, pixel_values and the image grid."""
    import numpy as np
    from PIL import Image

    from qwen25vl_lite.imageproc import process_image

    torch.manual_seed(1)
    img = Image.fromarray((np.random.rand(56, 84, 3) * 255).astype(np.uint8))
    proc = process_image(img)
    grid = proc["grid_thw"]
    n_vis = int(grid.prod().item()) // (tiny_hf_config.vision_config.spatial_merge_size ** 2)
    ids = [10, 11, tiny_hf_config.vision_start_token_id] + [tiny_hf_config.image_token_id] * n_vis + [
        tiny_hf_config.vision_end_token_id, 12, 13, 14,
    ]
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "pixel_values": proc["pixel_values"],
        "image_grid_thw": grid,
    }
