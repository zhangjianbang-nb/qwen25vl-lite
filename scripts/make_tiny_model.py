"""Build a tiny random-weight Qwen2.5-VL directory for tests, demos and benchmarks.

No download, no network: the point is a model whose *geometry* matches the real architecture
so the engine's code paths are exercised, at a size that runs on a CPU in seconds.

Usage::

    python scripts/make_tiny_model.py --out /tmp/tiny-qwen25vl
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def make_tiny(out: str, seed: int = 0) -> str:
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    vision = Qwen2_5_VLVisionConfig(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=4, in_chans=3,
        out_hidden_size=64, patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
        window_size=112, fullatt_block_indexes=[1], tokens_per_second=2, hidden_act="silu",
    )
    text = Qwen2_5_VLTextConfig(
        hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128, max_position_embeddings=512,
        rms_norm_eps=1e-6, rope_theta=1_000_000.0,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        hidden_act="silu", use_sliding_window=False, sliding_window=32768,
        tie_word_embeddings=True,
    )
    cfg = Qwen2_5_VLConfig(
        text_config=text, vision_config=vision,
        image_token_id=100, video_token_id=101,
        vision_start_token_id=98, vision_end_token_id=99,
        eos_token_id=1, bos_token_id=0, pad_token_id=0, tie_word_embeddings=True,
    )

    torch.manual_seed(seed)
    model = Qwen2_5_VLForConditionalGeneration(cfg).to(torch.float32).eval()
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory to write the model into")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = make_tiny(args.out, args.seed)
    print(f"wrote tiny Qwen2.5-VL to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
