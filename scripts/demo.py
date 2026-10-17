"""End-to-end demo that needs no download and no GPU.

Run::

    python scripts/demo.py

It builds a tiny random-weight Qwen2.5-VL, loads it through the streaming loader, serves a
multimodal request through the scheduler, shows prefix-cache reuse, and prints the memory plan
for the 8 GB target machine.  The point is to exercise every layer of the stack in one go
without pretending the output text is meaningful -- the weights are random.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qwen25vl_lite.config import Qwen25VLConfig  # noqa: E402
from qwen25vl_lite.engine import EngineConfig, LLMEngine, ModelRunner  # noqa: E402
from qwen25vl_lite.imageproc import process_image  # noqa: E402
from qwen25vl_lite.model import apply_quantization, build_model, load_weights  # noqa: E402
from qwen25vl_lite.planner import kv_bytes_per_token, plan_memory  # noqa: E402
from qwen25vl_lite.sampling import SamplingParams  # noqa: E402
from qwen25vl_lite.vision_budget import TIERS, summarise_tiers, tier_budget  # noqa: E402


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    from make_tiny_model import make_tiny

    model_dir = os.path.join(tempfile.gettempdir(), "qwen25vl-lite-demo")
    if not os.path.isdir(model_dir):
        make_tiny(model_dir)
    cfg = Qwen25VLConfig.from_pretrained(model_dir)
    banner("configuration")
    print(cfg.describe())
    print(f"KV per token: {kv_bytes_per_token(cfg, 'float16').describe()}")

    banner("vision tiers")
    print(summarise_tiers())

    banner("memory plan for the 8 GB target")
    plan = plan_memory(cfg, vram_bytes=8 * (1 << 30), ram_bytes=96 * (1 << 30),
                       weight_dtype="int4", kv_dtype="float16")
    print(plan.summary())

    banner("streaming weight load")
    model = build_model(cfg)
    t0 = time.time()
    stats = load_weights(model_dir, model, dtype=torch.float32)
    print(f"loaded {stats['loaded']} tensors in {time.time()-t0:.2f}s "
          f"(tied head skipped: {stats['skipped_tied']})")

    banner("quantisation")
    model_q = build_model(cfg)
    apply_quantization(model_q, "int4_groupwise", group_size=16)
    load_weights(model_dir, model_q, dtype=torch.float32)
    rep = model_q.memory_report()
    print(f"float weights {rep['float_bytes']/1e6:.2f} MB, quantised {rep['quantized_bytes']/1e6:.2f} MB")

    banner("multimodal request through the scheduler")
    image = Image.fromarray((np.random.rand(300, 400, 3) * 255).astype(np.uint8))
    proc = process_image(image, tier_budget(TIERS["half"]))
    n_vis = proc["grid_thw"].prod().item() // (cfg.vision_config.spatial_merge_size ** 2)

    ids = ([1, 2, 3, cfg.vision_start_token_id] + [cfg.image_token_id] * int(n_vis)
           + [cfg.vision_end_token_id, 4, 5, 6])
    ecfg = EngineConfig(num_blocks=256, block_size=16, max_num_seqs=4,
                        max_num_batched_tokens=512, max_model_len=4096,
                        vision_tier="half", enable_prefix_cache=True)
    runner = ModelRunner(model, ecfg, device="cpu")
    engine = LLMEngine(runner, ecfg)
    print(f"vision tokens for this image: {int(n_vis)} (tier 'half')")

    engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True),
                       pixel_values=proc["pixel_values"], image_grid_thw=proc["grid_thw"])
    outs = engine.run_until_complete()
    print(f"generated {len(outs[0].output_token_ids)} tokens: {outs[0].output_token_ids}")
    print(f"stats: {engine.stats()}")

    banner("prefix-cache reuse on a repeated request")
    engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True),
                       pixel_values=proc["pixel_values"], image_grid_thw=proc["grid_thw"])
    engine.run_until_complete()
    print(f"second request reused {engine.finished[-1].metrics['cached_tokens']:.0f} tokens of its prompt")

    banner("attention backend report")
    print(runner.backend_report.summary() if runner.backend_report else str(runner.backends))

    banner("done")
    print("All stages ran. The generated text is meaningless: the weights are random.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
