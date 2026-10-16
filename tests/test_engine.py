"""End-to-end engine tests: scheduling, prefix reuse, preemption, FastV, HF parity.

The engine's greedy output is compared against ``transformers.generate`` so the whole stack
(scheduler -> paged cache -> sampling) is checked against the reference implementation, not
just the individual layers.
"""

from __future__ import annotations

import pytest
import torch

from qwen25vl_lite.config import Qwen25VLConfig
from qwen25vl_lite.engine import EngineConfig, LLMEngine, ModelRunner
from qwen25vl_lite.model import build_model, load_weights
from qwen25vl_lite.sampling import SamplingParams


def _stack(tiny_model_dir, **ecfg_kwargs):
    model_dir, hf = tiny_model_dir
    cfg = Qwen25VLConfig.from_pretrained(model_dir)
    model = build_model(cfg)
    load_weights(model_dir, model, strict=True)
    model.eval()
    hf.model.rope_deltas = None
    base = dict(num_blocks=64, block_size=16, max_num_batched_tokens=128,
                max_model_len=1024, enable_prefix_cache=True)
    base.update(ecfg_kwargs)
    ecfg = EngineConfig(**base)
    runner = ModelRunner(model, ecfg, device="cpu")
    return LLMEngine(runner, ecfg), cfg, model_dir, hf


def test_greedy_generation_matches_hf_generate(tiny_model_dir):
    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    prompt = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    with torch.no_grad():
        ref = hf.generate(
            input_ids=torch.tensor([prompt]),
            attention_mask=torch.ones((1, len(prompt)), dtype=torch.long),
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=0,
        )
    ref_new = ref[0, len(prompt):].tolist()

    engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
    out = engine.run_until_complete()
    assert len(out) == 1
    got = out[0].output_token_ids
    assert got == ref_new, f"engine greedy output {got} != HF {ref_new}"


def test_identical_requests_reuse_the_prefix_cache(tiny_model_dir):
    """A repeated prompt must reuse its KV -- but only once the first request has run.

    Requests admitted in the *same* batch cannot reuse each other's prefix: the first one has
    not finished prefill yet when the second is scheduled.  That ordering is part of the
    contract, so the test drives the two requests in sequence.
    """
    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    prompt = list(range(1, 41))  # 40 tokens = 2 full blocks

    engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
    engine.run_until_complete()
    assert engine.stats()["prefix_cache_hits"] == 0, "the first request has nothing to reuse"

    engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
    engine.run_until_complete()

    first, second = engine.finished
    assert first.output_token_ids == second.output_token_ids, (
        "a prefix-cache hit must reproduce the same continuation"
    )
    assert second.metrics["cached_tokens"] >= 32, (
        f"second request only reused {second.metrics['cached_tokens']} tokens"
    )
    assert engine.stats()["prefix_cache_hits"] >= 1


def test_concurrent_requests_all_finish(tiny_model_dir):
    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    for i in range(4):
        engine.add_request(list(range(1 + i, 21 + i)), SamplingParams(temperature=0.0, max_tokens=4,
                                                                    ignore_eos=True))
    outs = engine.run_until_complete()
    assert len(outs) == 4
    assert all(o.finish_reason == "length" for o in outs)
    assert engine.stats()["decode_steps"] > 0


def test_preemption_keeps_the_pool_within_budget(tiny_model_dir):
    """With a deliberately tiny pool, the engine must preempt rather than over-allocate."""
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, num_blocks=6, max_num_seqs=4,
                                        enable_prefix_cache=False)
    for i in range(4):
        engine.add_request(list(range(1 + i, 33 + i)), SamplingParams(temperature=0.0, max_tokens=6,
                                                                     ignore_eos=True))
    outs = engine.run_until_complete()
    stats = engine.stats()
    assert len(outs) == 4, "every request must still complete after preemption"
    assert stats["preemptions"] > 0, "the test configuration should have forced preemption"
    assert stats["kv_blocks_used"] <= stats["kv_blocks_total"]


def test_preemption_by_recompute_also_completes(tiny_model_dir):
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, num_blocks=6, max_num_seqs=4,
                                        enable_prefix_cache=False, preemption="recompute")
    for i in range(4):
        engine.add_request(list(range(1 + i, 33 + i)), SamplingParams(temperature=0.0, max_tokens=4,
                                                                     ignore_eos=True))
    outs = engine.run_until_complete()
    assert len(outs) == 4
    assert engine.stats()["preemptions"] > 0


def test_multimodal_request_through_the_engine(tiny_model_dir, tiny_hf_config, tiny_inputs):
    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    ids = tiny_inputs["input_ids"][0].tolist()
    engine.add_request(
        ids,
        SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        pixel_values=tiny_inputs["pixel_values"],
        image_grid_thw=tiny_inputs["image_grid_thw"],
    )
    outs = engine.run_until_complete()
    assert len(outs) == 1
    assert len(outs[0].output_token_ids) == 4
    assert engine.stats()["vision_tokens_total"] > 0


def test_vision_placeholder_mismatch_is_rejected(tiny_model_dir, tiny_hf_config, tiny_inputs):
    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    ids = tiny_inputs["input_ids"][0].tolist()
    bad = ids[:]
    # Drop one image placeholder so the feature rows no longer match.
    for i, t in enumerate(bad):
        if t == cfg.image_token_id:
            bad.pop(i)
            break
    with pytest.raises(ValueError):
        engine.add_request(bad, SamplingParams(temperature=0.0, max_tokens=2),
                           pixel_values=tiny_inputs["pixel_values"],
                           image_grid_thw=tiny_inputs["image_grid_thw"])


def test_fastv_prunes_and_still_completes(tiny_model_dir, tiny_hf_config, tiny_inputs):
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, enable_fastv=True, fastv_ratio=0.5,
                                        fastv_layer=1, max_num_batched_tokens=512)
    ids = tiny_inputs["input_ids"][0].tolist()
    n_image = sum(1 for t in ids if t == cfg.image_token_id)

    engine.add_request(
        ids,
        SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
        pixel_values=tiny_inputs["pixel_values"],
        image_grid_thw=tiny_inputs["image_grid_thw"],
    )
    outs = engine.run_until_complete()
    assert len(outs) == 1
    seq = outs[0]
    remaining = sum(1 for t in seq.prompt_token_ids if t == cfg.image_token_id)
    assert remaining < n_image, (
        f"FastV kept every image token ({remaining}/{n_image}); pruning did not happen"
    )
    assert len(seq.output_token_ids) == 3


def test_fastv_refuses_a_prompt_that_does_not_fit_one_chunk(tiny_model_dir):
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, enable_fastv=True, fastv_ratio=0.5,
                                        max_num_batched_tokens=8)
    engine.add_request(list(range(1, 40)), SamplingParams(temperature=0.0, max_tokens=2,
                                                          ignore_eos=True))
    with pytest.raises(ValueError):
        engine.run_until_complete()


def test_long_prompt_is_chunked_not_rejected(tiny_model_dir):
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, max_num_batched_tokens=16)
    prompt = list(range(1, 65))  # 64 tokens -> 4 chunks
    engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True))
    outs = engine.run_until_complete()
    assert len(outs) == 1
    assert engine.stats()["prefill_chunks"] >= 4


def test_request_longer_than_max_model_len_is_rejected(tiny_model_dir):
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, max_model_len=64)
    with pytest.raises(ValueError):
        engine.add_request(list(range(1, 100)), SamplingParams(temperature=0.0, max_tokens=1))
