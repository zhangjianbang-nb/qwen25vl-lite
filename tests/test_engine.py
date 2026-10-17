"""End-to-end engine tests: scheduling, prefix reuse, preemption, FastV, HF parity.

The engine's greedy output is compared against ``transformers.generate`` so the whole stack
(scheduler -> paged cache -> sampling) is checked against the reference implementation, not
just the individual layers.
"""

from __future__ import annotations

import time

import pytest
import torch

from qwen25vl_lite.config import Qwen25VLConfig
from qwen25vl_lite.engine import EngineConfig, EngineStats, LLMEngine, ModelRunner
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


def test_requests_sharing_a_prefix_reuse_it(tiny_model_dir):
    """Two prompts with the same opening and different tails must share the cached blocks.

    The cache key used to be the whole-prompt fingerprint, so this hit only when the *entire*
    prompt matched -- a shared system prompt bought nothing, which is the case a chat template
    spends most of its tokens on.
    """
    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    shared = list(range(1, 41))  # 40 tokens = 2 full blocks at block_size 16
    first_prompt = shared + [50, 51, 52, 53]
    second_prompt = shared + [60, 61, 62, 63]

    engine.add_request(first_prompt, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
    engine.run_until_complete()

    engine.add_request(second_prompt, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
    engine.run_until_complete()

    second = engine.finished[-1]
    assert second.metrics["cached_tokens"] >= 32, (
        f"shared prefix reused only {second.metrics['cached_tokens']} tokens"
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
    # `kv_blocks_used <= kv_blocks_total` holds by construction (`in_use()` is defined as
    # num_blocks - len(free)), so it asserts nothing.  The real claim is that the pool returns
    # to empty once every sequence has finished: no leaked blocks, no double ownership.
    assert stats["kv_blocks_used"] == 0, (
        f"{stats['kv_blocks_used']} blocks were still held after every request finished"
    )


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


# ---------------------------------------------------------------------------------------
# regressions found by the delivery validators (2026-10-15)
# ---------------------------------------------------------------------------------------


def test_engine_applies_sampling_penalties(tiny_model_dir):
    """`apply_penalties` existed but was never called on the engine path, so repetition,
    presence and frequency penalties silently had no effect."""
    from qwen25vl_lite.sampling import apply_penalties

    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    prompt = [3, 4, 5, 6, 7, 8]
    engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
    out = engine.run_until_complete()[0]
    logits = torch.randn(cfg.text_config.vocab_size)
    logits[out.output_token_ids[0]] = 10.0
    plain = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    penalised = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True,
                               repetition_penalty=100.0)
    assert int(torch.argmax(apply_penalties(logits, out.output_token_ids, plain))) == \
        out.output_token_ids[0]
    assert int(torch.argmax(apply_penalties(logits, out.output_token_ids, penalised))) != \
        out.output_token_ids[0], (
        "a repetition penalty of 100 must suppress the token that was already generated"
    )


def test_engine_rejects_invalid_and_unsupported_sampling_params(tiny_model_dir):
    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    with pytest.raises(ValueError):
        engine.add_request([1, 2, 3], SamplingParams(temperature=-1.0))
    with pytest.raises(ValueError):
        engine.add_request([1, 2, 3], SamplingParams(top_p=0.0))
    with pytest.raises(ValueError, match="stop"):
        engine.add_request([1, 2, 3], SamplingParams(stop=("</s>",)))


def test_prefix_cache_references_are_released_when_a_request_finishes(tiny_model_dir):
    """The engine never called `release`, so cached blocks were pinned forever.  After the
    request finishes, the cache keeps its own reference but the pool must be reclaimable."""
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, enable_prefix_cache=True)
    prompt = list(range(20, 60))
    engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
    engine.run_until_complete()
    pc = engine.runner.prefix_cache
    assert pc.hits + pc.misses > 0
    users = [n.refcount for n in pc.nodes.values() if n.block_ids]
    assert users and all(r == 0 for r in users), (
        f"cached nodes still report live users after every request finished: {users}"
    )


def test_stats_payload_is_json_serialisable_before_any_decode_step(tiny_model_dir):
    """A monitor reads /stats as JSON, so every value must serialise -- including the rolling
    decode window, which is deliberately kept out of the dataclass fields for that reason."""
    import json

    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    stats = engine.stats()
    json.dumps(stats)  # raises TypeError if any value is a deque or tensor

    assert stats["decode_tokens_per_s"] == 0.0, "no decode step has run yet"
    assert stats["decode_step_ms"] == 0.0
    assert stats["decode_window"] == 0
    assert stats["decode_active"] == 0.0
    assert stats["decode_idle_s"] == -1.0, "-1.0 means 'no step has ever run'"


def test_stats_report_a_live_decode_rate_after_decoding(tiny_model_dir):
    import json

    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    engine.add_request(list(range(1, 21)), SamplingParams(temperature=0.0, max_tokens=6,
                                                         ignore_eos=True))
    engine.run_until_complete()
    stats = engine.stats()
    json.dumps(stats)

    assert stats["decode_steps"] > 0
    assert stats["decode_window"] > 0, "the rolling window should hold the steps just run"
    assert stats["decode_active"] == 1.0, "the last step finished milliseconds ago"
    assert stats["decode_tokens_per_s"] > 0.0
    assert stats["decode_step_ms"] > 0.0


def test_decode_rate_falls_to_zero_once_the_engine_goes_idle():
    """A monitor must not keep showing the last rate it saw.

    The window entries carry their own end time, so a step that finished long ago stops
    counting -- 10 s is far past the four-step-time freshness bound for a 12 ms step -- and the
    rate reads 0 with `decode_idle_s` saying how long the engine has been quiet.
    """
    st = EngineStats()
    assert st.decode_active is False
    assert st.decode_tokens_per_s == 0.0
    assert st.decode_idle_s == -1.0

    st.record_decode_step(0.012, 1)
    assert st.decode_active is True
    assert st.decode_tokens_per_s == pytest.approx(1 / 0.012)
    assert st.decode_idle_s < 0.1

    st._decode_window[0] = (time.perf_counter() - 10.0, 0.012, 1)
    assert st.decode_active is False
    assert st.decode_tokens_per_s == 0.0, "a 10 s old step is not the current rate"
    assert st.decode_step_ms == 0.0, "the step-cost columns describe *now*, so they zero too"
    assert st.decode_window == 0
    assert st.decode_idle_s == pytest.approx(10.0, abs=0.05)


def test_decode_rate_tracks_batch_size_not_cumulative_counters(tiny_model_dir):
    """The rate is tokens/second over the recent window, so a batch of N tokens at one step
    time reports N/step_time -- it is a *rate*, not a token tally."""
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, max_num_seqs=4)
    for i in range(4):
        engine.add_request(list(range(1 + i, 21 + i)),
                           SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
    engine.run_until_complete()

    st = engine.runner.stats
    fresh = st._fresh_steps()
    assert fresh, "the steps just run must still be fresh"
    expected = sum(n for _, _, n in fresh) / sum(s for _, s, _ in fresh)
    assert st.decode_tokens_per_s == pytest.approx(expected)
    assert st.decode_tokens_per_s > 0.0
