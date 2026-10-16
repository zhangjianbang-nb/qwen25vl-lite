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
from qwen25vl_lite.rope import text_position_ids
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
    # The response reports the submitted prompt (FastV rewrites the internal one), so pruning is
    # observed through the statistic instead -- which is also the number FastV used to overstate.
    prefilled = engine.stats()["vision_tokens_total"]
    assert prefilled < n_image, (
        f"FastV kept every image token ({prefilled}/{n_image}); pruning did not happen"
    )
    assert len(seq.output_token_ids) == 3


def test_fastv_refuses_only_the_offending_request(tiny_model_dir):
    """FastV needs the whole prompt in one chunk and refuses otherwise -- but refusing must not
    take the engine down: the other requests still have to be served."""
    # fastv_layer=1, because the tiny fixture has two layers and a prune layer must exist.
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, enable_fastv=True, fastv_ratio=0.5,
                                        fastv_layer=1, max_num_batched_tokens=8)
    engine.add_request(list(range(1, 40)), SamplingParams(temperature=0.0, max_tokens=2,
                                                          ignore_eos=True))
    engine.add_request([3, 4, 5], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
    outs = engine.run_until_complete()
    by_id = {o.seq_id: o for o in outs}
    assert len(outs) == 2, "the engine must finish every request, including the refused one"
    assert by_id[0].finish_reason == "error", "the over-long FastV request must be refused"
    assert by_id[0].error and "chunk" in by_id[0].error
    assert by_id[1].finish_reason != "error", "the short request must still be served"
    assert len(by_id[1].output_token_ids) == 2


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


def test_preemption_releases_prefix_cache_references(tiny_model_dir):
    """Preempting a sequence must drop its cache user reference.  Preemption is easy to forget
    and leaks the node permanently, because `match` is the only thing that raises the count."""
    # 8 blocks of 16 = 128 tokens: one 65-token request fits, two at once do not, so the
    # scheduler must preempt.  The cached prefix holds blocks too, which is what exercises the
    # cache-eviction path under pressure.
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, num_blocks=8, max_num_seqs=2,
                                        enable_prefix_cache=True)
    # First request populates the cache; the rest reuse and then get preempted.
    engine.add_request(list(range(20, 84)), SamplingParams(temperature=0.0, max_tokens=3,
                                                           ignore_eos=True))
    first = engine.run_until_complete()
    assert len(first) == 1
    for i in range(3):
        engine.add_request(list(range(20, 84)) + [50 + i],
                           SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True))
    outs = engine.run_until_complete()
    # `run_until_complete` returns only what finished during that call, so the warm-up request
    # is not repeated here.
    assert len(first) == 1
    assert len(outs) == 3, "every request must complete despite preemption"
    pc = engine.runner.prefix_cache
    users = [n.refcount for n in pc.nodes.values() if n.block_ids]
    assert all(r == 0 for r in users), (
        f"a preempted sequence left its cache reference behind: {users}"
    )


def test_block_aligned_repeated_prompt_leaves_no_pinned_cache_node(tiny_model_dir):
    """The block-aligned case is the one that leaks.

    `match` raises the user count on *every* node it walks, but a sequence keeps one token
    back for prefill, so a prompt whose length is an exact multiple of `block_size` adopts one
    node fewer than it matched.  The surplus node's count was raised and never lowered, which
    pinned it (and its block) forever.  The earlier tests used 40- and 65-token prompts and so
    never hit the boundary.
    """
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, num_blocks=32, block_size=16,
                                        enable_prefix_cache=True)
    prompt = list(range(20, 84))  # 64 tokens: exactly 4 blocks
    assert len(prompt) % 16 == 0
    engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
    engine.run_until_complete()
    for _ in range(3):  # each repeat is a full-prefix hit and the leaking path
        engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
        engine.run_until_complete()
    pc = engine.runner.prefix_cache
    users = [n.refcount for n in pc.nodes.values() if n.block_ids]
    assert users and all(r == 0 for r in users), (
        f"a block-aligned repeat pinned cache nodes permanently: {users}"
    )
    # The cache legitimately keeps its own block reference per cached node, so the pool is not
    # fully free.  What must not happen is *drain*: repeating the same prompt has to leave the
    # pool exactly where it was, not one block poorer per repeat.
    free_now = engine.runner.block_manager.num_free
    engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
    engine.run_until_complete()
    assert engine.runner.block_manager.num_free == free_now, (
        f"repeating the prompt drained the block pool ({free_now} -> "
        f"{engine.runner.block_manager.num_free})"
    )


def test_kv_dtype_selects_the_storage_layout(tiny_model_dir):
    """`kv_dtype` used to be a field nothing read.  It is now the single source of truth:
    `int8` turns on quantised KV, and the float names pick the storage dtype."""
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, kv_dtype="int8")
    assert engine.runner.cache.quantize is True
    assert engine.ecfg.quantize_kv is True

    engine, cfg, model_dir, hf = _stack(tiny_model_dir, kv_dtype="float32")
    assert engine.runner.cache.quantize is False
    assert engine.runner.cache.k.dtype == torch.float32

    with pytest.raises(ValueError, match="kv_dtype"):
        EngineConfig(kv_dtype="int4")


def test_fastv_layer_beyond_the_model_depth_is_rejected_at_construction(tiny_model_dir):
    """A prune layer that does not exist used to surface as a missing-probabilities error in
    the middle of a request.  It is a configuration error and is rejected as one."""
    model_dir, hf = tiny_model_dir
    cfg = Qwen25VLConfig.from_pretrained(model_dir)
    model = build_model(cfg)
    load_weights(model_dir, model, strict=True)
    depth = cfg.text_config.num_hidden_layers
    ecfg = EngineConfig(num_blocks=8, block_size=16, enable_fastv=True, fastv_ratio=0.5,
                        fastv_layer=depth)
    with pytest.raises(ValueError, match="does not exist"):
        ModelRunner(model, ecfg, device="cpu")


def test_fastv_reports_the_tokens_it_actually_prefilled(tiny_model_dir, tiny_inputs):
    """`vision_tokens_total` counted the image tokens in the *submitted* prompt, so FastV made
    the statistic overstate the work done.  It now counts the survivors."""
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, enable_fastv=True, fastv_ratio=0.5,
                                        fastv_layer=1, max_num_batched_tokens=512)
    ids = tiny_inputs["input_ids"][0].tolist()
    n_image = sum(1 for t in ids if t == cfg.image_token_id)
    engine.add_request(
        ids, SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
        pixel_values=tiny_inputs["pixel_values"], image_grid_thw=tiny_inputs["image_grid_thw"],
    )
    out = engine.run_until_complete()[0]
    assert out.prompt_token_ids == ids, (
        "FastV rewrote the prompt; the response must report what the caller submitted"
    )
    assert engine.stats()["vision_tokens_total"] < n_image, (
        f"pruning must reduce the counted vision tokens ({engine.stats()['vision_tokens_total']}"
        f" vs {n_image} submitted)"
    )


def test_fastv_recomputes_rope_delta_for_the_pruned_prompt(tiny_model_dir, tiny_inputs):
    """Pruning shortens the prompt, and `rope_delta` is derived from its positions.  Leaving the
    old value made the first decode position fall *behind* the prompt's own positions, so every
    generated token was rotated wrongly."""
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, enable_fastv=True, fastv_ratio=0.5,
                                        fastv_layer=1, max_num_batched_tokens=512)
    ids = tiny_inputs["input_ids"][0].tolist()
    engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
                       pixel_values=tiny_inputs["pixel_values"],
                       image_grid_thw=tiny_inputs["image_grid_thw"])
    engine.step()
    seq = engine.running[0]
    assert len(seq.prompt_token_ids) < len(ids), "the fixture must actually prune"
    # The invariant is about the *original* cursor, not the largest surviving position: if the
    # last image token is pruned, recomputing from max() would restart decoding inside the
    # prompt's own coordinate range.  Derive the cursor independently from the submitted ids.
    orig_pos, _delta = engine.runner.model.position_ids(
        torch.tensor([ids]), image_grid_thw=tiny_inputs["image_grid_thw"]
    )
    cursor = int(orig_pos.max()) + 1
    pos = text_position_ids(1, len(seq.prompt_token_ids), seq.rope_delta, "cpu")
    assert int(pos.max()) == cursor, (
        f"the first decoded token got position {int(pos.max())}, but the prompt's cursor is "
        f"{cursor}; decoding must continue from the cursor"
    )


def test_fastv_requests_stay_out_of_the_prefix_cache(tiny_model_dir, tiny_inputs):
    """A pruned request's tokens no longer match what the cache would be asked to look up, so
    registering or adopting it would silently never hit -- and adopting it would skip pruning
    entirely, making the same prompt produce different output depending on cache state."""
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, enable_fastv=True, fastv_ratio=0.5,
                                        fastv_layer=1, max_num_batched_tokens=512,
                                        enable_prefix_cache=True)
    ids = tiny_inputs["input_ids"][0].tolist()
    outs = []
    for _ in range(2):
        engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True),
                           pixel_values=tiny_inputs["pixel_values"],
                           image_grid_thw=tiny_inputs["image_grid_thw"])
        outs.append(engine.run_until_complete()[0])
    assert outs[0].output_token_ids == outs[1].output_token_ids, (
        "the same FastV request produced different output on the second run"
    )
    assert engine.stats()["prefix_cache_hits"] == 0, (
        "a pruned request must not register or adopt a prefix; a hit means it was served unpruned"
    )


def test_disabling_chunked_prefill_with_a_long_prompt_is_refused_at_submission(tiny_model_dir):
    """A prompt that cannot fit one step is unschedulable.  It is refused when submitted: raising
    from inside the scheduler left the request at the head of the queue and made every later step
    fail, so an unrelated request would have taken the whole engine down."""
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, enable_chunked_prefill=False,
                                        max_num_batched_tokens=8)
    with pytest.raises(ValueError, match="chunked prefill is disabled"):
        engine.add_request(list(range(1, 40)), SamplingParams(temperature=0.0, max_tokens=2,
                                                             ignore_eos=True))
    # The engine is still healthy: a request that does fit is served normally.
    engine.add_request([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
    assert len(engine.run_until_complete()) == 1


def test_run_until_complete_reports_only_this_call(tiny_model_dir):
    """`self.finished` accumulates, so returning it wholesale re-reported earlier requests."""
    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    engine.add_request(list(range(3, 20)), SamplingParams(temperature=0.0, max_tokens=2,
                                                          ignore_eos=True))
    assert len(engine.run_until_complete()) == 1
    engine.add_request(list(range(30, 47)), SamplingParams(temperature=0.0, max_tokens=2,
                                                           ignore_eos=True))
    assert len(engine.run_until_complete()) == 1, "the earlier request was reported twice"


def test_kv_storage_dtype_must_match_the_compute_dtype(tiny_model_dir):
    """torch's attention kernels need q/k/v in one dtype.  A narrower cache would need an
    explicit cast that does not exist, so the mismatch is refused up front instead of failing
    inside the first forward pass."""
    model_dir, hf = tiny_model_dir
    cfg = Qwen25VLConfig.from_pretrained(model_dir)
    model = build_model(cfg)
    load_weights(model_dir, model, strict=True)
    ecfg = EngineConfig(num_blocks=8, block_size=16, kv_dtype="float16", dtype=torch.float32)
    with pytest.raises(ValueError, match="KV storage must match"):
        ModelRunner(model, ecfg, device="cpu")


def test_fastv_survives_recompute_preemption_without_pruning_twice(tiny_model_dir, tiny_inputs):
    """Recompute preemption cannot restore a FastV-pruned prompt: re-running the dense pass would
    prune an already-pruned sequence, so the output would depend on whether a preemption happened.
    A pruned request is therefore preempted by swap, which never re-runs prefill."""
    engine, cfg, model_dir, hf = _stack(tiny_model_dir, enable_fastv=True, fastv_ratio=0.5,
                                        fastv_layer=1, max_num_batched_tokens=512,
                                        preemption="recompute", num_blocks=64, max_num_seqs=2)
    ids = tiny_inputs["input_ids"][0].tolist()
    engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
                       pixel_values=tiny_inputs["pixel_values"],
                       image_grid_thw=tiny_inputs["image_grid_thw"])
    engine.step()  # prefill (and prune) completes; the sequence is now running
    seq = engine.running[0]
    assert seq.extra.get("fastv_will_prune")
    assert engine._preempt_one() is True
    assert seq.status == "swapped", (
        "a FastV sequence was preempted by recompute; re-prefill would prune it a second time"
    )
    outs = engine.run_until_complete()
    assert len(outs) == 1
    assert outs[0].output_token_ids, "the preempted request must still finish"
    # One pruning pass: at least half the submitted image tokens must have survived.
    submitted = sum(1 for t in ids if t == cfg.image_token_id)
    assert int(engine.stats()["vision_tokens_total"]) >= submitted // 2, (
        "the sequence was pruned more than once"
    )


def test_cancel_finishes_a_sequence_and_frees_its_blocks(tiny_model_dir):
    """The serving layer ends the HTTP stream when a stop string appears; without a cancel the
    engine kept generating to max_tokens and cached a prefix nobody would reuse."""
    engine, cfg, model_dir, hf = _stack(tiny_model_dir)
    seq_id = engine.add_request(list(range(3, 25)),
                                SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True))
    engine.step()
    free_before = engine.runner.block_manager.num_free
    assert engine.cancel(seq_id) is True
    assert not engine.has_work(), "cancelling the only request must leave the engine idle"
    assert engine.runner.block_manager.num_free > free_before, "the sequence's blocks were not freed"
    finished = [o for o in engine.finished if o.seq_id == seq_id]
    assert finished and finished[0].finish_reason == "stop"
    assert engine.cancel(seq_id) is False, "cancelling twice must report that nothing was live"


def test_engine_config_rejects_unusable_limits():
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        EngineConfig(max_num_batched_tokens=0)
    with pytest.raises(ValueError, match="max_model_len"):
        EngineConfig(max_model_len=4, block_size=16)
