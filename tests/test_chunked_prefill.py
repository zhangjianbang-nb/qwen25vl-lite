"""Gate A, part 2 -- the execution paths must agree with each other and with HF.

The review of 2026-10-15 singled this out: chunked prefill and paged decode are where
"runs fine, answers wrong" bugs live.  So the tests here are equivalence tests, not smoke
tests:

* single-pass prefill vs chunked prefill at several chunk sizes, including chunk sizes that
  do not divide the block size and prompt lengths that straddle block boundaries;
* paged decode vs HF's own incremental decoding, step by step;
* the prefix cache must reproduce identical logits for a repeated prefix.
"""

from __future__ import annotations

import pytest
import torch

from qwen25vl_lite.config import Qwen25VLConfig
from qwen25vl_lite.engine import EngineConfig, ModelRunner
from qwen25vl_lite.cache import SequenceState, prefix_fingerprint
from qwen25vl_lite.model import build_model, load_weights
from qwen25vl_lite.rope import mm_token_type_ids


def _model_and_cfg(tiny_model_dir):
    model_dir, hf = tiny_model_dir
    cfg = Qwen25VLConfig.from_pretrained(model_dir)
    model = build_model(cfg)
    load_weights(model_dir, model, strict=True)
    model.eval()
    # HF keeps `rope_deltas` as *model* state, so it survives across tests that share the
    # session-scoped fixture.  A stale delta from an image prompt would silently shift every
    # later decode position, so clear it before use.
    hf.model.rope_deltas = None
    return model, cfg, model_dir, hf


def _make_seq(cfg, ids, block_size, image_grid_thw=None, vision_features=None):
    seq = SequenceState(seq_id=0, prompt_token_ids=list(ids), block_size=block_size)
    pos, delta = None, None
    return seq, pos, delta


def _prefill(model, cfg, ids, chunk, block_size, image_grid_thw=None, pixel_values=None):
    """Run a prefill with the given chunk size using a fresh cache; return last logits."""
    ecfg = EngineConfig(
        num_blocks=256,
        block_size=block_size,
        max_num_batched_tokens=max(chunk, 8),
        max_model_len=4096,
        enable_prefix_cache=False,
    )
    runner = ModelRunner(model, ecfg, device="cpu")
    seq = SequenceState(seq_id=0, prompt_token_ids=list(ids), block_size=block_size)
    if pixel_values is not None:
        seq.vision_features = runner.embed_vision(pixel_values, image_grid_thw)
        seq.grid_thw = image_grid_thw
    pos, delta = model.position_ids(torch.tensor([ids], dtype=torch.long), image_grid_thw=image_grid_thw)
    seq.prompt_position_ids = pos
    seq.rope_delta = int(delta[0, 0].item())

    logits = None
    while seq.num_computed_tokens < len(ids):
        start = seq.num_computed_tokens
        end = min(len(ids), start + chunk)
        logits = runner.run_prefill_chunk(seq, start, end)
        seq.num_computed_tokens = end
    return logits, runner, seq


@pytest.mark.parametrize("chunk", [16, 8, 5, 3])
@pytest.mark.parametrize("length", [15, 16, 17, 33, 64])
def test_chunked_prefill_matches_single_pass(tiny_model_dir, tiny_hf_config, chunk, length):
    model, cfg, model_dir, hf = _model_and_cfg(tiny_model_dir)
    ids = [(7 + i) % cfg.vocab_size for i in range(length)]
    mm = mm_token_type_ids(torch.tensor([ids]), cfg.image_token_id, cfg.video_token_id)

    with torch.no_grad():
        ref = hf(input_ids=torch.tensor([ids]), mm_token_type_ids=mm).logits[:, -1, :]
        got, _, _ = _prefill(model, cfg, ids, chunk, block_size=16)
    assert torch.allclose(ref, got[0, -1], atol=1e-4, rtol=1e-3), (
        f"chunk={chunk} length={length}: chunked prefill diverged from a single pass by "
        f"{(ref - got[0, -1]).abs().max().item():.3e}"
    )


def test_chunked_prefill_with_image_matches_hf(tiny_model_dir, tiny_hf_config, tiny_inputs):
    model, cfg, model_dir, hf = _model_and_cfg(tiny_model_dir)
    ids = tiny_inputs["input_ids"][0].tolist()
    mm = mm_token_type_ids(tiny_inputs["input_ids"], cfg.image_token_id, cfg.video_token_id)

    with torch.no_grad():
        ref = hf(
            input_ids=tiny_inputs["input_ids"],
            pixel_values=tiny_inputs["pixel_values"],
            image_grid_thw=tiny_inputs["image_grid_thw"],
            mm_token_type_ids=mm,
        ).logits[:, -1, :]
        got, _, _ = _prefill(
            model, cfg, ids, chunk=8, block_size=16,
            image_grid_thw=tiny_inputs["image_grid_thw"], pixel_values=tiny_inputs["pixel_values"],
        )
    assert torch.allclose(ref, got[0, -1], atol=1e-4, rtol=1e-3), (
        f"chunked multimodal prefill diverged by {(ref - got[0, -1]).abs().max().item():.3e}"
    )


def test_paged_decode_matches_hf_incremental(tiny_model_dir, tiny_hf_config):
    """Step-by-step decode through the paged cache vs HF's own KV cache."""
    model, cfg, model_dir, hf = _model_and_cfg(tiny_model_dir)
    prompt = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    mm = mm_token_type_ids(torch.tensor([prompt]), cfg.image_token_id, cfg.video_token_id)
    n_steps = 6

    # 1) HF reference trajectory: prefill, then feed back the token HF itself would pick.
    with torch.no_grad():
        out = hf(input_ids=torch.tensor([prompt]), mm_token_type_ids=mm, use_cache=True)
        past = out.past_key_values
        ref_logits = [out.logits[:, -1, :]]
        fed = []
        cur = int(out.logits[:, -1, :].argmax(-1).item())
        for _ in range(n_steps):
            fed.append(cur)
            step = hf(input_ids=torch.tensor([[cur]]), past_key_values=past, use_cache=True,
                      mm_token_type_ids=torch.zeros((1, 1), dtype=torch.int32))
            past = step.past_key_values
            ref_logits.append(step.logits[:, -1, :])
            cur = int(step.logits[:, -1, :].argmax(-1).item())

    # 2) The same trajectory through the engine's paged cache.
    ecfg = EngineConfig(num_blocks=256, block_size=16, max_num_batched_tokens=64,
                        max_model_len=4096, enable_prefix_cache=False)
    runner = ModelRunner(model, ecfg, device="cpu")
    seq = SequenceState(seq_id=0, prompt_token_ids=list(prompt), block_size=16)
    pos, delta = model.position_ids(torch.tensor([prompt], dtype=torch.long))
    seq.prompt_position_ids = pos
    seq.rope_delta = int(delta[0, 0].item())

    got = [runner.run_prefill_chunk(seq, 0, len(prompt))[0, -1]]
    seq.num_computed_tokens = len(prompt)
    seq.output_token_ids = [fed[0]]
    for token in fed[1:]:
        got.append(runner.run_decode([seq])[0])
        seq.output_token_ids.append(token)
    # One more step so the logits after the final fed token are covered too.
    got.append(runner.run_decode([seq])[0])

    assert len(got) == len(ref_logits)
    for i, (ref, mine) in enumerate(zip(ref_logits, got)):
        assert torch.allclose(ref[0], mine, atol=1e-4, rtol=1e-3), (
            f"paged decode step {i} diverged by {(ref[0] - mine).abs().max().item():.3e}"
        )


def test_decode_batch_matches_independent_single_sequences(tiny_model_dir):
    """Batching decode must not let one sequence see another sequence's KV padding."""
    model, cfg, model_dir, _hf = _model_and_cfg(tiny_model_dir)
    prompts = [[5, 6, 7, 8, 9, 10], [21, 22, 23], [31, 32, 33, 34, 35, 36, 37, 38]]

    def run(seq_prompts, batched: bool):
        ecfg = EngineConfig(num_blocks=256, block_size=16, max_num_batched_tokens=64,
                            max_model_len=4096, enable_prefix_cache=False)
        runner = ModelRunner(model, ecfg, device="cpu")
        seqs = []
        for i, p in enumerate(seq_prompts):
            s = SequenceState(seq_id=i, prompt_token_ids=list(p), block_size=16)
            pos, delta = model.position_ids(torch.tensor([p], dtype=torch.long))
            s.prompt_position_ids = pos
            s.rope_delta = int(delta[0, 0].item())
            first = runner.run_prefill_chunk(s, 0, len(p))[0, -1]
            s.num_computed_tokens = len(p)
            s.output_token_ids = [int(first.argmax(-1).item())]
            seqs.append(s)

        out = []
        for _ in range(4):
            if batched:
                lg = runner.run_decode(seqs)
                for i, s in enumerate(seqs):
                    s.output_token_ids.append(int(lg[i].argmax(-1).item()))
            else:
                for s in seqs:
                    lg = runner.run_decode([s])
                    s.output_token_ids.append(int(lg[0].argmax(-1).item()))
            out.append([s.output_token_ids[-1] for s in seqs])
        return out

    a = run(prompts, batched=True)
    b = run(prompts, batched=False)
    assert a == b, f"batched decode diverged from sequential decode:\nbatched={a}\nsequential={b}"


def test_prefix_cache_reuses_and_matches_fresh_compute(tiny_model_dir):
    model, cfg, model_dir, _hf = _model_and_cfg(tiny_model_dir)
    ecfg = EngineConfig(num_blocks=256, block_size=16, max_num_batched_tokens=64,
                        max_model_len=4096, enable_prefix_cache=True)
    runner = ModelRunner(model, ecfg, device="cpu")

    prompt = list(range(1, 41))

    def make():
        s = SequenceState(seq_id=0, prompt_token_ids=list(prompt), block_size=16)
        pos, delta = model.position_ids(torch.tensor([prompt], dtype=torch.long))
        s.prompt_position_ids = pos
        s.rope_delta = int(delta[0, 0].item())
        s.prefix_key = prefix_fingerprint(prompt, rope_theta=cfg.text_config.rope_theta,
                                          mrope_section=cfg.text_config.mrope_section)
        return s

    s1 = make()
    logits1 = runner.run_prefill_chunk(s1, 0, len(prompt))
    s1.num_computed_tokens = len(prompt)
    runner.prefix_cache.insert(prompt, s1.block_table[: len(prompt) // 16], s1.prefix_key)

    s2 = make()
    matched, blocks = runner.prefix_cache.match(prompt, s2.prefix_key)
    assert matched == len(prompt) // 16 * 16, f"expected a full-block prefix hit, got {matched}"
    runner.block_manager.fork(blocks)
    s2.block_table = list(blocks)
    s2.num_computed_tokens = matched
    s2.image_cursor = 0
    logits2 = runner.run_prefill_chunk(s2, matched, len(prompt))

    assert torch.allclose(logits1[0, -1], logits2[0, -1], atol=1e-5, rtol=1e-4), (
        "reusing a cached prefix produced different logits than recomputing it"
    )
    assert runner.prefix_cache.stats()["hits"] >= 1


def test_kv_int8_quantisation_stays_close(tiny_model_dir):
    model, cfg, model_dir, _hf = _model_and_cfg(tiny_model_dir)
    prompt = list(range(1, 40))

    def last_logits(quantize_kv: bool):
        ecfg = EngineConfig(num_blocks=256, block_size=16, max_num_batched_tokens=64,
                            max_model_len=4096, enable_prefix_cache=False, quantize_kv=quantize_kv)
        runner = ModelRunner(model, ecfg, device="cpu")
        s = SequenceState(seq_id=0, prompt_token_ids=list(prompt), block_size=16)
        pos, delta = model.position_ids(torch.tensor([prompt], dtype=torch.long))
        s.prompt_position_ids = pos
        s.rope_delta = int(delta[0, 0].item())
        out = runner.run_prefill_chunk(s, 0, len(prompt))
        s.num_computed_tokens = len(prompt)
        s.output_token_ids = [int(out[0, -1].argmax(-1).item())]
        return runner.run_decode([s])

    ref = last_logits(False)
    q = last_logits(True)
    rel = (ref - q).abs().max() / ref.abs().max()
    assert rel < 0.05, f"INT8 KV changed decode logits by {rel.item()*100:.2f}% (relative)"
