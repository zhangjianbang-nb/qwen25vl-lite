"""CUDA-graph decode: opt-in, and bit-identical to the eager decode step.

A graph that silently differed would still emit fluent tokens, so the claim is stronger than
"close": the captured graph replays exactly the kernel sequence the eager path runs, and
these tests assert token-for-token equality through the whole engine.
"""

from __future__ import annotations

import pytest
import torch

from qwen25vl_lite.engine import EngineConfig, LLMEngine, ModelRunner
from qwen25vl_lite.graphs import bucket_width, graph_supported
from qwen25vl_lite.model import build_model, load_weights
from qwen25vl_lite.sampling import SamplingParams

NEEDS_GPU = not torch.cuda.is_available()
GPU_REASON = "needs CUDA"


def test_bucket_width_rounds_up():
    assert bucket_width(32, 16) == 32
    assert bucket_width(33, 16) == 48
    assert bucket_width(1, 16) == 16
    # bucket=1 reproduces the exact width, which re-captures on every block crossing.
    assert bucket_width(33, 1) == 33


def test_graph_supported_rejects_cpu():
    assert graph_supported("cpu") is False


@pytest.mark.skipif(NEEDS_GPU, reason=GPU_REASON)
def test_cuda_graph_requires_a_cuda_device():
    """Asking for graphs on CPU must fail loudly, not silently fall back."""
    with pytest.raises(ValueError):
        EngineConfig(cuda_graph=True, device="cpu")


def _engine(model_dir, device, cuda_graph):
    from qwen25vl_lite.config import Qwen25VLConfig

    cfg = Qwen25VLConfig.from_pretrained(model_dir)
    model = build_model(cfg)
    load_weights(model_dir, model, device=device, dtype=torch.float32)
    model.eval()
    ecfg = EngineConfig(
        num_blocks=64, block_size=16, max_num_seqs=2, max_num_batched_tokens=512,
        max_model_len=512, device=device, dtype=torch.float32,
        cuda_graph=cuda_graph, cuda_graph_bucket=16,
    )
    runner = ModelRunner(model, ecfg, device=device)
    return LLMEngine(runner, ecfg), runner


@pytest.mark.skipif(NEEDS_GPU, reason=GPU_REASON)
def test_graph_decode_produces_the_same_tokens_as_eager(tiny_model_dir):
    model_dir, _ = tiny_model_dir
    prompt = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] * 3

    eager_engine, _ = _engine(model_dir, "cuda", False)
    eager_engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=24,
                                                    ignore_eos=True))
    eager_out = eager_engine.run_until_complete()

    graph_engine, graph_runner = _engine(model_dir, "cuda", True)
    graph_engine.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=24,
                                                    ignore_eos=True))
    graph_out = graph_engine.run_until_complete()

    assert graph_runner.decode_graphs is not None
    assert graph_runner.decode_graphs.stats["replays"] > 0
    assert graph_runner.decode_graphs.stats["captures"] >= 1
    # The capture-time self-check: every graph must have matched the eager forward exactly.
    assert graph_runner.decode_graphs.stats["worst_diff"] == 0.0
    assert graph_out[0].output_token_ids == eager_out[0].output_token_ids


@pytest.mark.skipif(NEEDS_GPU, reason=GPU_REASON)
def test_graph_is_reused_while_the_block_table_width_is_unchanged(tiny_model_dir):
    """Bucketing must make captures rare: 24 tokens inside one bucket is one capture."""
    model_dir, _ = tiny_model_dir
    engine, runner = _engine(model_dir, "cuda", True)
    engine.add_request([10, 11, 12, 13, 14, 15, 16, 17] * 2,
                       SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=True))
    engine.run_until_complete()
    stats = runner.decode_graphs.stats
    assert stats["replays"] >= 15
    assert stats["captures"] <= 2, f"expected rare captures, got {stats['captures']}"
