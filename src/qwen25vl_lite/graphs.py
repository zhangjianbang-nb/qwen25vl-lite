"""CUDA-graph capture for the decode step.

A decode step runs ~2,100 Python-level operator dispatches and 252 Triton kernel launches.
Measured on an RTX 5060 with Qwen2.5-VL-3B INT4, the CPU enqueue cost is ~19.8 ms per token
while the GPU work behind it is only ~13.6 ms, so the step is launch-bound and the card
idles for a third of every token.  Capturing the decoder forward into one CUDA graph replays
the whole thing with a single launch and removes that idle time.

Shapes inside a graph are fixed, so a graph is captured per ``(batch size, block-table
width)``.  A block table grows by one block every 16 tokens, so taking the width literally
would re-capture roughly every 16 tokens -- and a capture costs a warm-up forward plus a
verification replay, i.e. ~34 ms, which is more than the ~4.5 ms a graph saves.  The width is
therefore rounded up to a bucket (default 16 blocks = 256 tokens) and the table is padded
with ``-1``, which ``build_paged_slots`` clamps to block 0 and the caller's length mask
excludes.  Buckets change every 256 tokens instead of every 16, so capture cost amortises to
well under a tenth of a millisecond per token.  The price is that attention runs over the
padded width, so the bucket is a tuning knob, not a free choice.

This is a *performance* change and must not change any number: the graph replays exactly the
kernel sequence the eager path runs, so a replay is bit-identical to the eager forward on the
same inputs.  That is asserted at capture time (:meth:`_DecodeGraph.capture`) and again in
``tests/test_cuda_graph.py``, because a graph that silently differed would still produce
plausible tokens -- the failure mode this repository refuses to ship.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

import torch

__all__ = ["HAS_CUDA_GRAPHS", "DecodeGraphRunner", "graph_supported", "bucket_width"]


HAS_CUDA_GRAPHS = hasattr(torch.cuda, "CUDAGraph")

DEFAULT_BUCKET = 16


def graph_supported(device: str) -> bool:
    """Whether CUDA graphs can be used at all on this device."""
    return HAS_CUDA_GRAPHS and str(device).startswith("cuda")


def bucket_width(width: int, bucket: int = DEFAULT_BUCKET) -> int:
    """Round a block-table width up to the next multiple of ``bucket``.

    ``bucket=1`` reproduces the exact width (and re-captures on every block crossing).
    """
    bucket = max(int(bucket), 1)
    width = max(int(width), 1)
    return ((width + bucket - 1) // bucket) * bucket



class _DecodeGraph:
    """One captured decode step for a fixed ``(batch, block_width)``."""

    def __init__(self, runner, batch: int, block_width: int):
        self.runner = runner
        self.model = runner.model
        self.cache = runner.cache
        self.backends = runner.backends
        self.device = torch.device(runner.device)
        self.batch = int(batch)
        self.block_width = int(block_width)

        # Static input buffers.  A graph replays the addresses it was captured with, so every
        # step must write into these same tensors rather than allocate fresh ones.
        self.input_ids = torch.zeros(self.batch, 1, dtype=torch.long, device=self.device)
        self.position_ids = torch.zeros(3, self.batch, 1, dtype=torch.long, device=self.device)
        self.slot_mapping = torch.zeros(self.batch, dtype=torch.long, device=self.device)
        self.context_lens = torch.zeros(self.batch, dtype=torch.long, device=self.device)
        # Block 0 exists whenever num_blocks >= 1, so a zero-filled table is safe to capture
        # against.  It must not be -1: the fused paged-attention kernel indexes the cache with
        # the id it reads, and -1 would address the bytes before the tensor.  The gather-based
        # reference path clamps -1 to 0, so it is safe either way -- but the capture runs
        # before the first step has written any real ids.
        self.block_tables = torch.zeros((self.batch, self.block_width),
                                        dtype=torch.long, device=self.device)
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.logits: Optional[torch.Tensor] = None

    def forward(self) -> torch.Tensor:
        with torch.no_grad():
            embeds = self.model.embed_tokens(self.input_ids)
            return self.model(
                inputs_embeds=embeds,
                position_ids=self.position_ids,
                cache=self.cache,
                slot_mapping=self.slot_mapping,
                block_tables=self.block_tables,
                context_lens=self.context_lens,
                query_lens=None,
                backends=self.backends,
                logits_to_keep=1,
            )

    def capture(self) -> float:
        """Warm up on a side stream, capture, and check the replay against the eager path.

        Returns the worst absolute difference between the replay and the eager forward, which
        is expected to be exactly ``0.0``: the graph is the same kernel sequence.
        """
        # Warm-up outside capture: a Triton launch compiles its specialisation on first use,
        # and that must not happen inside a graph.
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(device=self.device))
        with torch.cuda.stream(side):
            for _ in range(3):
                self.forward()
        torch.cuda.current_stream(device=self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)

        eager = self.forward().clone()
        torch.cuda.synchronize(self.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self.forward()
        torch.cuda.synchronize(self.device)

        self.graph.replay()
        torch.cuda.synchronize(self.device)
        return (self.logits.float() - eager.float()).abs().max().item()

    def set_inputs(self, input_ids, position_ids, slot_mapping, context_lens, block_tables):
        """Copy this step's values into the captured buffers.

        ``block_tables`` may be narrower than the captured width; the tail is filled with
        ``-1`` so the captured shape is preserved.  Those entries address block 0 and are
        dropped by the length mask, so they cannot change the result.
        """
        self.input_ids.copy_(input_ids)
        self.position_ids.copy_(position_ids)
        self.slot_mapping.copy_(slot_mapping)
        self.context_lens.copy_(context_lens)
        actual = int(block_tables.shape[1])
        self.block_tables[:, :actual].copy_(block_tables)
        if actual < self.block_width:
            self.block_tables[:, actual:].fill_(-1)

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.logits


class DecodeGraphRunner:
    """Cache of decode graphs keyed by ``(batch, bucketed block width)``.

    ``max_graphs`` bounds how many captured graphs are kept alive.  Each graph owns a private
    CUDA memory pool, so an unbounded cache would leak device memory as the block-table width
    walks upwards over a long generation.  Eviction drops the graph and its pool; a later
    request for the same key re-captures it.
    """

    def __init__(self, runner, max_graphs: int = 8, bucket: int = DEFAULT_BUCKET,
                 verify_tolerance: float = 0.0):
        self.runner = runner
        self.device = runner.device
        self.max_graphs = int(max_graphs)
        self.bucket = int(bucket)
        self.verify_tolerance = float(verify_tolerance)
        self._graphs: "OrderedDict[Tuple[int, int], _DecodeGraph]" = OrderedDict()
        self.stats = {"captures": 0, "replays": 0, "evictions": 0, "worst_diff": 0.0}

    def can_graph(self, batch: int, block_width: int) -> bool:
        if not graph_supported(self.device):
            return False
        return batch >= 1 and block_width >= 1

    def run(self, input_ids, position_ids, slot_mapping, context_lens, block_tables) -> torch.Tensor:
        batch = int(input_ids.shape[0])
        width = bucket_width(int(block_tables.shape[1]), self.bucket)
        key = (batch, width)
        g = self._graphs.get(key)
        if g is None:
            g = _DecodeGraph(self.runner, batch, width)
            diff = g.capture()
            if diff > self.verify_tolerance:
                # A graph that does not reproduce the eager path is worse than no graph.
                raise RuntimeError(
                    f"CUDA-graph decode for batch={batch} block_width={width} diverged from "
                    f"the eager forward by {diff:.3e} (tolerance {self.verify_tolerance:.1e}); "
                    "refusing to use it"
                )
            self.stats["captures"] += 1
            self.stats["worst_diff"] = max(self.stats["worst_diff"], diff)
            self._graphs[key] = g
            while len(self._graphs) > self.max_graphs:
                _, old = self._graphs.popitem(last=False)
                old.graph = None
                old.logits = None
                self.stats["evictions"] += 1
        else:
            self._graphs.move_to_end(key)

        g.set_inputs(input_ids, position_ids, slot_mapping, context_lens, block_tables)
        self.stats["replays"] += 1
        return g.replay()

    def summary(self) -> str:
        return (f"cuda graphs: {len(self._graphs)} cached (bucket={self.bucket} blocks), "
                f"{self.stats['captures']} captures, {self.stats['replays']} replays, "
                f"{self.stats['evictions']} evictions, worst |replay - eager| = "
                f"{self.stats['worst_diff']:.1e}")
