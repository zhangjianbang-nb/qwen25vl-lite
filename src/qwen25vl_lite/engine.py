"""The inference engine: scheduling, paged execution, prefix reuse, preemption.

Design notes that matter for correctness:

* **Continuous batching** means prefill and decode are scheduled in the same iteration and a
  new request can join mid-flight.  Decode is batched across sequences; prefill is executed
  one sequence per chunk.  Batching heterogeneous-length prefills together is a documented
  TODO rather than a claim -- it needs padded masks and buys little on a single 8 GB card.
* **Chunked prefill** runs a long prompt in pieces of ``max_num_batched_tokens``.  The
  absolute position of every chunk matters: a chunk covering tokens ``[1000, 1512)`` must
  use positions 1000..1511 and a causal mask offset by ``context_len - query_len``.  Getting
  this wrong produces plausible-looking output, so ``tests/test_chunked_prefill.py`` compares
  chunked prefill against a single-pass prefill for several chunk sizes and boundary lengths.
* **Prefix caching** keys on token ids *plus* a fingerprint of the vision embeddings and the
  RoPE configuration, because ``<|image_pad|>`` runs of equal length are indistinguishable by
  id alone.
* **Preemption** frees or swaps a victim's blocks when the pool runs dry.  The victim's KV is
  never silently discarded without either being copied to host memory (swap) or having its
  computed-token counter reset (recompute).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .attention import BackendReport, pick_backends, probe_backends
from .cache import BlockManager, PagedKVCache, RadixPrefixCache, SequenceState, prefix_fingerprint
from .config import Qwen25VLConfig
from .model import Qwen25VLLite
from .rope import text_position_ids
from .sampling import SamplingParams, sample_token
from .vision_budget import TIERS, VisionTier, fastv_select

__all__ = ["EngineConfig", "RequestOutput", "ModelRunner", "LLMEngine", "EngineStats"]


@dataclass
class EngineConfig:
    """Scheduling and memory configuration."""

    num_blocks: int = 2048
    block_size: int = 16
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 4096
    max_model_len: int = 32768
    enable_prefix_cache: bool = True
    enable_chunked_prefill: bool = True
    quantize_kv: bool = False
    kv_dtype: str = "float16"
    vision_tier: str = "full"
    enable_fastv: bool = False
    fastv_ratio: float = 0.5
    fastv_layer: int = 2
    preemption: str = "swap"
    attn_backends: Optional[List[str]] = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.num_blocks <= 0:
            raise ValueError("block_size and num_blocks must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.preemption not in ("swap", "recompute"):
            raise ValueError("preemption must be 'swap' or 'recompute'")
        if self.enable_fastv and self.fastv_layer >= 1 and self.fastv_ratio <= 0:
            raise ValueError("enable_fastv requires a positive fastv_ratio")


@dataclass
class RequestOutput:
    """What the caller gets back for one finished request."""

    seq_id: int
    prompt_token_ids: List[int]
    output_token_ids: List[int]
    finish_reason: str
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)


@dataclass
class EngineStats:
    steps: int = 0
    prefill_chunks: int = 0
    decode_steps: int = 0
    preemptions: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_tokens_saved: int = 0
    vision_tokens_total: int = 0

    def summary(self) -> str:
        return (
            f"steps={self.steps} prefill_chunks={self.prefill_chunks} decode_steps={self.decode_steps} "
            f"preemptions={self.preemptions} prefix_hits={self.prefix_cache_hits} "
            f"prefix_tokens_saved={self.prefix_cache_tokens_saved} vision_tokens={self.vision_tokens_total}"
        )


class ModelRunner:
    """Owns the KV cache, the block pool, the prefix cache, and the forward passes."""

    def __init__(self, model: Qwen25VLLite, engine_cfg: EngineConfig, device: Optional[str] = None):
        self.model = model
        self.cfg: Qwen25VLConfig = model.cfg
        self.ecfg = engine_cfg
        self.device = device or engine_cfg.device

        text = self.cfg.text_config
        self.cache = PagedKVCache(
            num_layers=text.num_hidden_layers,
            num_blocks=engine_cfg.num_blocks,
            block_size=engine_cfg.block_size,
            num_kv_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            dtype=engine_cfg.dtype,
            device=self.device,
            quantize=engine_cfg.quantize_kv,
        )
        self.block_manager = BlockManager(engine_cfg.num_blocks, engine_cfg.block_size)
        self.prefix_cache = RadixPrefixCache(self.block_manager)
        self.stats = EngineStats()

        if engine_cfg.attn_backends:
            self.backends = list(engine_cfg.attn_backends)
            self.backend_report = None
        else:
            report = probe_backends(
                device=self.device,
                dtype=engine_cfg.dtype,
                head_dim=text.head_dim,
                num_heads=text.num_attention_heads,
                num_kv_heads=text.num_key_value_heads,
            )
            self.backends = report.chosen
            self.backend_report: Optional[BackendReport] = report

    # -- helpers ---------------------------------------------------------------------
    def tier(self, name: Optional[str] = None) -> VisionTier:
        return TIERS[name or self.ecfg.vision_tier]

    @property
    def fastv_enabled(self) -> bool:
        """True when pruning is requested either explicitly or by the selected tier.

        The `pruned` tier carries ``fastv_ratio=0.5``; selecting it must actually prune, or the
        tier would be a label that does nothing.
        """
        return self.ecfg.enable_fastv or self.tier().fastv_ratio > 0.0

    @property
    def fastv_ratio(self) -> float:
        return self.ecfg.fastv_ratio if self.ecfg.enable_fastv else self.tier().fastv_ratio

    @property
    def fastv_layer(self) -> int:
        return self.ecfg.fastv_layer if self.ecfg.enable_fastv else self.tier().fastv_layer

    def _block_table_tensor(self, seqs: Sequence[SequenceState]) -> torch.Tensor:
        width = max((len(s.block_table) for s in seqs), default=1)
        table = torch.full((len(seqs), width), -1, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            table[i, : len(s.block_table)] = torch.tensor(s.block_table, dtype=torch.long, device=self.device)
        return table

    def ensure_blocks(self, seq: SequenceState, upto: int) -> None:
        """Grow ``seq.block_table`` so positions ``[0, upto)`` are addressable."""
        need = seq.num_blocks_needed(upto)
        missing = need - len(seq.block_table)
        if missing <= 0:
            return
        seq.block_table.extend(self.block_manager.allocate(missing))

    def _chunk_inputs(self, seq: SequenceState, start: int, end: int):
        """Embeddings, position ids and slot mapping for prompt tokens ``[start, end)``."""
        ids = seq.prompt_token_ids[start:end]
        id_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        embeds = self.model.embed_tokens(id_tensor)

        n_img = sum(1 for t in ids if t == self.cfg.image_token_id)
        if n_img:
            if seq.vision_features is None:
                raise ValueError(f"sequence {seq.seq_id} contains image tokens but has no vision features")
            feats = seq.vision_features[seq.image_cursor : seq.image_cursor + n_img]
            if feats.shape[0] != n_img:
                raise ValueError(
                    f"sequence {seq.seq_id} needs {n_img} image feature rows but only "
                    f"{feats.shape[0]} remain"
                )
            seq.image_cursor += n_img
            mask = (id_tensor == self.cfg.image_token_id).unsqueeze(-1)
            embeds = embeds.masked_scatter(mask, feats.to(embeds.dtype))

        pos = seq.prompt_position_ids[:, :, start:end]
        slots = torch.tensor(seq.slot_mapping(start, end), dtype=torch.long, device=self.device)
        return embeds, pos, slots

    # -- execution -------------------------------------------------------------------
    def run_prefill_chunk(self, seq: SequenceState, start: int, end: int) -> torch.Tensor:
        """Run one prefill chunk and return the logits of its final position."""
        self.ensure_blocks(seq, end)
        embeds, pos, slots = self._chunk_inputs(seq, start, end)
        ctx = torch.tensor([end], dtype=torch.long, device=self.device)
        qlen = torch.tensor([end - start], dtype=torch.long, device=self.device)
        logits = self.model(
            inputs_embeds=embeds,
            position_ids=pos,
            cache=self.cache,
            slot_mapping=slots,
            block_tables=self._block_table_tensor([seq]),
            context_lens=ctx,
            query_lens=qlen,
            backends=self.backends,
            logits_to_keep=1,
        )
        return logits

    def run_prefill_fastv(self, seq: SequenceState) -> torch.Tensor:
        """Prefill with FastV-style pruning of low-attention image tokens.

        Layers ``0..L`` run densely over the full prompt (so importance is measured with full
        context), then the low-share image tokens are dropped and only the survivors are
        written to the paged cache.  Layers ``L+1..`` then run over the shorter sequence.

        The whole prompt must fit in one chunk: pruning after a *partial* prefill would need
        the importance signal from tokens that have not been processed yet.
        """
        prompt_len = len(seq.prompt_token_ids)
        if prompt_len > self.ecfg.max_num_batched_tokens:
            raise ValueError(
                f"FastV prefill needs the whole prompt in one chunk ({prompt_len} > "
                f"{self.ecfg.max_num_batched_tokens}); raise max_num_batched_tokens or disable FastV"
            )
        image_positions = [i for i, t in enumerate(seq.prompt_token_ids) if t == self.cfg.image_token_id]
        if not image_positions:
            return self.run_prefill_chunk(seq, 0, prompt_len)

        self.ensure_blocks(seq, prompt_len)
        embeds, pos, _ = self._chunk_inputs(seq, 0, prompt_len)

        captured: Dict[str, torch.Tensor] = {}

        def hook(_layer_idx: int, hidden: torch.Tensor, probs) -> torch.Tensor:
            if probs is not None:
                captured["probs"] = probs
            return hidden

        kv: list = []
        layer = self.fastv_layer
        hidden = self.model.language_model(
            embeds,
            pos,
            attn_mask=None,
            backends=self.backends,
            capture_probs_at=layer,
            stop_layer=layer + 1,
            kv_collector=kv,
            layer_hook=hook,
        )
        if "probs" not in captured:
            raise RuntimeError(
                f"FastV expected attention probabilities from layer {layer} but none were captured"
            )
        probs = captured["probs"]

        keep = fastv_select(probs, image_positions, self.fastv_ratio)
        keep_mask = torch.ones(prompt_len, dtype=torch.bool)
        ordinal = {pos_i: idx for idx, pos_i in enumerate(image_positions)}
        for i in image_positions:
            if i not in set(int(x) for x in keep):
                keep_mask[i] = False

        kept_ordinals = [ordinal[i] for i in sorted(int(x) for x in keep)]
        new_ids = [t for t, k in zip(seq.prompt_token_ids, keep_mask.tolist()) if k]

        seq.prompt_token_ids = new_ids
        seq.prompt_position_ids = pos[:, :, keep_mask]
        if seq.vision_features is not None:
            seq.vision_features = seq.vision_features[kept_ordinals]
            seq.image_cursor = seq.vision_features.shape[0]
        self._shrink_blocks(seq, len(new_ids))

        n_heads = self.cache.num_kv_heads
        head_dim = self.cache.head_dim
        slots = torch.tensor(seq.slot_mapping(0, len(new_ids)), dtype=torch.long, device=self.device)
        for layer_idx, (k, v) in enumerate(kv):
            kk = k[0][:, keep_mask].transpose(0, 1).contiguous().reshape(-1, n_heads, head_dim)
            vv = v[0][:, keep_mask].transpose(0, 1).contiguous().reshape(-1, n_heads, head_dim)
            self.cache.write(layer_idx, kk, vv, slots)

        ctx = torch.tensor([len(new_ids)], dtype=torch.long, device=self.device)
        out = self.model.language_model(
            hidden[0][keep_mask].unsqueeze(0),
            seq.prompt_position_ids,
            cache=self.cache,
            slot_mapping=slots,
            block_tables=self._block_table_tensor([seq]),
            context_lens=ctx,
            query_lens=ctx,
            backends=self.backends,
            start_layer=layer + 1,
        )
        return self.model.lm_head(out[:, -1:, :])

    def _shrink_blocks(self, seq: SequenceState, length: int) -> None:
        need = seq.num_blocks_needed(length)
        while len(seq.block_table) > need:
            extra = seq.block_table.pop()
            self.block_manager.free_blocks([extra])

    def run_decode(self, seqs: Sequence[SequenceState]) -> torch.Tensor:
        """One decode step for a batch of sequences; returns ``[batch, vocab]`` logits.

        Blocks are grown here rather than relying on the caller, so a direct ``run_decode``
        cannot index past the end of a sequence's block table.
        """
        for s in seqs:
            self.ensure_blocks(s, s.num_computed_tokens + 1)

        id_tensor = torch.tensor([[s.output_token_ids[-1]] for s in seqs], dtype=torch.long, device=self.device)
        embeds = self.model.embed_tokens(id_tensor)

        slots = []
        context_lens = []
        pos_parts = []
        for s in seqs:
            slots.append(s.slot_mapping(s.num_computed_tokens, s.num_computed_tokens + 1)[0])
            context_lens.append(s.num_computed_tokens + 1)
            pos_parts.append(text_position_ids(1, s.num_computed_tokens, s.rope_delta, self.device))

        pos = torch.cat(pos_parts, dim=1) if len(pos_parts) > 1 else pos_parts[0]
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=self.device)
        ctx = torch.tensor(context_lens, dtype=torch.long, device=self.device)

        logits = self.model(
            inputs_embeds=embeds,
            position_ids=pos,
            cache=self.cache,
            slot_mapping=slot_tensor,
            block_tables=self._block_table_tensor(seqs),
            context_lens=ctx,
            query_lens=None,
            backends=self.backends,
            logits_to_keep=1,
        )
        for s in seqs:
            s.num_computed_tokens += 1
        return logits[:, -1, :]

    def embed_vision(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.embed_images(pixel_values, grid_thw, backends=self.backends)

    def free(self, seq: SequenceState) -> None:
        if seq.block_table:
            self.block_manager.free_blocks(seq.block_table)
            seq.block_table = []


class LLMEngine:
    """Iteration-level scheduler on top of :class:`ModelRunner`."""

    def __init__(self, runner: ModelRunner, engine_cfg: Optional[EngineConfig] = None):
        self.runner = runner
        self.ecfg = engine_cfg or runner.ecfg
        self.waiting: List[SequenceState] = []
        self.running: List[SequenceState] = []
        self.swapped: List[SequenceState] = []
        self.finished: List[RequestOutput] = []
        self._next_id = 0
        self._start_time = time.time()
        # Wall-clock time of the first emitted token per sequence.  TTFT is a property of the
        # request, not of the batch it happened to share, so it is recorded at emission rather
        # than inferred afterwards from a batch-level timer.
        self._first_token_at: Dict[int, float] = {}

    # -- request lifecycle -------------------------------------------------------------
    def add_request(
        self,
        prompt_token_ids: Sequence[int],
        sampling_params: SamplingParams,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        priority: int = 0,
    ) -> int:
        """Register a request and return its sequence id."""
        ids = list(int(t) for t in prompt_token_ids)
        if not ids:
            raise ValueError("prompt is empty")
        if len(ids) > self.ecfg.max_model_len:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds max_model_len {self.ecfg.max_model_len}")

        seq = SequenceState(
            seq_id=self._next_id,
            prompt_token_ids=ids,
            block_size=self.ecfg.block_size,
            sampling_params=vars(sampling_params).copy(),
            priority=priority,
        )
        self._next_id += 1

        n_image_tokens = sum(1 for t in ids if t == self.runner.cfg.image_token_id)
        if pixel_values is not None:
            feats = self.runner.embed_vision(pixel_values, image_grid_thw)
            if feats.shape[0] != n_image_tokens:
                raise ValueError(
                    f"vision features ({feats.shape[0]} rows) do not match image placeholders "
                    f"({n_image_tokens}) in the prompt"
                )
            seq.vision_features = feats
            seq.grid_thw = image_grid_thw
        elif n_image_tokens:
            raise ValueError("prompt contains image placeholders but no pixel_values were supplied")

        pos, delta = self.runner.model.position_ids(
            torch.tensor([ids], dtype=torch.long), image_grid_thw=image_grid_thw
        )
        seq.prompt_position_ids = pos
        seq.rope_delta = int(delta[0, 0].item())
        seq.extra["n_image_tokens"] = n_image_tokens

        seq.prefix_key = prefix_fingerprint(
            ids,
            image_features=seq.vision_features,
            grid_thw=image_grid_thw,
            rope_theta=self.runner.cfg.text_config.rope_theta,
            mrope_section=self.runner.cfg.text_config.mrope_section,
        )
        self.waiting.append(seq)
        return seq.seq_id

    def _adopt_prefix(self, seq: SequenceState) -> None:
        """Reuse a cached prefix if one matches (token ids *and* vision fingerprint)."""
        if not self.ecfg.enable_prefix_cache:
            return
        matched, blocks = self.runner.prefix_cache.match(seq.prompt_token_ids, seq.prefix_key)
        matched = min(matched, len(seq.prompt_token_ids) - 1)  # keep >= 1 token to prefill
        matched -= matched % self.ecfg.block_size
        if matched <= 0:
            return
        blocks = blocks[: matched // self.ecfg.block_size]
        seq.block_table = list(blocks)
        self.runner.block_manager.fork(blocks)
        seq.cached_tokens = matched
        seq.num_computed_tokens = matched
        seq.image_cursor = sum(1 for t in seq.prompt_token_ids[:matched] if t == self.runner.cfg.image_token_id)
        self.runner.stats.prefix_cache_hits += 1
        self.runner.stats.prefix_cache_tokens_saved += matched

    def _register_prefix(self, seq: SequenceState) -> None:
        if not self.ecfg.enable_prefix_cache:
            return
        n_blocks = len(seq.prompt_token_ids) // self.ecfg.block_size
        if n_blocks <= 0:
            return
        blocks = seq.block_table[:n_blocks]
        if len(blocks) < n_blocks:
            return
        self.runner.prefix_cache.insert(seq.prompt_token_ids, blocks, seq.prefix_key)

    # -- scheduling --------------------------------------------------------------------
    def _num_free_blocks(self) -> int:
        return self.runner.block_manager.num_free

    def _schedule(self) -> Tuple[List[SequenceState], int]:
        """Pick the prefill work for this step; returns ``(ready_seqs, tokens_budget_left)``.

        Two passes, because a chunked prefill spans several iterations: first continue the
        sequences already admitted but not fully prefilled, then admit new arrivals with
        whatever budget is left.  Forgetting the first pass is how a scheduler deadlocks with
        work still queued.
        """
        budget = self.ecfg.max_num_batched_tokens
        ready: List[SequenceState] = []

        # Pass 1: unfinished prefills already in `running`.
        for seq in list(self.running):
            if budget <= 0:
                break
            remaining = len(seq.prompt_token_ids) - seq.num_computed_tokens
            if remaining <= 0:
                continue
            if self.ecfg.enable_chunked_prefill:
                chunk = min(remaining, budget)
            elif remaining <= budget:
                chunk = remaining
            else:
                continue
            try:
                self.runner.ensure_blocks(seq, seq.num_computed_tokens + chunk)
            except MemoryError:
                if not self._preempt_one():
                    return ready, budget
                continue
            seq.extra["chunk"] = chunk
            ready.append(seq)
            budget -= chunk

        # Pass 2: admit new requests while there is room and budget.
        while self.waiting and len(self.running) < self.ecfg.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            self._adopt_prefix(seq)
            remaining = len(seq.prompt_token_ids) - seq.num_computed_tokens
            if remaining <= 0:
                self.waiting.pop(0)
                self.running.append(seq)
                continue
            if self.ecfg.enable_chunked_prefill:
                chunk = min(remaining, budget)
            elif remaining <= budget:
                chunk = remaining
            else:
                break

            try:
                self.runner.ensure_blocks(seq, seq.num_computed_tokens + chunk)
            except MemoryError:
                if not self._preempt_one():
                    break
                continue

            self.waiting.pop(0)
            self.running.append(seq)
            seq.extra["chunk"] = chunk
            ready.append(seq)
            budget -= chunk

        return ready, budget

    def _preempt_one(self) -> bool:
        """Free blocks by evicting the least valuable running sequence."""
        if not self.running:
            return False
        # Prefer to evict the newest, lowest-priority sequence: it has the least work invested.
        victim = min(self.running, key=lambda s: (s.priority, -s.arrival))
        self.running.remove(victim)
        self.runner.stats.preemptions += 1
        if self.ecfg.preemption == "swap":
            victim.swap_payload = self.runner.cache.swap_out(victim.block_table)
            victim.status = "swapped"
            self.swapped.append(victim)
        else:
            # Recompute mode: the sequence goes back to the waiting queue, not into limbo.
            victim.num_computed_tokens = 0
            victim.cached_tokens = 0
            victim.image_cursor = 0
            victim.status = "waiting"
            self.waiting.insert(0, victim)
        self.runner.free(victim)
        return True

    def _resume_swapped(self) -> None:
        while self.swapped and len(self.running) < self.ecfg.max_num_seqs:
            seq = self.swapped[0]
            payload = seq.swap_payload
            if payload is None:
                self.swapped.pop(0)
                continue
            n_blocks = self.runner.cache.swap_block_count(payload)
            try:
                seq.block_table = self.runner.block_manager.allocate(n_blocks)
            except MemoryError:
                return
            self.runner.cache.swap_in(seq.block_table, payload)
            seq.swap_payload = None
            seq.status = "running"
            self.swapped.pop(0)
            self.running.append(seq)

    # -- stepping ---------------------------------------------------------------------
    def step(self) -> List[RequestOutput]:
        """Run one scheduling iteration; returns any requests that finished."""
        done: List[RequestOutput] = []
        self._resume_swapped()

        ready, _budget = self._schedule()
        running_ids = {id(s) for s in self.running}
        for seq in ready:
            if id(seq) not in running_ids:
                # A later MemoryError preempted this sequence; it will be rescheduled.
                seq.extra.pop("chunk", None)
                continue
            start = seq.num_computed_tokens
            chunk = int(seq.extra.pop("chunk", self.ecfg.max_num_batched_tokens))
            end = min(len(seq.prompt_token_ids), start + chunk)
            if end <= start:
                continue
            try:
                if self.runner.fastv_enabled and start == 0:
                    logits = self.runner.run_prefill_fastv(seq)
                    end = len(seq.prompt_token_ids)
                else:
                    logits = self.runner.run_prefill_chunk(seq, start, end)
            except MemoryError:
                # The pool shrank between scheduling and execution; preempt and retry later.
                if self._preempt_one():
                    continue
                raise
            seq.num_computed_tokens = end
            self.runner.stats.prefill_chunks += 1
            if start == 0:
                self.runner.stats.vision_tokens_total += seq.extra.get("n_image_tokens", 0)

            if seq.num_computed_tokens >= len(seq.prompt_token_ids):
                seq.status = "running"
                self._register_prefix(seq)
                token = sample_token(logits[0, -1], self._params(seq), self._generator(seq))
                self._accept(seq, token, done)

        decode_seqs = [
            s for s in self.running
            if s.status == "running" and s.num_computed_tokens >= len(s.prompt_token_ids) and s.output_token_ids
        ]
        while decode_seqs:
            try:
                logits = self.runner.run_decode(decode_seqs)
                break
            except MemoryError:
                if not self._preempt_one():
                    raise
                decode_seqs = [
                    s for s in self.running
                    if s.status == "running" and s.num_computed_tokens >= len(s.prompt_token_ids)
                    and s.output_token_ids
                ]
        if decode_seqs:
            self.runner.stats.decode_steps += 1
            for i, s in enumerate(decode_seqs):
                token = sample_token(logits[i], self._params(s), self._generator(s))
                self._accept(s, token, done)

        self.runner.stats.steps += 1
        self.finished.extend(done)
        return done

    def _params(self, seq: SequenceState) -> SamplingParams:
        return SamplingParams(**seq.sampling_params)

    def _generator(self, seq: SequenceState) -> Optional[torch.Generator]:
        seed = seq.sampling_params.get("seed")
        if seed is None:
            return None
        gen = torch.Generator()
        gen.manual_seed(int(seed) + len(seq.output_token_ids))
        return gen

    def _accept(self, seq: SequenceState, token: int, done: List[RequestOutput]) -> None:
        params = self._params(seq)
        seq.output_token_ids.append(int(token))
        if len(seq.output_token_ids) == 1:
            self._first_token_at[seq.seq_id] = time.time()
        reason: Optional[str] = None
        if int(token) == self.runner.cfg.eos_token_id and not params.ignore_eos:
            reason = "stop"
        elif int(token) in set(params.stop_token_ids or ()):
            reason = "stop"
        elif len(seq.output_token_ids) >= params.max_tokens:
            reason = "length"
        elif len(seq.prompt_token_ids) + len(seq.output_token_ids) >= self.ecfg.max_model_len:
            reason = "length"

        if reason:
            seq.status = "finished"
            seq.finish_reason = reason
            self.runner.free(seq)
            if seq in self.running:
                self.running.remove(seq)
            done.append(
                RequestOutput(
                    seq_id=seq.seq_id,
                    prompt_token_ids=list(seq.prompt_token_ids),
                    output_token_ids=list(seq.output_token_ids),
                    finish_reason=reason,
                    metrics={
                        "cached_tokens": float(seq.cached_tokens),
                        "prompt_tokens": float(len(seq.prompt_token_ids)),
                        "completion_tokens": float(len(seq.output_token_ids)),
                        "ttft_s": max(0.0, self._first_token_at.get(seq.seq_id, seq.arrival) - seq.arrival),
                        "elapsed_s": time.time() - seq.arrival,
                    },
                )
            )

    # -- driver ------------------------------------------------------------------------
    def has_work(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    def run_until_complete(self, max_steps: int = 100_000, max_stall: int = 64) -> List[RequestOutput]:
        """Drive the engine until every request finishes.

        ``max_stall`` guards against a scheduler that keeps iterating without computing
        anything: if the total number of computed tokens does not change for that many
        consecutive steps, the engine raises instead of spinning forever.  Silent spinning is
        the failure mode that looks like "the server is just slow".
        """
        steps = 0
        stall = 0
        last_progress = self._progress_marker()
        while self.has_work() and steps < max_steps:
            self.step()
            steps += 1
            marker = self._progress_marker()
            if marker == last_progress:
                stall += 1
                if stall >= max_stall:
                    raise RuntimeError(
                        f"scheduler made no progress for {max_stall} steps "
                        f"(waiting={len(self.waiting)} running={len(self.running)} swapped={len(self.swapped)})"
                    )
            else:
                stall = 0
                last_progress = marker
        if self.has_work():
            raise RuntimeError(f"engine did not finish within {max_steps} steps")
        return self.finished

    def _progress_marker(self) -> tuple:
        return (
            len(self.waiting),
            len(self.running),
            len(self.swapped),
            len(self.finished),
            sum(s.num_computed_tokens + len(s.output_token_ids) for s in self.running),
        )

    def stats(self) -> Dict[str, float]:
        pc = self.runner.prefix_cache.stats()
        return {
            **vars(self.runner.stats),
            "waiting": float(len(self.waiting)),
            "running": float(len(self.running)),
            "swapped": float(len(self.swapped)),
            "kv_blocks_used": float(self.runner.block_manager.in_use()),
            "kv_blocks_total": float(self.runner.block_manager.num_blocks),
            "prefix_hit_rate": pc["hit_rate"],
            "kv_cache_bytes": float(self.runner.cache.nbytes()),
        }
