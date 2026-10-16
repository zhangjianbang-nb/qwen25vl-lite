"""Memory planning for a target device.

The external review of 2026-10-15 was right that the original budget was optimistic, and
right about *why*: it conflated three different numbers that users read as one.

This module therefore reports them separately and never merges them:

* **logical context** -- the longest sequence the engine will accept (limited by
  ``max_position_embeddings`` and by what fits after swapping);
* **GPU-resident KV** -- how many tokens of KV actually live in VRAM (the number that can
  be attended to at full speed);
* **CPU swap capacity** -- how many additional tokens can be paged out to host memory.

KV size per token is computed, not guessed: ``2 (K and V) * layers * kv_heads * head_dim *
bytes``.  For Qwen2.5-VL-3B in fp16 that is ``2 * 36 * 2 * 128 * 2 = 36864`` bytes, i.e.
36 KiB/token -- so 250k tokens of fp16 KV is 8.6 GiB on its own, which cannot fit in 8 GB.
Any claim about context length has to say which of the three numbers it refers to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import Qwen25VLConfig, bytes_per_weight

__all__ = ["KVBytes", "MemoryPlan", "kv_bytes_per_token", "plan_memory", "format_bytes"]


@dataclass
class KVBytes:
    """Per-token KV cost, split out so the arithmetic is auditable."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_element: float

    @property
    def per_token(self) -> float:
        return 2.0 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_element

    def describe(self) -> str:
        return (
            f"2 x {self.num_layers} layers x {self.num_kv_heads} kv_heads x {self.head_dim} head_dim "
            f"x {self.bytes_per_element:g} B = {self.per_token/1024:.1f} KiB/token"
        )


def kv_bytes_per_token(cfg: Qwen25VLConfig, kv_dtype: str = "float16") -> KVBytes:
    bytes_per = {"float16": 2.0, "bfloat16": 2.0, "float32": 4.0, "int8": 1.0}[kv_dtype]
    t = cfg.text_config
    return KVBytes(t.num_hidden_layers, t.num_key_value_heads, t.head_dim, bytes_per)


def format_bytes(n: float) -> str:
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if abs(n) >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n:.0f} B"


@dataclass
class MemoryPlan:
    """A concrete deployment plan plus the evidence behind each number."""

    weights_bytes: float
    kv_per_token: float
    kv_pool_tokens: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_layers: int
    cpu_layers: int
    activation_reserve_bytes: float
    runtime_reserve_bytes: float
    kv_dtype: str
    weight_dtype: str
    notes: List[str] = field(default_factory=list)

    @property
    def kv_pool_bytes(self) -> float:
        return self.kv_pool_tokens * self.kv_per_token

    @property
    def gpu_bytes(self) -> float:
        return self.weights_bytes + self.kv_pool_bytes + self.activation_reserve_bytes + self.runtime_reserve_bytes

    def summary(self) -> str:
        lines = [
            "memory plan",
            f"  weights ({self.weight_dtype:<5})        {format_bytes(self.weights_bytes)}",
            f"  KV per token             {self.kv_per_token/1024:.1f} KiB",
            f"  GPU-resident KV pool     {format_bytes(self.kv_pool_bytes)} ({self.kv_pool_tokens:,} tokens, {self.kv_dtype})",
            f"  vision/prefill reserve   {format_bytes(self.activation_reserve_bytes)}",
            f"  runtime + allocator      {format_bytes(self.runtime_reserve_bytes)}",
            f"  ----------------------------------------",
            f"  total on GPU             {format_bytes(self.gpu_bytes)}",
            f"  decoder layers on GPU    {self.gpu_layers} of {self.gpu_layers + self.cpu_layers}",
            f"  max concurrent sequences {self.max_num_seqs}",
            f"  prefill chunk size       {self.max_num_batched_tokens}",
        ]
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def plan_memory(
    cfg: Qwen25VLConfig,
    *,
    vram_bytes: float,
    ram_bytes: float,
    weight_dtype: str = "int4",
    group_size: int = 128,
    kv_dtype: str = "float16",
    gpu_memory_utilization: float = 0.90,
    block_size: int = 16,
    max_num_seqs: Optional[int] = None,
    target_context: Optional[int] = None,
    max_vision_tokens: int = 1280,
    runtime_reserve_bytes: Optional[float] = None,
) -> MemoryPlan:
    """Produce a deployment plan for ``vram_bytes`` of device memory.

    ``runtime_reserve_bytes`` defaults to a measured-style 8% of the usable budget rather
    than a hard-coded 500 MB: CUDA context size, cuBLAS workspaces and allocator slack all
    scale with the workload, so a constant would be a guess dressed up as a fact.
    """
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")

    params = cfg.num_params()
    text_params = params["text"]
    vision_params = params["vision"]

    # Only decoder linears are quantised (see model.QUANTIZABLE_SUFFIXES), so the vision
    # tower and the embedding/head matrices stay at compute width.
    text_nonlinear_share = 0.18  # embeddings + norms + head (rough, reported as such)
    text_quantised = text_params * (1.0 - text_nonlinear_share)
    text_float = text_params * text_nonlinear_share

    bytes_w = bytes_per_weight(weight_dtype, group_size)
    weights_bytes = text_quantised * bytes_w + (text_float + vision_params) * 2.0

    kv = kv_bytes_per_token(cfg, kv_dtype)
    kv_per_token = kv.per_token

    usable = vram_bytes * gpu_memory_utilization
    reserve = runtime_reserve_bytes if runtime_reserve_bytes is not None else usable * 0.08

    # Vision activations during prefill: the tower holds `depth` intermediate copies of the
    # token stream plus attention temporaries.  Windows keep attention memory bounded, so the
    # dominant term is depth x tokens x hidden.
    vision_activation = max_vision_tokens * cfg.vision_config.hidden_size * 2.0 * min(cfg.vision_config.depth, 8) * 3.0
    lm_prefill_activation = max_vision_tokens * cfg.hidden_size * 2.0 * 4.0
    activation_reserve = vision_activation + lm_prefill_activation

    remaining = usable - weights_bytes - activation_reserve - reserve
    notes: List[str] = []

    if remaining <= 0:
        notes.append(
            "weights plus reserves exceed the usable budget; nothing is left for KV. "
            "Use a smaller quantisation or move decoder layers to the host."
        )
        kv_pool_tokens = 0
        gpu_layers = 0
    else:
        kv_pool_tokens = int(remaining // kv_per_token)

    total_layers = cfg.text_config.num_hidden_layers
    # Per-layer cost must be derived from the same quantity that produced `weights_bytes`, or
    # the two numbers do not reconcile: `weights_bytes` is the quantised text weights plus the
    # float text weights plus the whole vision tower.
    per_layer_bytes = weights_bytes / max(total_layers, 1)

    if weights_bytes > usable - activation_reserve - reserve:
        # Not everything fits: keep whole layers resident and stream the rest.  The KV pool was
        # sized from `remaining`, which already had the weights subtracted, so it must be added
        # back before deciding how many layers fit.
        head_room = max(usable - activation_reserve - reserve - kv_pool_tokens * kv_per_token, 0.0)
        gpu_layers = int(min(total_layers, head_room // per_layer_bytes))
        notes.append(
            f"only {gpu_layers}/{total_layers} decoder layers fit on device; the rest stream from "
            "pinned host memory (see offload.StaticLayerOffloader).  Expect decode throughput to be "
            "PCIe-bound for the streamed layers."
        )
    else:
        gpu_layers = total_layers
    cpu_layers = total_layers - gpu_layers

    if max_num_seqs is None:
        # A sequence needs at least one prompt's worth of KV; assume a modest 2048-token
        # prompt so the number of concurrent requests is not an arbitrary constant.
        per_seq = 2048
        max_num_seqs = max(1, min(64, kv_pool_tokens // per_seq)) if kv_pool_tokens else 1

    # A prefill chunk must fit in the KV pool, so the floor cannot exceed it: a chunk larger
    # than the pool would demand more blocks than exist and churn through preemption.
    max_num_batched_tokens = min(
        max(512, max_vision_tokens + 2048),
        kv_pool_tokens if kv_pool_tokens else 512,
    )

    if target_context is not None:
        needed = target_context * kv_per_token
        if needed > kv_pool_tokens * kv_per_token:
            notes.append(
                f"target context {target_context:,} tokens needs {format_bytes(needed)} of KV, but only "
                f"{kv_pool_tokens:,} tokens ({format_bytes(kv_pool_tokens*kv_per_token)}) fit on device. "
                "The remainder must live in host memory via swap; decode over swapped KV is slower and "
                "must be measured, not assumed."
            )

    host_kv_tokens = int(max(0.0, (ram_bytes - weights_bytes - activation_reserve) // kv_per_token))
    notes.append(
        f"host memory could hold roughly {host_kv_tokens:,} more KV tokens ({format_bytes(host_kv_tokens*kv_per_token)}); "
        "these are swap capacity, not GPU-resident capacity."
    )
    notes.append(
        f"logical context ceiling is {cfg.text_config.max_position_embeddings:,} tokens from the config; "
        "the GPU-resident figure above is what is fast, and the host figure is what is merely possible."
    )

    return MemoryPlan(
        weights_bytes=weights_bytes,
        kv_per_token=kv_per_token,
        kv_pool_tokens=max(kv_pool_tokens, 0),
        max_num_seqs=int(max_num_seqs),
        max_num_batched_tokens=int(max_num_batched_tokens),
        gpu_layers=gpu_layers,
        cpu_layers=cpu_layers,
        activation_reserve_bytes=activation_reserve,
        runtime_reserve_bytes=reserve,
        kv_dtype=kv_dtype,
        weight_dtype=weight_dtype,
        notes=notes,
    )
