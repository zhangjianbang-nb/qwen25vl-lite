"""Qwen2.5-VL text decoder.

The decoder is a standard Qwen2 architecture (GQA, SwiGLU, RMSNorm, no biases on
``o_proj``/MLP) with one twist: rotary embeddings are **MRoPE**, so ``cos``/``sin`` carry
three axis tables instead of one.

Attention supports three execution shapes through one entry point:

* ``dense``   -- no cache, additive mask or plain causal (used by the HF-equivalence test);
* ``paged``   -- write K/V into the paged cache and read the whole context back (decode);
* ``chunked`` -- many queries whose keys live partly in the cache (prefill chunking).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .attention import build_chunk_mask, build_decode_mask, build_paged_slots, sdpa
from .cache import PagedKVCache
from .config import TextConfig
from .layers import RMSNorm
from .paged_attn_triton import can_use as paged_kernel_can_use
from .paged_attn_triton import paged_attn_kernel_enabled, paged_decode_attention
from .rope import apply_multimodal_rotary_pos_emb, build_inv_freq, mrope_cos_sin

__all__ = ["RotaryEmbedding", "TextMLP", "TextAttention", "TextDecoderLayer", "TextModel"]


class RotaryEmbedding(nn.Module):
    """MRoPE tables.  ``cos_sin`` accepts ``[3, batch, seq]`` position ids.

    The inverse frequencies are computed lazily rather than stored as a buffer: the model is
    built on the ``meta`` device and buffers would otherwise need materialising separately
    from the parameters.  They are 128 floats, so recomputing on a device change is free.
    """

    def __init__(self, head_dim: int, rope_theta: float = 1_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self._inv_freq: Optional[torch.Tensor] = None

    def inv_freq(self, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self._inv_freq is None or self._inv_freq.device != torch.device(device or "cpu"):
            self._inv_freq = build_inv_freq(self.head_dim, self.rope_theta, device=device, dtype=dtype)
        return self._inv_freq

    def cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype):
        return mrope_cos_sin(position_ids, self.inv_freq(position_ids.device), dtype)


class TextMLP(nn.Module):
    """SwiGLU feed-forward, no biases (Qwen2 convention)."""

    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TextAttention(nn.Module):
    def __init__(self, cfg: TextConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = cfg.head_dim
        self.scaling = self.head_dim ** -0.5
        # Qwen2.5-VL keeps biases on q/k/v and none on o.
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def project(self, hidden_states: torch.Tensor):
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: Optional[PagedKVCache] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        query_lens: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        backends: Optional[Sequence[str]] = None,
        want_probs: bool = False,
        kv_collector: Optional[list] = None,
        paged_slots: Optional[torch.Tensor] = None,
        paged_mask: Optional[torch.Tensor] = None,
    ):
        """Returns the attention output; with ``want_probs=True`` also the probabilities.

        ``kv_collector`` receives ``(k, v)`` in the dense (batch, kv_heads, seq, head_dim)
        layout after RoPE -- the FastV prefill path needs those to write a cache that matches
        the *pruned* sequence rather than the full one.
        """
        bsz, q_len, _ = hidden_states.shape
        q, k, v = self.project(hidden_states)
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.cfg.mrope_section)
        if kv_collector is not None:
            kv_collector.append((k, v))

        if cache is not None:
            if slot_mapping is None or block_tables is None or context_lens is None:
                raise ValueError("paged attention needs slot_mapping, block_tables and context_lens")
            cache.write(self.layer_idx, k.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim),
                        v.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim), slot_mapping)
            # The padded KV width, read from the block table's shape rather than from the
            # device.  The engine builds the slot index and the length mask once per step and
            # passes them down: both depend only on the block table and the context lengths,
            # so recomputing them per layer was ~8 redundant launches times 36 layers a token.
            max_len = block_tables.shape[1] * cache.block_size
            # Decode has exactly one query per sequence, and that is the shape the fused
            # flash-decoding kernel is built for: it reads the block table directly, stops at
            # each sequence's own context, and never materialises a dense gather or a mask.
            # Chunked prefill keeps the reference path, which the kernel does not cover.
            if (query_lens is None and not want_probs and paged_attn_kernel_enabled()
                    and paged_kernel_can_use(
                        q, cache.k[self.layer_idx], cache.v[self.layer_idx], context_lens)):
                out = paged_decode_attention(
                    q, cache.k[self.layer_idx], cache.v[self.layer_idx],
                    block_tables, context_lens, scale=self.scaling,
                )
                return self.o_proj(out.transpose(1, 2).reshape(bsz, q_len, -1).contiguous())

            if paged_slots is None:
                paged_slots = build_paged_slots(block_tables, cache.block_size, max_len)
            k, v = cache.gather_slots(self.layer_idx, paged_slots)

            if paged_mask is not None:
                mask = paged_mask
            elif query_lens is None:
                # Decode: one query per sequence attending to its own context.  The gathered
                # KV is padded to the batch maximum, so a length mask is required.
                mask = build_decode_mask(context_lens, q.device, q.dtype, max_len=max_len)
            else:
                mask = build_chunk_mask(query_lens, context_lens, q.device, q.dtype,
                                        max_q=q.shape[2], max_ctx=max_len)
            if want_probs:
                offset = max_len - q_len
                out, probs = self._probs(q, k, v, mask, offset, backends)
            else:
                out = sdpa(q, k, v, attn_mask=mask, scale=self.scaling, backends=backends,
                           enable_gqa=True, num_kv_groups=self.num_kv_groups)
                probs = None
        else:
            if want_probs:
                out, probs = self._probs(q, k, v, attn_mask, 0, backends)
            else:
                if attn_mask is None:
                    out = sdpa(q, k, v, is_causal=True, scale=self.scaling, backends=backends,
                               enable_gqa=True, num_kv_groups=self.num_kv_groups)
                else:
                    out = sdpa(q, k, v, attn_mask=attn_mask, scale=self.scaling, backends=backends,
                               enable_gqa=True, num_kv_groups=self.num_kv_groups)
                probs = None

        out = out.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
        out = self.o_proj(out)
        return (out, probs) if want_probs else out

    def _probs(self, q, k, v, mask, offset, backends):
        from .attention import attention_with_probs

        return attention_with_probs(q, k, v, attn_mask=mask, scale=self.scaling,
                                    num_kv_groups=self.num_kv_groups, query_offset=offset)


class TextDecoderLayer(nn.Module):
    def __init__(self, cfg: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = TextAttention(cfg, layer_idx)
        self.mlp = TextMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, **kw):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states, cos, sin, **kw)
        if isinstance(attn_out, tuple):
            hidden_states, probs = attn_out
        else:
            hidden_states, probs = attn_out, None
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return (hidden_states, probs) if probs is not None else hidden_states


class TextModel(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TextDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        cache: Optional[PagedKVCache] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        query_lens: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        backends: Optional[Sequence[str]] = None,
        capture_probs_at: Optional[int] = None,
        layer_hook=None,
        start_layer: int = 0,
        stop_layer: Optional[int] = None,
        kv_collector: Optional[list] = None,
    ):
        """Run the decoder stack (optionally a contiguous slice of it).

        ``layer_hook`` is called as ``layer_hook(layer_idx, hidden_states, probs)`` after each
        layer; it exists so the FastV stage can prune image tokens mid-stack without this
        module knowing anything about pruning.  ``start_layer``/``stop_layer`` let the FastV
        prefill run the first ``L+1`` layers densely, prune, then continue from ``L+1``.
        """
        cos, sin = self.rotary_emb.cos_sin(position_ids, inputs_embeds.dtype)
        hidden_states = inputs_embeds

        # The slot index depends only on the block table, and the decode mask only on the
        # context lengths -- neither changes from layer to layer.  Building them here once
        # removes ~8 launches per layer, which at 36 layers is most of a decode step's
        # dispatch cost.  The chunked-prefill mask is left to the layer because FastV pruning
        # can change the query length mid-stack.
        paged_slots = paged_mask = None
        if cache is not None and block_tables is not None and context_lens is not None:
            # The fused kernel walks the block table itself, so it needs neither the slot index
            # nor the length mask.  Building them anyway would allocate two [batch, max_len]
            # tensors every step for nothing -- and with a bucketed block-table width, max_len
            # is padded upwards, so the waste grows with the bucket.
            if not (query_lens is None and paged_attn_kernel_enabled()):
                max_len = block_tables.shape[1] * cache.block_size
                paged_slots = build_paged_slots(block_tables, cache.block_size, max_len)
                if query_lens is None:
                    paged_mask = build_decode_mask(context_lens, inputs_embeds.device,
                                                   inputs_embeds.dtype, max_len=max_len)

        stop = len(self.layers) if stop_layer is None else stop_layer
        for i in range(start_layer, stop):
            layer = self.layers[i]
            want = capture_probs_at is not None and i == capture_probs_at
            hidden_states = layer(
                hidden_states, cos, sin, cache=cache, slot_mapping=slot_mapping,
                block_tables=block_tables, context_lens=context_lens, query_lens=query_lens,
                attn_mask=attn_mask, backends=backends, want_probs=want,
                kv_collector=kv_collector, paged_slots=paged_slots, paged_mask=paged_mask,
            )
            probs = None
            if want:
                hidden_states, probs = hidden_states
            if layer_hook is not None:
                hidden_states = layer_hook(i, hidden_states, probs)
        return self.norm(hidden_states) if stop >= len(self.layers) else hidden_states
