"""Paged KV cache, block manager and radix prefix cache.

Why paging: a contiguous KV buffer forces you to reserve ``max_model_len`` per sequence,
which is what makes long-context serving on an 8 GB card impossible.  Blocks of
``block_size`` tokens let sequences share and grow incrementally, and let a *shared prefix*
be referenced rather than copied.

Why a radix prefix cache: a multi-turn VL conversation re-sends the same image and the same
earlier turns.  Re-prefilling an image costs seconds; reusing its KV costs nothing.  The
cache key is therefore **not** just token ids -- it is the token ids *plus* a fingerprint of
the vision embeddings and the RoPE configuration, because two requests can carry identical
token ids while differing in pixels (``<image>`` placeholders are indistinguishable by id).
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .attention import gather_paged_kv

__all__ = [
    "BlockManager",
    "PagedKVCache",
    "RadixPrefixCache",
    "SequenceState",
    "prefix_fingerprint",
]


class BlockManager:
    """Allocates and reference-counts physical KV blocks."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free: List[int] = list(range(num_blocks))
        self.refcount: Dict[int, int] = {i: 0 for i in range(num_blocks)}

    @property
    def num_free(self) -> int:
        return len(self.free)

    def allocate(self, n: int = 1) -> List[int]:
        if n > len(self.free):
            raise MemoryError(f"cannot allocate {n} blocks; only {len(self.free)} free")
        out = [self.free.pop() for _ in range(n)]
        for b in out:
            self.refcount[b] = 1
        return out

    def fork(self, blocks: Sequence[int]) -> None:
        """Add one reference to each block (used when a sequence adopts a cached prefix)."""
        for b in blocks:
            self.refcount[b] = self.refcount.get(b, 0) + 1

    def free_blocks(self, blocks: Sequence[int]) -> None:
        """Release one reference to each block.

        Strict on purpose: an unknown id or a double free means the caller has lost track of
        ownership, and the lenient version of this method silently pushed an out-of-range id
        onto the free list -- which `allocate` would then hand out, producing a slot mapping
        that writes outside the KV tensor.  Failing loudly is the only safe behaviour for a
        memory manager.
        """
        for b in blocks:
            b = int(b)
            if not 0 <= b < self.num_blocks:
                raise ValueError(f"block id {b} is outside [0, {self.num_blocks})")
            rc = self.refcount.get(b)
            if rc is None:
                raise ValueError(f"block {b} was never allocated")
            if rc <= 0:
                raise ValueError(f"block {b} is already free (double free)")
            rc -= 1
            if rc == 0:
                self.refcount[b] = 0
                if b not in self.free:
                    self.free.append(b)
            else:
                self.refcount[b] = rc

    def in_use(self) -> int:
        return self.num_blocks - len(self.free)


@dataclass
class _RadixNode:
    tokens: Tuple[int, ...]
    block_ids: Tuple[int, ...]
    parent: Optional[int]
    children: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    last_used: float = 0.0
    refcount: int = 0


class RadixPrefixCache:
    """Block-granular prefix cache over a radix tree.

    Only whole blocks are cached; the trailing partial block of a request is never inserted
    because its contents depend on tokens that follow.
    """

    def __init__(self, block_manager: BlockManager, max_nodes: int = 4096):
        self.bm = block_manager
        self.block_size = block_manager.block_size
        self.nodes: Dict[int, _RadixNode] = {}
        self.root = -1
        self._next_id = 0
        self.max_nodes = max_nodes
        self.hits = 0
        self.misses = 0

    # -- keying -----------------------------------------------------------------------
    def _block_key(self, tokens: Sequence[int]) -> Tuple[int, ...]:
        return tuple(int(t) for t in tokens)

    def match(self, tokens: Sequence[int], extra_key: str = "") -> Tuple[int, List[int], List[int]]:
        """Longest cached prefix of ``tokens`` -> ``(matched_tokens, block_ids, node_ids)``.

        ``node_ids`` is returned alongside the blocks because ownership must be released by
        *node identity*, not by physical block id: after a sequence is preempted and resumed it
        holds freshly allocated blocks, so its block table no longer names the nodes it adopted.
        Releasing by block id would then decrement the wrong node (or none at all).

        ``extra_key`` (see :func:`prefix_fingerprint`) is mixed into the node key so that two
        requests with identical token ids but different pixels cannot collide.
        """
        if len(tokens) < self.block_size:
            self.misses += 1
            return 0, [], []
        node_id = self.root
        matched_tokens = 0
        blocks: List[int] = []
        node_ids: List[int] = []
        while matched_tokens + self.block_size <= len(tokens):
            key = self._block_key(tokens[matched_tokens : matched_tokens + self.block_size])
            if extra_key:
                key = (hash(extra_key) & 0xFFFFFFFF,) + key
            node = self.nodes.get(node_id)
            nxt = node.children.get(key) if node else None
            if nxt is None:
                break
            child = self.nodes[nxt]
            blocks.extend(child.block_ids)
            matched_tokens += len(child.tokens)
            child.last_used = time.time()
            child.refcount += 1
            node_ids.append(nxt)
            node_id = nxt
        if blocks:
            self.hits += 1
        else:
            self.misses += 1
        return matched_tokens, blocks, node_ids

    def insert(self, tokens: Sequence[int], block_ids: Sequence[int], extra_key: str = "") -> None:
        """Register whole blocks of ``tokens`` as a cached prefix.

        New nodes start with ``refcount == 0``: the *registering* sequence is not a user of the
        cache, it merely donated blocks.  Only a later ``match`` makes a node in-use.  Counting
        the registrar as a user would leave every donated node permanently un-evictable, because
        the engine only releases adopters.
        """
        n_blocks = len(block_ids)
        if n_blocks == 0:
            return
        if len(tokens) < n_blocks * self.block_size:
            raise ValueError("fewer tokens than blocks supplied")
        node_id = self.root
        pos = 0
        for i in range(n_blocks):
            chunk = self._block_key(tokens[pos : pos + self.block_size])
            key = (hash(extra_key) & 0xFFFFFFFF,) + chunk if extra_key else chunk
            node = self.nodes.get(node_id)
            if node is None:
                self.nodes[node_id] = _RadixNode(tokens=(), block_ids=(), parent=None)
                node = self.nodes[node_id]
            if key in node.children:
                child = self.nodes[node.children[key]]
                child.last_used = time.time()
            else:
                cid = self._next_id
                self._next_id += 1
                self.nodes[cid] = _RadixNode(
                    tokens=chunk, block_ids=(block_ids[i],), parent=node_id, last_used=time.time(), refcount=0
                )
                node.children[key] = cid
                self.bm.fork([block_ids[i]])
            node_id = node.children[key]
            pos += self.block_size
        self._evict_if_needed()

    def release(self, node_ids: Sequence[int]) -> None:
        """Drop one *user* reference from each of ``node_ids``.

        Takes node ids, not block ids: a sequence that was preempted and resumed holds freshly
        allocated blocks, so its block table no longer names the nodes it adopted.  Matching by
        physical block id would decrement the wrong node, or none, and leak the adopted node.

        Ownership model, which an earlier version got wrong in both directions:

        * The cache holds exactly **one** block reference per cached node, taken via ``bm.fork``
          at ``insert`` time and dropped only when the node is evicted.
        * ``node.refcount`` counts **live users** (sequences that matched this node), and exists
          solely to decide whether a node is safe to evict.

        So this method must *not* call ``bm.free_blocks``: the caller frees its own reference
        through ``runner.free``, and freeing again here would be a double free.
        """
        for nid in node_ids:
            node = self.nodes.get(int(nid))
            if node is not None and node.refcount > 0:
                node.refcount -= 1

    def _subtree_is_evictable(self, node_id: int) -> bool:
        """True when nothing in this subtree has a live user."""
        stack = [node_id]
        while stack:
            nid = stack.pop()
            node = self.nodes.get(nid)
            if node is None:
                continue
            if node.refcount > 0:
                return False
            stack.extend(node.children.values())
        return True

    def _drop_subtree(self, node_id: int) -> None:
        """Remove a node and everything below it, dropping the cache's block references."""
        stack = [node_id]
        while stack:
            nid = stack.pop()
            node = self.nodes.pop(nid, None)
            if node is None:
                continue
            stack.extend(node.children.values())
            if node.block_ids:
                self.bm.free_blocks(node.block_ids)

    def _evictable(self) -> List[int]:
        """Node ids that may be dropped, least-recently-used first."""
        return sorted(
            (nid for nid, n in self.nodes.items()
             if n.parent is not None and n.refcount == 0 and self._subtree_is_evictable(nid)),
            key=lambda nid: self.nodes[nid].last_used,
        )

    def _detach_and_drop(self, nid: int) -> None:
        node = self.nodes.get(nid)
        if node is None:
            return
        parent = self.nodes.get(node.parent) if node.parent is not None else None
        if parent is not None:
            for key, cid in list(parent.children.items()):
                if cid == nid:
                    parent.children.pop(key, None)
        self._drop_subtree(nid)

    def evict(self, count: int = 1) -> int:
        """Force-evict up to ``count`` cached nodes; returns how many were dropped.

        The engine calls this under memory pressure: cached prefixes hold block references, and
        with a small pool those blocks are exactly what a new request needs.  Without a way to
        give them back, the scheduler can stall with work still queued.
        """
        dropped = 0
        for nid in self._evictable():
            if dropped >= count:
                break
            self._detach_and_drop(nid)
            dropped += 1
        return dropped

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used evictable nodes until the tree is back in budget.

        An earlier version popped only the victim's *children*, never removed the victim
        itself, and never released its blocks -- so the tree grew without bound and the block
        pool drained permanently.  Both halves are fixed here.

        ``max_nodes`` counts cached nodes, not the root, and a victim is only dropped when its
        whole subtree has no live user: eviction must never free a block that a running
        sequence is still reading.
        """
        # `len(self.nodes)` includes the lazily created root, which holds no blocks.
        cached = len(self.nodes) - (1 if self.root in self.nodes else 0)
        if cached <= self.max_nodes:
            return
        for nid in self._evictable():
            cached = len(self.nodes) - (1 if self.root in self.nodes else 0)
            if cached <= self.max_nodes:
                break
            self._detach_and_drop(nid)

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "nodes": len(self.nodes),
        }


def prefix_fingerprint(
    token_ids: Sequence[int],
    image_features: Optional[torch.Tensor] = None,
    grid_thw: Optional[torch.Tensor] = None,
    rope_theta: float = 1_000_000.0,
    mrope_section: Sequence[int] = (16, 24, 24),
) -> str:
    """Fingerprint that makes a prefix cache entry safe for multimodal requests.

    Token ids alone are not enough: ``<|image_pad|>`` runs of equal length occur for many
    different images.  Hashing the *embedding bytes* of the vision features (quantised to
    8-bit so tiny numerical differences do not defeat reuse) plus the grid and the RoPE
    configuration ties the entry to the actual forward-pass inputs.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(b"v1")
    arr = torch.as_tensor(list(token_ids), dtype=torch.int64).numpy().tobytes()
    h.update(arr)
    if grid_thw is not None:
        h.update(torch.as_tensor(grid_thw).to(torch.int64).numpy().tobytes())
    if image_features is not None:
        f = image_features.detach().to(torch.float32)
        scale = f.abs().amax().clamp_min(1e-6)
        q = torch.round(f / scale * 127.0).to(torch.int8)
        h.update(q.numpy().tobytes())
    h.update(f"{rope_theta}:{tuple(int(s) for s in mrope_section)}".encode())
    return h.hexdigest()


class PagedKVCache:
    """Per-layer paged key/value storage with optional INT8 quantisation.

    Layout is ``[num_layers, num_blocks, block_size, kv_heads, head_dim]``.  With
    ``quantize=True`` values are stored as INT8 with one fp16 scale per (token, head);
    a per-token scale (rather than per-block) keeps writes order-independent, which matters
    because chunked prefill fills a block over several steps.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device="cpu",
        quantize: bool = False,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.quantize = quantize
        self.device = device
        self.dtype = dtype

        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        if quantize:
            self.k = torch.zeros(shape, dtype=torch.int8, device=device)
            self.v = torch.zeros(shape, dtype=torch.int8, device=device)
            self.k_scale = torch.zeros((num_layers, num_blocks, block_size, num_kv_heads), dtype=torch.float16, device=device)
            self.v_scale = torch.zeros_like(self.k_scale)
        else:
            self.k = torch.zeros(shape, dtype=dtype, device=device)
            self.v = torch.zeros(shape, dtype=dtype, device=device)
            self.k_scale = self.v_scale = None

    def nbytes(self) -> int:
        total = self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()
        if self.quantize:
            total += self.k_scale.numel() * self.k_scale.element_size() * 2
        return total

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Scatter ``[num_tokens, kv_heads, head_dim]`` into the given physical slots."""
        n = slot_mapping.numel()
        if k.shape[0] != n or v.shape[0] != n:
            raise ValueError(f"slot_mapping has {n} entries but k/v have {k.shape[0]}/{v.shape[0]} tokens")
        kf = self.k[layer_idx].view(self.num_blocks * self.block_size, self.num_kv_heads, self.head_dim)
        vf = self.v[layer_idx].view(self.num_blocks * self.block_size, self.num_kv_heads, self.head_dim)
        if self.quantize:
            ks = k.float().abs().amax(dim=-1).clamp_min(1e-8) / 127.0
            vs = v.float().abs().amax(dim=-1).clamp_min(1e-8) / 127.0
            kq = torch.round(k.float() / ks.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
            vq = torch.round(v.float() / vs.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
            ksf = self.k_scale[layer_idx].view(self.num_blocks * self.block_size, self.num_kv_heads)
            vsf = self.v_scale[layer_idx].view(self.num_blocks * self.block_size, self.num_kv_heads)
            ksf[slot_mapping] = ks.to(torch.float16)
            vsf[slot_mapping] = vs.to(torch.float16)
            kf[slot_mapping] = kq
            vf[slot_mapping] = vq
        else:
            kf[slot_mapping] = k.to(self.dtype)
            vf[slot_mapping] = v.to(self.dtype)

    def gather(
        self,
        layer_idx: int,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense ``[batch, kv_heads, max_len, head_dim]`` view of the requested sequences."""
        if not self.quantize:
            kb = self.k[layer_idx]
            vb = self.v[layer_idx]
            return gather_paged_kv(kb, vb, block_tables, context_lens, self.block_size)

        batch, max_blocks = block_tables.shape
        max_len = max(int(context_lens.max().item()), 1)
        k_out = torch.zeros(batch, self.num_kv_heads, max_len, self.head_dim, dtype=self.dtype, device=self.device)
        v_out = torch.zeros_like(k_out)
        for b in range(batch):
            n = int(context_lens[b].item())
            if n == 0:
                continue
            nblocks = (n + self.block_size - 1) // self.block_size
            ids = block_tables[b, :nblocks].tolist()
            if any(i < 0 for i in ids):
                raise ValueError(f"sequence {b} references an unallocated block: {ids}")
            idx = torch.tensor(ids, device=self.device, dtype=torch.long)
            kb = self.k[layer_idx][idx].reshape(-1, self.num_kv_heads, self.head_dim)[:n]
            vb = self.v[layer_idx][idx].reshape(-1, self.num_kv_heads, self.head_dim)[:n]
            ks = self.k_scale[layer_idx][idx].reshape(-1, self.num_kv_heads)[:n]
            vs = self.v_scale[layer_idx][idx].reshape(-1, self.num_kv_heads)[:n]
            k_out[b, :, :n] = (kb.float() * ks.unsqueeze(-1).float()).transpose(0, 1).to(self.dtype)
            v_out[b, :, :n] = (vb.float() * vs.unsqueeze(-1).float()).transpose(0, 1).to(self.dtype)
        return k_out, v_out

    def copy_blocks(self, src_blocks: Sequence[int], dst_blocks: Sequence[int]) -> None:
        """Copy physical blocks (used by swap/preemption instead of re-prefilling)."""
        src = torch.tensor(list(src_blocks), device=self.device, dtype=torch.long)
        dst = torch.tensor(list(dst_blocks), device=self.device, dtype=torch.long)
        self.k[:, dst] = self.k[:, src]
        self.v[:, dst] = self.v[:, src]
        if self.quantize:
            self.k_scale[:, dst] = self.k_scale[:, src]
            self.v_scale[:, dst] = self.v_scale[:, src]

    def swap_out(self, blocks: Sequence[int]) -> Dict[str, torch.Tensor]:
        """Detach a sequence's KV to CPU (preemption by swap).

        The payload records how many blocks it holds.  On restore the *same* count must be
        allocated, because a block's physical id determines which logical positions it
        holds -- allocating a different number would silently mis-map the KV.
        """
        idx = torch.tensor(list(blocks), device=self.device, dtype=torch.long)
        payload: Dict[str, torch.Tensor] = {
            "k": self.k[:, idx].to("cpu", copy=True),
            "v": self.v[:, idx].to("cpu", copy=True),
            "num_blocks": torch.tensor([len(blocks)], dtype=torch.long),
        }
        if self.quantize:
            payload["k_scale"] = self.k_scale[:, idx].to("cpu", copy=True)
            payload["v_scale"] = self.v_scale[:, idx].to("cpu", copy=True)
        return payload

    def swap_in(self, blocks: Sequence[int], payload: Dict[str, torch.Tensor]) -> None:
        """Restore a payload written by :meth:`swap_out` into freshly allocated blocks."""
        expected = int(payload["num_blocks"].item()) if "num_blocks" in payload else payload["k"].shape[1]
        if len(blocks) != expected:
            raise ValueError(
                f"swap payload holds {expected} blocks but {len(blocks)} were allocated; "
                "restoring into a different block count would mis-map the KV"
            )
        idx = torch.tensor(list(blocks), device=self.device, dtype=torch.long)
        self.k[:, idx] = payload["k"].to(self.device)
        self.v[:, idx] = payload["v"].to(self.device)
        if self.quantize:
            self.k_scale[:, idx] = payload["k_scale"].to(self.device)
            self.v_scale[:, idx] = payload["v_scale"].to(self.device)

    def swap_block_count(self, payload: Dict[str, torch.Tensor]) -> int:
        return int(payload["num_blocks"].item()) if "num_blocks" in payload else payload["k"].shape[1]


@dataclass
class SequenceState:
    """Everything the scheduler needs to know about one in-flight request."""

    seq_id: int
    prompt_token_ids: List[int]
    block_size: int
    sampling_params: dict = field(default_factory=dict)
    priority: int = 0
    output_token_ids: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    cached_tokens: int = 0  # tokens satisfied by a prefix-cache hit (independent of decode steps)
    block_table: List[int] = field(default_factory=list)
    prefix_node_ids: List[int] = field(default_factory=list)  # cache nodes this seq adopted
    status: str = "waiting"  # waiting | running | swapped | finished
    vision_features: Optional[torch.Tensor] = None
    grid_thw: Optional[torch.Tensor] = None
    image_cursor: int = 0  # how many image feature rows have already been consumed
    rope_delta: int = 0
    prefix_key: str = ""
    swap_payload: Optional[Dict[str, torch.Tensor]] = None
    arrival: float = field(default_factory=time.time)
    finish_reason: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.all_token_ids)

    def num_blocks_needed(self, upto: Optional[int] = None) -> int:
        n = self.num_tokens if upto is None else upto
        return (n + self.block_size - 1) // self.block_size

    def slot_mapping(self, start: int, end: int) -> List[int]:
        """Physical slots for logical positions ``[start, end)``."""
        return [
            self.block_table[pos // self.block_size] * self.block_size + (pos % self.block_size)
            for pos in range(start, end)
        ]
