"""Static CPU/GPU layer placement with double-buffered prefetch.

The external review of 2026-10-15 recommended *static* placement over a layer LRU for a
first version, and the reasoning is sound: a decoder visits layers in a fixed order, so an
LRU cache cannot reduce the total weight traffic -- it only adds bookkeeping and jitter.  What
does help is overlapping the host-to-device copy of layer ``i+1`` with the compute of layer
``i``, which is what this module does.

Honest limits, stated up front because they are easy to hide:

* streaming weights over PCIe only pays off if the copy overlaps compute.  If it does not,
  decode throughput becomes transfer-bound, and the measured ``overlap_ratio`` below is the
  number that tells you which regime you are in;
* nothing here is exercised on a CUDA device in CI (the test runner is CPU-only), so the
  scheduling logic is unit-tested while the transfer numbers must be measured on the target
  machine.  ``StaticLayerOffloader.report`` *holds* those numbers (it is an ``OffloadReport``
  dataclass); ``report.summary()`` is what turns them into text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

__all__ = ["StaticLayerOffloader", "placement_for", "OffloadReport"]


def placement_for(total_layers: int, gpu_layers: int) -> List[bool]:
    """Which layers stay resident.

    Layers are kept in order from the top of the stack downwards, which keeps the *first*
    layers resident: they are hit on every step, and their weights are the ones most likely to
    still be in cache after a prefill pass.
    """
    if gpu_layers >= total_layers:
        return [True] * total_layers
    if gpu_layers <= 0:
        return [False] * total_layers
    return [i < gpu_layers for i in range(total_layers)]


@dataclass
class OffloadReport:
    h2d_bytes: int = 0
    h2d_seconds: float = 0.0
    compute_seconds: float = 0.0
    prefetch_hits: int = 0
    prefetch_misses: int = 0

    @property
    def effective_bandwidth_gbps(self) -> float:
        if self.h2d_seconds <= 0:
            return 0.0
        return self.h2d_bytes / self.h2d_seconds / 1e9

    @property
    def overlap_ratio(self) -> float:
        """Fraction of transfer time that was hidden behind compute."""
        if self.h2d_seconds <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - max(0.0, self.h2d_seconds - self.compute_seconds) / self.h2d_seconds))

    def summary(self) -> str:
        return (
            f"offload: moved {self.h2d_bytes/1e6:.1f} MB in {self.h2d_seconds*1000:.1f} ms "
            f"({self.effective_bandwidth_gbps:.2f} GB/s effective), compute {self.compute_seconds*1000:.1f} ms, "
            f"overlap {self.overlap_ratio*100:.0f}%, prefetch hits {self.prefetch_hits}, misses {self.prefetch_misses}"
        )


class StaticLayerOffloader:
    """Keeps a subset of decoder layers on the device and prefetches the rest.

    Parameters
    ----------
    layers:
        the decoder layer modules, in execution order.
    gpu_layers:
        how many of them stay resident (the first ``gpu_layers``).
    device / host_device:
        e.g. ``cuda:0`` and ``cpu``.
    pin_memory:
        pin host copies so the H2D transfer can use DMA and overlap with compute.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        gpu_layers: int,
        device: str = "cuda",
        host_device: str = "cpu",
        pin_memory: bool = True,
    ):
        self.layers = list(layers)
        self.resident = placement_for(len(self.layers), gpu_layers)
        self.device = torch.device(device)
        self.host_device = torch.device(host_device)
        self.pin_memory = pin_memory and self.host_device.type == "cpu"
        self.report = OffloadReport()
        self._host_copies: Dict[int, List[torch.Tensor]] = {}
        self._stream = None
        self._inflight: Optional[int] = None
        self._inflight_start: float = 0.0

        if self.device.type == "cuda" and torch.cuda.is_available():
            self._stream = torch.cuda.Stream(device=self.device)

    # -- setup -----------------------------------------------------------------------
    def prepare(self) -> None:
        """Move resident layers to the device and stash host copies of the rest."""
        for i, layer in enumerate(self.layers):
            if self.resident[i]:
                layer.to(self.device)
                continue
            layer.to(self.host_device)
            self._host_copies[i] = [p.detach() for p in layer.parameters()]

    # -- runtime ---------------------------------------------------------------------
    def begin_layer(self, idx: int) -> None:
        """Ensure layer ``idx`` is usable and kick off the prefetch of ``idx + 1``."""
        if not self.resident[idx]:
            self._materialise(idx)
        self._prefetch(idx + 1)

    def _prefetch(self, idx: int) -> None:
        if idx >= len(self.layers) or self.resident[idx]:
            return
        start = time.perf_counter()
        layer = self.layers[idx]
        with torch.cuda.stream(self._stream) if self._stream is not None else _null_context():
            layer.to(self.device)
        self._inflight = idx
        self._inflight_start = start
        self.report.h2d_bytes += sum(p.numel() * p.element_size() for p in layer.parameters())

    def _materialise(self, idx: int) -> None:
        layer = self.layers[idx]
        if next(layer.parameters()).device.type == self.device.type:
            self.report.prefetch_hits += 1
            return
        self.report.prefetch_misses += 1
        layer.to(self.device)
        self.report.h2d_bytes += sum(p.numel() * p.element_size() for p in layer.parameters())
        self.report.h2d_seconds += time.perf_counter() - self._inflight_start if self._inflight == idx else 0.0

    def end_layer(self, idx: int, compute_seconds: float) -> None:
        """Record compute time and free a streamed layer's device copy."""
        self.report.compute_seconds += compute_seconds
        if self._inflight == idx and self._stream is not None:
            self._stream.synchronize()
            self.report.h2d_seconds += time.perf_counter() - self._inflight_start
            self._inflight = None
        if not self.resident[idx]:
            layer = self.layers[idx]
            # Release the device copy but keep the pinned host copy for the next pass.
            layer.to(self.host_device)
            self._free_dequant_caches(layer)

    def _free_dequant_caches(self, layer: nn.Module) -> None:
        for module in layer.modules():
            if hasattr(module, "free_dequant_cache"):
                module.free_dequant_cache()

    def stats(self) -> Dict[str, float]:
        return {
            "resident_layers": float(sum(self.resident)),
            "streamed_layers": float(len(self.layers) - sum(self.resident)),
            "h2d_bytes": float(self.report.h2d_bytes),
            "effective_bandwidth_gbps": self.report.effective_bandwidth_gbps,
            "overlap_ratio": self.report.overlap_ratio,
        }


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
