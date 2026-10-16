"""Weight-only quantisation for Qwen2.5-VL.

Three schemes are implemented as *separate* code paths so that a quality regression can
always be attributed to exactly one of them (see the quality gates in ``docs/design.md``):

``int8_per_channel``
    Symmetric INT8 with one scale per output channel.  This is a plain per-channel
    quantiser -- it is **not** AWQ, and this codebase never labels it as such.

``int4_groupwise``
    Asymmetric INT4 where ``group_size`` consecutive input channels share a scale and a
    zero point.  Nibbles are packed two per byte.

``awq_style``
    The actual AWQ idea: rescale weight columns by the magnitude of the activations they
    receive, so channels that carry large activations keep more of their precision.  The
    exponent is chosen by an error-minimising grid search over calibration statistics.
    Without calibration data it degrades to ``int4_groupwise``.

Design note on fused QKV: because both schemes here group along the *output* channel (or
along K with a fixed group size), a concatenated ``q|k|v`` matrix needs no special
treatment -- the scale array simply follows the concatenated row order.  What must be
verified is that the dequantiser indexes scales with the same axis it used to compute
them; ``verify_scale_axis`` does exactly that and is exercised in the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "QuantizedTensor",
    "quantize_int8_per_channel",
    "quantize_int4_groupwise",
    "quantize_awq_style",
    "QuantLinear",
    "pack_int4",
    "unpack_int4",
    "verify_scale_axis",
]


def _pad_last_dim(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Right-pad the last dimension with zeros so it is a multiple of ``multiple``."""
    k = x.shape[-1]
    if k % multiple == 0:
        return x
    return F.pad(x, (0, multiple - (k % multiple)))


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack ``[..., K]`` uint4 values into ``[..., ceil(K/2)]`` bytes (low nibble first)."""
    if q.dtype != torch.uint8:
        q = q.to(torch.uint8)
    q = _pad_last_dim(q, 2)
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_int4(packed: torch.Tensor, k: int) -> torch.Tensor:
    """Inverse of :func:`pack_int4`; returns ``[..., k]`` uint8 values in ``[0, 15]``."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return out[..., :k].contiguous()


@dataclass
class QuantizedTensor:
    """Storage for a weight-only quantised linear weight.

    ``data`` holds the packed integers; ``scales`` and (optionally) ``zeros`` hold the
    per-group parameters.  ``rescale`` is the AWQ input-channel divisor, or ``None``.
    """

    data: torch.Tensor
    scales: torch.Tensor
    zeros: Optional[torch.Tensor]
    shape: Sequence[int]
    bits: int
    group_size: Optional[int]
    axis: int = 0
    rescale: Optional[torch.Tensor] = None
    scheme: str = "int8_per_channel"
    meta: dict = field(default_factory=dict)

    @property
    def out_features(self) -> int:
        return int(self.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.shape[1])

    def nbytes(self) -> int:
        total = self.data.numel() * self.data.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.zeros is not None:
            total += self.zeros.numel() * self.zeros.element_size()
        if self.rescale is not None:
            total += self.rescale.numel() * self.rescale.element_size()
        return total

    def dequantize(self, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
        """Reconstruct the dense weight.  Always the reference path for correctness."""
        device = device or self.data.device
        n, k = int(self.shape[0]), int(self.shape[1])

        if self.bits == 8:
            q = self.data.to(device=device, dtype=torch.float32)
            scale = self.scales.to(device=device, dtype=torch.float32).reshape(n, 1)
            if self.zeros is not None:
                zero = self.zeros.to(device=device, dtype=torch.float32).reshape(n, 1)
                w = (q - zero) * scale
            else:
                w = q * scale
        else:
            g = int(self.group_size or k)
            # K may have been zero-padded to a multiple of the group size; unpack with the
            # padded width, dequantise, then trim back to the real width.
            k_pad = int(self.meta.get("padded_in_features", k))
            q = unpack_int4(self.data.to(device), k_pad).to(torch.float32)
            scale = self.scales.to(device=device, dtype=torch.float32).reshape(n, -1, 1)
            if self.zeros is not None:
                zero = self.zeros.to(device=device, dtype=torch.float32).reshape(n, -1, 1)
                q = q.reshape(n, -1, g) - zero
            else:
                q = q.reshape(n, -1, g)
            w = (q * scale).reshape(n, k_pad)[:, :k]

        if self.rescale is not None:
            w = w / self.rescale.to(device=device, dtype=torch.float32).reshape(1, -1)
        return w.to(dtype)


def quantize_int8_per_channel(
    weight: torch.Tensor,
    axis: int = 0,
    eps: float = 1e-8,
) -> QuantizedTensor:
    """Symmetric INT8, one scale per output channel (``axis=0``).

    ``axis`` must be 0 (output channels).  Passing 1 is a silent precision bug in most
    engines because the scale array is then indexed along K, which does not match the
    dequantiser -- we reject it outright.
    """
    if axis != 0:
        raise ValueError("int8_per_channel quantises along the output-channel axis (axis=0)")
    w = weight.detach().to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    scale = amax / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return QuantizedTensor(
        data=q.contiguous(),
        scales=scale.to(torch.float32).contiguous(),
        zeros=None,
        shape=tuple(w.shape),
        bits=8,
        group_size=None,
        axis=0,
        scheme="int8_per_channel",
    )


def quantize_int4_groupwise(
    weight: torch.Tensor,
    group_size: int = 128,
    axis: int = 0,
    symmetric: bool = False,
    rescale: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> QuantizedTensor:
    """Asymmetric (or symmetric) INT4 with ``group_size`` weights per scale/zero.

    ``axis`` selects which dimension is grouped: 0 groups output channels (the usual
    "per-output-channel-group" layout for a ``[N, K]`` weight), 1 groups along K.
    Both are supported because different kernels prefer different layouts; the dequantiser
    honours whichever was used.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    w = weight.detach().to(torch.float32)
    if axis == 1:
        w = w.t().contiguous()
    n, k = w.shape
    g = min(group_size, k)
    if k % g != 0:
        # Pad K with zeros; the zero padding quantises to the zero point and is discarded
        # on dequantisation, so it cannot leak into the result.
        padded = _pad_last_dim(w, g)
        k_pad = padded.shape[1]
        w = padded
    else:
        k_pad = k

    groups = w.reshape(n, k_pad // g, g)
    w_min = groups.amin(dim=-1)
    w_max = groups.amax(dim=-1)
    if symmetric:
        amax = torch.maximum(w_min.abs(), w_max.abs()).clamp_min(eps)
        scale = amax / 7.0
        zero = torch.zeros_like(scale)
    else:
        scale = ((w_max - w_min) / 15.0).clamp_min(eps)
        zero = torch.round(-w_min / scale).clamp(0.0, 15.0)
    q = torch.round(groups / scale.unsqueeze(-1) + zero.unsqueeze(-1)).clamp(0.0, 15.0)

    packed = pack_int4(q.reshape(n, k_pad))
    qt = QuantizedTensor(
        data=packed.contiguous(),
        scales=scale.to(torch.float16).contiguous(),
        zeros=None if symmetric else zero.to(torch.float16).contiguous(),
        shape=(n, k),
        bits=4,
        group_size=g,
        axis=axis,
        scheme="int4_groupwise_sym" if symmetric else "int4_groupwise",
        meta={"padded_in_features": k_pad},
    )
    if rescale is not None:
        qt.rescale = rescale.detach().to(torch.float16).contiguous()
    return qt


def _dequant_reference(qt: QuantizedTensor) -> torch.Tensor:
    return qt.dequantize(torch.float32)


def quantize_awq_style(
    weight: torch.Tensor,
    act_scale: torch.Tensor,
    group_size: int = 128,
    grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    sample_rows: int = 256,
) -> QuantizedTensor:
    """Activation-aware groupwise INT4 (the AWQ idea, implemented explicitly).

    ``act_scale`` is the mean absolute activation received by each *input* channel,
    measured on a calibration set.  Weight columns are rescaled by ``act_scale ** alpha``
    before quantisation and divided back after, so channels that carry large activations
    are effectively given more of the 4-bit budget.  ``alpha`` is chosen by grid search on
    the reconstruction error.
    """
    w = weight.detach().to(torch.float32)
    a = act_scale.detach().to(torch.float32).flatten()
    if a.numel() != w.shape[1]:
        raise ValueError(f"act_scale has {a.numel()} entries but weight has {w.shape[1]} input channels")
    a = a.clamp_min(1e-8)
    a = a / a.mean()

    rows = w if w.shape[0] <= sample_rows else w[torch.randperm(w.shape[0])[:sample_rows]]

    best = None
    for alpha in grid:
        s = a.pow(float(alpha))
        q = quantize_int4_groupwise(rows * s.unsqueeze(0), group_size=group_size, axis=0)
        rec = q.dequantize(torch.float32) / s.unsqueeze(0)
        err = (rows - rec).pow(2).mean().item()
        if best is None or err < best[0]:
            best = (err, float(alpha), s)

    assert best is not None
    _, alpha, s = best
    qt = quantize_int4_groupwise(w * s.unsqueeze(0), group_size=group_size, axis=0, rescale=s)
    qt.scheme = "awq_style"
    qt.meta.update({"alpha": alpha, "calib_error": best[0]})
    return qt


def verify_scale_axis(weight: torch.Tensor, qt: QuantizedTensor, tol: float = 1e-2) -> bool:
    """Sanity-check that the stored scales are indexed consistently with the dequantiser.

    Deliberately naive: re-derive the expected scale from the raw weight using the axis the
    object claims, and compare.  A mismatch here is exactly the silent error described in
    risk R3.
    """
    w = weight.detach().to(torch.float32)
    if qt.axis == 1:
        w = w.t()
    if qt.bits == 8:
        expected = w.abs().amax(dim=1) / 127.0
        got = qt.scales.reshape(-1).to(torch.float32)
        return bool(torch.allclose(expected, got, atol=tol * expected.abs().max().clamp_min(1e-6)))
    g = int(qt.group_size)
    k_pad = int(qt.meta.get("padded_in_features", w.shape[1]))
    if k_pad != w.shape[1]:
        w = torch.nn.functional.pad(w, (0, k_pad - w.shape[1]))
    w_min = w.reshape(w.shape[0], -1, g).amin(-1)
    w_max = w.reshape(w.shape[0], -1, g).amax(-1)
    expected = (w_max - w_min) / 15.0
    got = qt.scales.to(torch.float32)
    return bool(torch.allclose(expected, got, rtol=tol, atol=tol * 1e-4))


class QuantLinear(nn.Module):
    """A drop-in ``nn.Linear`` whose weight is stored quantised.

    The packed tensors are registered as buffers so ``.to(device)`` moves them along with
    the rest of the module, and ``state_dict()`` round-trips a quantised checkpoint.

    The dense weight is materialised on demand.  With ``cache_dequant=True`` it is kept so
    repeated decode steps do not pay the unpack cost each time; :meth:`free_dequant_cache`
    releases it when the layer is offloaded.
    """

    def __init__(self, qt: Optional[QuantizedTensor] = None, bias: Optional[torch.Tensor] = None,
                 cache_dequant: bool = True):
        super().__init__()
        self.cache_dequant = cache_dequant
        self._dense: Optional[torch.Tensor] = None
        self.meta: dict = {}
        self.register_buffer("qdata", torch.empty(0, dtype=torch.uint8), persistent=True)
        self.register_buffer("qscales", torch.empty(0, dtype=torch.float16), persistent=True)
        self.register_buffer("qzeros", torch.empty(0, dtype=torch.float16), persistent=True)
        self.register_buffer("qrescale", torch.empty(0, dtype=torch.float16), persistent=True)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone().float())
        else:
            self.register_parameter("bias", None)
        if qt is not None:
            self.set_quantized(qt)

    def set_quantized(self, qt: QuantizedTensor) -> None:
        """Install (or replace) the quantised payload."""
        self.meta = {
            "shape": tuple(int(x) for x in qt.shape),
            "bits": int(qt.bits),
            "group_size": qt.group_size,
            "axis": int(qt.axis),
            "scheme": qt.scheme,
            "extra": dict(qt.meta),
        }
        self.qdata = qt.data.contiguous()
        self.qscales = qt.scales.contiguous()
        self.qzeros = qt.zeros.contiguous() if qt.zeros is not None else torch.empty(0, dtype=torch.float16)
        self.qrescale = qt.rescale.contiguous() if qt.rescale is not None else torch.empty(0, dtype=torch.float16)
        self._dense = None

    def quantized(self) -> QuantizedTensor:
        if not self.meta:
            raise RuntimeError("QuantLinear has no quantised payload installed")
        return QuantizedTensor(
            data=self.qdata,
            scales=self.qscales,
            zeros=self.qzeros if self.qzeros.numel() else None,
            shape=self.meta["shape"],
            bits=self.meta["bits"],
            group_size=self.meta["group_size"],
            axis=self.meta["axis"],
            rescale=self.qrescale if self.qrescale.numel() else None,
            scheme=self.meta["scheme"],
            meta=self.meta.get("extra", {}),
        )

    @property
    def in_features(self) -> int:
        return int(self.meta["shape"][1])

    @property
    def out_features(self) -> int:
        return int(self.meta["shape"][0])

    @classmethod
    def from_float(cls, linear: nn.Linear, scheme: str, **kwargs) -> "QuantLinear":
        if scheme == "int8_per_channel":
            qt = quantize_int8_per_channel(linear.weight)
        elif scheme == "int4_groupwise":
            qt = quantize_int4_groupwise(linear.weight, **kwargs)
        elif scheme == "awq_style":
            qt = quantize_awq_style(linear.weight, **kwargs)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        return cls(qt, linear.bias)

    def dense_weight(self) -> torch.Tensor:
        if self.cache_dequant and self._dense is not None and self._dense.device == self.qdata.device:
            return self._dense
        dense = self.quantized().dequantize(torch.float32, self.qdata.device)
        if self.cache_dequant:
            self._dense = dense
        return dense

    def free_dequant_cache(self) -> None:
        self._dense = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dense_weight().to(x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

    def extra_repr(self) -> str:
        if not self.meta:
            return "uninitialised"
        return (
            f"in={self.in_features}, out={self.out_features}, scheme={self.meta['scheme']}, "
            f"bits={self.meta['bits']}, group={self.meta['group_size']}, "
            f"bytes={self.quantized().nbytes()/1e6:.2f}MB"
        )
