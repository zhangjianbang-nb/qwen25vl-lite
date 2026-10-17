"""The fused INT4 kernel: opt-in, and numerically equal to the reference path.

The reference path is the definition of correct.  These tests run on CPU (the default state
and the unsupported-configuration guards) and, when a CUDA device and Triton are present, on
GPU against real quantised weights.
"""

from __future__ import annotations

import pytest
import torch

from qwen25vl_lite import int4_triton
from qwen25vl_lite.quant import (
    QuantLinear,
    int4_kernel_enabled,
    quantize_int4_groupwise,
    set_int4_kernel,
)

NEEDS_GPU = not (torch.cuda.is_available() and int4_triton.HAS_TRITON)
GPU_REASON = "needs CUDA and Triton"


@pytest.fixture(autouse=True)
def _reset_kernel_flag():
    """The flag is process-global; never let one test's choice leak into the next."""
    set_int4_kernel(False)
    yield
    set_int4_kernel(False)


def _qt(n: int, k: int, g: int = 128, device: str = "cpu"):
    torch.manual_seed(0)
    w = torch.randn(n, k, dtype=torch.float32) * 0.02
    qt = quantize_int4_groupwise(w, group_size=g)
    return type(qt)(
        data=qt.data.to(device), scales=qt.scales.to(device), zeros=qt.zeros.to(device),
        shape=qt.shape, bits=4, group_size=g, axis=0, scheme=qt.scheme, meta=qt.meta,
    )


def test_kernel_is_off_by_default():
    assert int4_kernel_enabled() is False


def test_can_use_rejects_unsupported_configurations_on_cpu():
    # No Triton on this interpreter, or no CUDA tensor: either way the reference path is kept.
    qt = _qt(64, 128)
    x = torch.randn(4, 128, dtype=torch.float16)
    assert int4_triton.can_use(x, qt) is False


@pytest.mark.skipif(NEEDS_GPU, reason=GPU_REASON)
def test_can_use_rejects_awq_and_odd_group():
    qt = _qt(256, 256, 128, device="cuda")
    x = torch.randn(4, 256, dtype=torch.float16, device="cuda")
    assert int4_triton.can_use(x, qt) is True

    awq = _qt(256, 256, 128, device="cuda")
    awq.rescale = torch.ones(256, dtype=torch.float16, device="cuda")
    assert int4_triton.can_use(x, awq) is False

    odd = _qt(256, 300, 100, device="cuda") if 300 % 100 == 0 else None
    if odd is not None:
        assert int4_triton.can_use(x, odd) is False


@pytest.mark.skipif(NEEDS_GPU, reason=GPU_REASON)
@pytest.mark.parametrize("n,k,g", [(256, 256, 128), (300, 384, 128), (512, 256, 64), (128, 128, 32)])
def test_fused_matches_reference(n, k, g):
    qt = _qt(n, k, g, device="cuda")
    for m in (1, 8, 64, 256):
        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        err = int4_triton.verify_against_reference(qt, x, atol=2e-2)
        assert err < 2e-2, f"n={n} k={k} g={g} m={m}: max error {err:.3e}"


@pytest.mark.skipif(NEEDS_GPU, reason=GPU_REASON)
def test_forward_switches_paths_without_changing_the_result():
    qt = _qt(256, 256, 128, device="cuda")
    lin = QuantLinear(qt, cache_dequant=False).to("cuda")
    x = torch.randn(4, 256, dtype=torch.float16, device="cuda")

    set_int4_kernel(False)
    reference = lin(x)
    set_int4_kernel(True)
    fused = lin(x)
    assert torch.allclose(reference.float(), fused.float(), atol=2e-2)
