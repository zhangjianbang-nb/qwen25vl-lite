"""Quality gates A / B / C.

Three gates, deliberately separate, because the review of 2026-10-15 drew a line between two
kinds of risk that are easy to conflate:

* **Gate A -- reference equivalence.** Quantisation and pruning off. Compares the engine
  against HuggingFace `transformers` on the same inputs. A failure here is a *bug*.
* **Gate B -- quantisation regression.** Each weight scheme evaluated separately against the
  float baseline on a fixed prompt set. A change here is a quality trade-off, and it must be
  measured rather than assumed.
* **Gate C -- vision-strategy regression.** Each vision tier evaluated separately against
  `full`. `full`'s correctness is *not* evidence for `half` or `thumbnail`.

Usage::

    python scripts/quality_gate.py --model /path/to/Qwen2.5-VL-3B-Instruct --gate A
    python scripts/quality_gate.py --model /path/to/model --gate B --schemes float32,int8,int4
    python scripts/quality_gate.py --model /path/to/model --gate C --tiers full,half,thumbnail

Without ``--model`` it builds a tiny random-weight model, so the harness itself is runnable
anywhere.  The numbers it prints on random weights are meaningless as *quality*; they are
meaningful as *mechanism* (does the pipeline run, and does the difference grow as expected).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qwen25vl_lite.config import Qwen25VLConfig, bytes_per_weight  # noqa: E402
from qwen25vl_lite.engine import EngineConfig, LLMEngine, ModelRunner  # noqa: E402
from qwen25vl_lite.imageproc import process_image  # noqa: E402
from qwen25vl_lite.model import apply_quantization, build_model, load_weights  # noqa: E402
from qwen25vl_lite.rope import mm_token_type_ids  # noqa: E402
from qwen25vl_lite.sampling import SamplingParams  # noqa: E402
from qwen25vl_lite.vision_budget import TIERS, tier_budget  # noqa: E402

# Fixed prompt set: the same inputs for every gate, so runs are comparable across changes.
TEXT_PROMPTS: List[List[int]] = [
    [3, 4, 5, 6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    list(range(30, 60)),
]

# Memory guard.  On a unified-memory machine, loading a model large enough to matter here
# evicts whatever else is running -- possibly the model the user is actually serving.  The
# gates therefore refuse to proceed unless the weights fit in a fraction of reclaimable
# memory, and `--force` is the explicit override.
_GUARD_MARGIN = 0.5
_FORCE = False


def _synthetic_image(size: Tuple[int, int] = (300, 420), seed: int = 0) -> Image.Image:
    """A structured image (bands + blocks) rather than pure noise, so pooling has something to
    destroy and the tier comparison is not degenerate."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for row in range(size[0]):
        arr[row, :, :] = (row % 40) * 6
    for _ in range(12):
        y = int(rng.integers(0, size[0] - 20))
        x = int(rng.integers(0, size[1] - 40))
        arr[y : y + 12, x : x + 30, :] = rng.integers(0, 255, size=(12, 30, 3))
    return Image.fromarray(arr)


def _free_bytes() -> int:
    """Reclaimable memory, not ``MemFree``.  On a unified-memory machine the difference is
    'looks full' versus 'is full'."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 1 << 62


def _storage_bytes_per_weight(scheme: str, dtype: torch.dtype) -> float:
    """Bytes per stored weight: the *scheme* decides, not the compute dtype."""
    if scheme == "int8":
        return bytes_per_weight("int8")
    if scheme in ("int4", "awq"):
        return bytes_per_weight("int4", 128)
    return bytes_per_weight(str(dtype).replace("torch.", ""))


def _guard(model_dir: str, scheme: str, dtype: torch.dtype) -> None:
    """Refuse to load a model that would evict whatever else this machine is serving.

    A gate that loads a 7B model on a box already serving one does not produce a quality
    number, it produces an outage.
    """
    if _FORCE:
        return
    cfg = Qwen25VLConfig.from_pretrained(model_dir)
    need = sum(cfg.num_params().values()) * _storage_bytes_per_weight(scheme, dtype)
    avail = _free_bytes()
    if need > avail * _GUARD_MARGIN:
        raise SystemExit(
            f"refusing to load {model_dir}: weights need ~{need / 1e9:.2f} GB, "
            f"only ~{avail / 1e9:.2f} GB is reclaimable.  Run this gate on the target card, "
            f"or pass --force if this machine is dedicated to the gate."
        )


def _load(model_dir: str, scheme: str, dtype: torch.dtype, device: str):
    _guard(model_dir, scheme, dtype)
    cfg = Qwen25VLConfig.from_pretrained(model_dir)
    model = build_model(cfg)
    if scheme in ("int8", "int4", "awq"):
        apply_quantization(model, {"int8": "int8_per_channel", "int4": "int4_groupwise",
                                   "awq": "awq_style"}[scheme], 128)
    load_weights(model_dir, model, device=device, dtype=dtype)
    model.eval()
    return cfg, model


def _engine(model, cfg, tier: str, **overrides):
    base = dict(num_blocks=512, block_size=16, max_num_seqs=4, max_num_batched_tokens=4096,
                max_model_len=4096, enable_prefix_cache=False, vision_tier=tier)
    base.update(overrides)
    ecfg = EngineConfig(**base)
    runner = ModelRunner(model, ecfg, device="cpu")
    return LLMEngine(runner, ecfg), runner


def _greedy(model_dir: str, scheme: str, dtype: torch.dtype, prompts: List[List[int]],
            tier: str = "full", n_tokens: int = 6) -> List[List[int]]:
    cfg, model = _load(model_dir, scheme, dtype, "cpu")
    engine, _ = _engine(model, cfg, tier)
    for ids in prompts:
        engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=n_tokens, ignore_eos=True))
    outs = engine.run_until_complete()
    by_id = {o.seq_id: o for o in outs}
    return [by_id[i].output_token_ids for i in range(len(prompts))]


def gate_a(model_dir: str, dtype: torch.dtype) -> Dict[str, float]:
    """Engine vs HuggingFace, quantisation off."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    _guard(model_dir, "float32", dtype)
    hf = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype)
    hf = hf.to(dtype).eval()
    hf.model.rope_deltas = None
    cfg, model = _load(model_dir, "float32", dtype, "cpu")

    # A *relative* criterion, matching tests/test_model_vs_hf.py.  An absolute threshold is
    # the wrong tool here: a model with random weights produces logits of order 1-10, so a
    # 4.5e-3 absolute gap is 1e-3 relative and means "float32 rounding", not "wrong
    # computation".  Gate A's claim is exactness, and `allclose` is how exactness is stated.
    atol, rtol = 1e-4, 1e-3
    worst_abs = 0.0
    worst_rel = 0.0
    rows = []
    with torch.no_grad():
        for ids in TEXT_PROMPTS:
            mm = mm_token_type_ids(torch.tensor([ids]), cfg.image_token_id, cfg.video_token_id)
            ref = hf(input_ids=torch.tensor([ids]), mm_token_type_ids=mm).logits[:, -1, :]
            mine = model(input_ids=torch.tensor([ids]))[:, -1, :]
            diff = (ref - mine).abs()
            worst_abs = max(worst_abs, float(diff.max()))
            worst_rel = max(worst_rel, float((diff / (ref.abs() + 1e-6)).max()))
            rows.append({"prompt_len": len(ids), "max_abs_diff": float(diff.max()),
                         "close": bool(torch.allclose(ref, mine, atol=atol, rtol=rtol))})

    img = _synthetic_image()
    proc = process_image(img, tier_budget(TIERS["full"]))
    n_vis = int(proc["grid_thw"].prod().item() // 4)
    ids = [1, 2, cfg.vision_start_token_id] + [cfg.image_token_id] * n_vis + [cfg.vision_end_token_id, 3, 4]
    # `mm_token_type_ids` is not optional in practice.  Without it, HF's
    # `compute_3d_position_ids` sets `can_compute_mrope = False` and silently falls back to
    # plain 1-D positions, ignoring the image grid -- so an "equivalence" check that omits it
    # is comparing against a model that did not use MRoPE at all.  That is exactly the trap
    # this gate exists to avoid.
    ids_t = torch.tensor([ids])
    mm = mm_token_type_ids(ids_t, cfg.image_token_id, cfg.video_token_id)
    with torch.no_grad():
        ref = hf(input_ids=ids_t, pixel_values=proc["pixel_values"],
                 image_grid_thw=proc["grid_thw"], mm_token_type_ids=mm).logits[:, -1, :]
        mine = model(input_ids=ids_t, pixel_values=proc["pixel_values"],
                     image_grid_thw=proc["grid_thw"])[:, -1, :]
    mm_diff = (ref - mine).abs()
    multimodal_close = bool(torch.allclose(ref, mine, atol=atol, rtol=rtol))
    multimodal_abs = float(mm_diff.max())

    passed = multimodal_close and all(r["close"] for r in rows)
    return {"worst_abs_diff": max(worst_abs, multimodal_abs),
            "worst_rel_diff": worst_rel,
            "multimodal_abs_diff": multimodal_abs,
            "multimodal_close": multimodal_close,
            "atol": atol, "rtol": rtol,
            "rows": rows, "pass": passed}


def gate_b(model_dir: str, dtype: torch.dtype, schemes: List[str], n_tokens: int) -> Dict[str, object]:
    """Each weight scheme vs the float baseline."""
    baseline = _greedy(model_dir, "float32", dtype, TEXT_PROMPTS, n_tokens=n_tokens)
    report: Dict[str, object] = {"baseline": baseline, "schemes": {}}
    for scheme in schemes:
        got = _greedy(model_dir, scheme, dtype, TEXT_PROMPTS, n_tokens=n_tokens)
        agree = sum(
            1 for a, b in zip(baseline, got) for x, y in zip(a, b) if x == y
        ) / max(1, sum(len(a) for a in baseline))
        report["schemes"][scheme] = {"token_agreement": agree, "outputs": got}
    return report


def gate_c(model_dir: str, dtype: torch.dtype, tiers: List[str], n_tokens: int) -> Dict[str, object]:
    """Each vision tier vs `full`, on the same images.

    The images are deliberately large (900x1200).  On a small image every tier resolves to the
    same grid -- correctly, because the image already fits under the tightest ceiling -- and
    the gate would compare three identical runs.
    """
    images = [_synthetic_image((900, 1200), seed=s) for s in range(3)]
    cfg = Qwen25VLConfig.from_pretrained(model_dir)
    report: Dict[str, object] = {"images": len(images), "tiers": {}}
    baseline_outputs: Optional[List[List[int]]] = None
    baseline_vision_tokens: Optional[List[int]] = None

    for tier in tiers:
        _, model = _load(model_dir, "float32", dtype, "cpu")
        engine, runner = _engine(model, cfg, tier)
        outs_tokens: List[List[int]] = []
        vision_counts: List[int] = []
        for img in images:
            proc = process_image(img, tier_budget(TIERS[tier]))
            n_vis = int(proc["grid_thw"].prod().item() // 4)
            vision_counts.append(n_vis)
            ids = ([1, 2, cfg.vision_start_token_id] + [cfg.image_token_id] * n_vis
                   + [cfg.vision_end_token_id, 3, 4])
            engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=n_tokens, ignore_eos=True),
                               pixel_values=proc["pixel_values"], image_grid_thw=proc["grid_thw"])
        outs = engine.run_until_complete()
        by_id = {o.seq_id: o for o in outs}
        for i in range(len(images)):
            outs_tokens.append(by_id[i].output_token_ids)
        entry = {"vision_tokens": vision_counts, "outputs": outs_tokens}
        if tier == "full":
            baseline_outputs = outs_tokens
            baseline_vision_tokens = vision_counts
        elif baseline_outputs is not None:
            entry["token_agreement_vs_full"] = sum(
                1 for a, b in zip(baseline_outputs, outs_tokens) for x, y in zip(a, b) if x == y
            ) / max(1, sum(len(a) for a in baseline_outputs))
            entry["vision_token_ratio_vs_full"] = [
                c / max(1, f) for c, f in zip(vision_counts, baseline_vision_tokens or vision_counts)
            ]
        report["tiers"][tier] = entry
    return report


def main() -> int:
    global _FORCE, _GUARD_MARGIN

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="model directory; omit with --tiny")
    parser.add_argument("--tiny", action="store_true",
                        help="build a tiny random-weight model instead of loading a real one "
                             "(a few hundred KB; for checking that the harness runs)")
    parser.add_argument("--gate", default="A", choices=["A", "B", "C", "all"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--schemes", default="int8,int4")
    parser.add_argument("--tiers", default="full,half,thumbnail")
    parser.add_argument("--max-tokens", type=int, default=6)
    parser.add_argument("--memory-margin", type=float, default=0.5,
                        help="refuse to load weights exceeding this fraction of reclaimable memory")
    parser.add_argument("--force", action="store_true",
                        help="override the memory guard (only when this machine is dedicated to the gate)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    _GUARD_MARGIN = args.memory_margin
    _FORCE = args.force

    model_dir = args.model
    if model_dir is None:
        if not args.tiny:
            print("error: pass --model <local model dir>, or --tiny to use a random fixture.")
            print("       Nothing is downloaded, and nothing large is loaded on a machine that")
            print("       may already be serving a model.")
            return 1
        from make_tiny_model import make_tiny

        model_dir = os.path.join(tempfile.gettempdir(), "qwen25vl-lite-gate")
        make_tiny(model_dir)
        print(f"no --model given; using tiny random model at {model_dir}")
        print("NOTE: with random weights the *quality* numbers are meaningless; the gates only")
        print("      demonstrate that the mechanism runs and that differences are detected.\n")

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    result: Dict[str, object] = {}

    if args.gate in ("A", "all"):
        result["A"] = gate_a(model_dir, dtype)
        r = result["A"]
        print(f"Gate A (vs transformers): worst abs diff {r['worst_abs_diff']:.3e}, "
              f"worst rel diff {r['worst_rel_diff']:.3e}, "
              f"multimodal {'close' if r['multimodal_close'] else 'DIVERGES'} "
              f"(atol={r['atol']}, rtol={r['rtol']}) -> {'PASS' if r['pass'] else 'FAIL'}")

    if args.gate in ("B", "all"):
        result["B"] = gate_b(model_dir, dtype, args.schemes.split(","), args.max_tokens)
        for scheme, entry in result["B"]["schemes"].items():
            print(f"Gate B: {scheme:<10} token agreement vs float baseline "
                  f"{entry['token_agreement']*100:.1f}%")

    if args.gate in ("C", "all"):
        result["C"] = gate_c(model_dir, dtype, args.tiers.split(","), args.max_tokens)
        for tier, entry in result["C"]["tiers"].items():
            extra = ""
            if "token_agreement_vs_full" in entry:
                extra = (f", agreement vs full {entry['token_agreement_vs_full']*100:.1f}%, "
                         f"vision-token ratio {entry['vision_token_ratio_vs_full']}")
            print(f"Gate C: {tier:<10} vision tokens {entry['vision_tokens']}{extra}")

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
