"""Command line entry points.

``qwen25vl-lite plan``     show the memory plan for a target device (and the KV arithmetic)
``qwen25vl-lite probe``    report which SDPA backends actually work on this machine
``qwen25vl-lite tiers``    list the vision-token strategies
``qwen25vl-lite bench``    run a short benchmark against a model directory
``qwen25vl-lite serve``    start the OpenAI-compatible server
``qwen25vl-lite generate`` one-shot generation from a prompt

Run as ``python -m qwen25vl_lite.cli ...`` or via the ``qwen25vl-lite`` console script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import torch

from .attention import probe_backends
from .config import Qwen25VLConfig
from .engine import EngineConfig, LLMEngine, ModelRunner
from .model import apply_quantization, build_model, load_weights
from .planner import format_bytes, kv_bytes_per_token, plan_memory
from .sampling import SamplingParams
from .vision_budget import TIERS, summarise_tiers, tier_budget


def _device_memory(device: str) -> float:
    """Total bytes of the requested device, or a clear error if it cannot be queried."""
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = int(device.split(":")[1]) if ":" in device else 0
        return float(torch.cuda.get_device_properties(idx).total_memory)
    if device == "cpu":
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    raise RuntimeError(
        f"cannot query memory for {device!r}: CUDA is unavailable. Pass --vram-gb explicitly."
    )


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = Qwen25VLConfig.from_pretrained(args.model) if args.model else Qwen25VLConfig()
    vram = args.vram_gb * (1 << 30) if args.vram_gb else _device_memory(args.device)
    ram = args.ram_gb * (1 << 30) if args.ram_gb else _device_memory("cpu")

    kv = kv_bytes_per_token(cfg, args.kv_dtype)
    print(cfg.describe())
    print(f"KV cost: {kv.describe()}")
    print()
    plan = plan_memory(
        cfg,
        vram_bytes=vram,
        ram_bytes=ram,
        weight_dtype=args.weight_dtype,
        group_size=args.group_size,
        kv_dtype=args.kv_dtype,
        max_vision_tokens=TIERS[args.vision_tier].max_vision_tokens,
        target_context=args.context,
    )
    print(plan.summary())
    if args.json:
        print()
        print(json.dumps({
            "weights_bytes": plan.weights_bytes,
            "kv_per_token": plan.kv_per_token,
            "kv_pool_tokens": plan.kv_pool_tokens,
            "gpu_layers": plan.gpu_layers,
            "cpu_layers": plan.cpu_layers,
            "max_num_seqs": plan.max_num_seqs,
            "max_num_batched_tokens": plan.max_num_batched_tokens,
        }, indent=2))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    report = probe_backends(device=device, dtype=dtype, head_dim=args.head_dim,
                            num_heads=args.heads, num_kv_heads=args.kv_heads)
    print(report.summary())
    if device != args.device:
        print(f"  note: {args.device} requested but CUDA is unavailable; probed {device} instead")
    return 0


def cmd_tiers(args: argparse.Namespace) -> int:
    print(summarise_tiers())
    print()
    for name, tier in TIERS.items():
        b = tier_budget(tier)
        print(f"{name:<10} pixel ceiling {b.max_pixels:,} (min {b.min_pixels:,})")
    return 0


def _load(args: argparse.Namespace):
    cfg = Qwen25VLConfig.from_pretrained(args.model)
    model = build_model(cfg)
    if args.weight_dtype in ("int8", "int4", "awq"):
        scheme = {"int8": "int8_per_channel", "int4": "int4_groupwise", "awq": "awq_style"}[args.weight_dtype]
        apply_quantization(model, scheme, args.group_size)
    t0 = time.time()
    stats = load_weights(args.model, model, device=args.device,
                         dtype={"float16": torch.float16, "bfloat16": torch.bfloat16,
                                "float32": torch.float32}[args.dtype])
    model.eval()
    print(f"loaded {stats['loaded']} tensors ({stats['quantized']} quantised) in {time.time()-t0:.1f}s")
    return cfg, model


def _engine(args: argparse.Namespace, model):
    pinned = None
    if getattr(args, "attn_backend", None):
        pinned = [b.strip() for b in args.attn_backend.split(",") if b.strip()]
    ecfg = EngineConfig(
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_prefix_cache=not args.no_prefix_cache,
        enable_chunked_prefill=not args.no_chunked_prefill,
        quantize_kv=args.quantize_kv,
        kv_dtype=args.kv_dtype,
        vision_tier=args.vision_tier,
        enable_fastv=args.fastv,
        fastv_ratio=args.fastv_ratio,
        fastv_layer=args.fastv_layer,
        preemption=args.preemption,
        attn_backends=pinned,
        device=args.device,
        dtype={"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype],
    )
    runner = ModelRunner(model, ecfg, device=args.device)
    return LLMEngine(runner, ecfg), runner


def cmd_bench(args: argparse.Namespace) -> int:
    cfg, model = _load(args)
    engine, runner = _engine(args, model)
    print()
    print(runner.backend_report.summary() if runner.backend_report else f"pinned backends: {runner.backends}")
    print(f"KV cache allocated: {format_bytes(runner.cache.nbytes())} over {args.num_blocks} blocks")

    prompt = list(range(1000, 1000 + args.prompt_tokens))
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    t0 = time.time()
    engine.add_request(prompt, params)
    outs = engine.run_until_complete()
    elapsed = time.time() - t0
    n_out = len(outs[0].output_token_ids)
    print(f"prefill {args.prompt_tokens} tokens + decode {n_out} tokens in {elapsed:.2f}s")
    print(f"  decode throughput: {n_out / elapsed:.1f} tok/s (includes prefill)")
    print(f"  stats: {engine.stats()}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    cfg, model = _load(args)
    engine, runner = _engine(args, model)
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    ids = [int(t) for t in args.prompt_token_ids.split(",")] if args.prompt_token_ids else None
    if ids is None:
        print("--prompt-token-ids is required (this CLI does not bundle a tokenizer)", file=sys.stderr)
        return 2
    engine.add_request(ids, params)
    outs = engine.run_until_complete()
    print(outs[0].output_token_ids)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import EngineService, run_server

    cfg, model = _load(args)
    engine, runner = _engine(args, model)
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: no tokenizer ({exc}); only /v1/completions with prompt_token_ids will work",
              file=sys.stderr)
    service = EngineService(engine, tokenizer=tokenizer, model_name=os.path.basename(args.model.rstrip("/")))
    run_server(service, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen25vl-lite", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model_opts(p, require: bool = True):
        p.add_argument("--model", required=require, help="HuggingFace snapshot directory")
        p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0 ...")
        p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
        p.add_argument("--weight-dtype", default="float32",
                       choices=["float32", "float16", "bfloat16", "int8", "int4", "awq"])
        p.add_argument("--group-size", type=int, default=128)

    def add_engine_opts(p):
        p.add_argument("--num-blocks", type=int, default=2048)
        p.add_argument("--block-size", type=int, default=16)
        p.add_argument("--max-num-seqs", type=int, default=8)
        p.add_argument("--max-num-batched-tokens", type=int, default=4096)
        p.add_argument("--max-model-len", type=int, default=32768)
        p.add_argument("--kv-dtype", default="float16", choices=["float16", "bfloat16", "float32", "int8"])
        p.add_argument("--quantize-kv", action="store_true")
        p.add_argument("--no-prefix-cache", action="store_true")
        p.add_argument("--no-chunked-prefill", action="store_true")
        p.add_argument("--vision-tier", default="full", choices=sorted(TIERS))
        p.add_argument("--fastv", action="store_true", help="enable FastV-style image-token pruning")
        p.add_argument("--fastv-ratio", type=float, default=0.5)
        p.add_argument("--fastv-layer", type=int, default=2)
        p.add_argument("--preemption", default="swap", choices=["swap", "recompute"])
        p.add_argument("--attn-backend", default=None,
                       help="comma-separated SDPA backends to pin, e.g. math,flash "
                            "(default: probe at runtime and pick the fastest that works)")

    p = sub.add_parser("plan", help="show the memory plan and KV arithmetic")
    p.add_argument("--model", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--vram-gb", type=float, default=None)
    p.add_argument("--ram-gb", type=float, default=None)
    p.add_argument("--weight-dtype", default="int4")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--kv-dtype", default="float16", choices=["float16", "bfloat16", "float32", "int8"])
    p.add_argument("--vision-tier", default="full", choices=sorted(TIERS))
    p.add_argument("--context", type=int, default=None, help="target context length to sanity-check")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("probe", help="probe which SDPA backends work here")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=2)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("tiers", help="list vision-token strategies")
    p.set_defaults(func=cmd_tiers)

    p = sub.add_parser("bench", help="run a short benchmark")
    add_model_opts(p)
    add_engine_opts(p)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=64)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("generate", help="one-shot generation from token ids")
    add_model_opts(p)
    add_engine_opts(p)
    p.add_argument("--prompt-token-ids", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("serve", help="start the OpenAI-compatible server")
    add_model_opts(p)
    add_engine_opts(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
