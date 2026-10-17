"""Benchmark harness: prefill and decode throughput, and where the memory went.

This is the *only* place in the repository that is meant to be run on the target card.  It
exists so that the numbers in a report are yours, measured on your hardware, rather than
numbers copied from a blog post.

    python bench/run_bench.py --model /path/to/Qwen2.5-VL-3B-Instruct --tier half --quant int4
    python bench/run_bench.py --model /path/to/model --prompt-len 2048 --gen-len 128 --batch-size 4

Two safety behaviours are deliberate:

* **Memory guard.**  Before anything is allocated it computes the weight footprint from the
  config and refuses to proceed if it does not fit in the free memory it can see.  On a
  unified-memory machine (GB10, Apple Silicon) a large model here will evict whatever else is
  running -- including a served model -- so this guard is not a nicety.
* **No implicit download.**  ``--model`` must be a local directory.  Nothing is fetched.

Reported numbers, and what they mean:

* ``prefill_tok_per_s``  -- prompt tokens processed per second, per sequence.
* ``decode_tok_per_s``   -- generated tokens per second, aggregated over the batch.
* ``ttft_ms``            -- time to first token, including image processing and vision encode.
* ``peak_mem_mb``        -- peak device memory (CUDA) or peak RSS (CPU), whichever applies.
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import sys
import time
from typing import List, Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qwen25vl_lite.config import Qwen25VLConfig, bytes_per_weight  # noqa: E402
from qwen25vl_lite.engine import EngineConfig, LLMEngine, ModelRunner  # noqa: E402
from qwen25vl_lite.model import apply_quantization, build_model, load_weights  # noqa: E402
from qwen25vl_lite.planner import format_bytes, kv_bytes_per_token, plan_memory  # noqa: E402
from qwen25vl_lite.quant import int4_kernel_enabled, set_int4_kernel  # noqa: E402
from qwen25vl_lite.paged_attn_triton import paged_attn_kernel_enabled  # noqa: E402
from qwen25vl_lite.sampling import SamplingParams  # noqa: E402


def free_bytes() -> int:
    """Free memory this process could realistically claim.

    ``MemAvailable`` rather than ``MemFree``: page cache is reclaimable, and on unified-memory
    machines the distinction is the difference between "looks full" and "is full".
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 1 << 62


def device_memory_used(device: str) -> int:
    if device == "cuda":
        return torch.cuda.max_memory_allocated()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def reset_memory_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def check_memory(cfg: Qwen25VLConfig, dtype_name: str, quant: str, group_size: int,
                 device: str, force: bool, margin: float) -> int:
    """Refuse to load a model that will not fit.  Returns the estimated weight bytes.

    The *scheme* decides bytes per weight, not the compute dtype: an int4 checkpoint stored
    at fp16 activations is still 0.5 bytes per weight plus scale/zero metadata.
    """
    if quant == "int8":
        bpw = bytes_per_weight("int8")
    elif quant in ("int4", "awq"):
        bpw = bytes_per_weight("int4", group_size or 128)
    else:
        bpw = bytes_per_weight(dtype_name)
    weight_bytes = int(sum(cfg.num_params().values()) * bpw)
    if device == "cuda":
        available = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    else:
        available = free_bytes()
    if weight_bytes > available * margin and not force:
        print(f"REFUSING to load: weights need ~{format_bytes(weight_bytes)}, "
              f"but only ~{format_bytes(available)} is available "
              f"(guard margin {margin:.0%}).")
        print("This machine may be serving a model; loading here would evict it. "
              "Re-run with --force only if you are certain the memory is yours.")
        raise SystemExit(2)
    return weight_bytes


def make_prompt_ids(vocab: int, length: int) -> List[int]:
    """Deterministic pseudo-prompt.  Benchmarks do not need meaningful text, only a
    realistic length and a stable token distribution (so prefix caching behaves the same
    way on every run)."""
    return [(7 * i + 11) % vocab for i in range(length)]


def bench(args) -> dict:
    device = args.device
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device == "cuda" and dtype == torch.float32:
        print("note: float32 on CUDA is unusual; the engine's quantised kernels are the point here")

    cfg = Qwen25VLConfig.from_pretrained(args.model)
    group_size = 128 if args.quant in ("int4", "awq") else 0
    check_memory(cfg, args.dtype, args.quant, group_size, device, args.force, args.memory_margin)

    weight_dtype = args.quant if args.quant != "float32" else args.dtype
    plan = plan_memory(cfg, vram_bytes=args.vram_gb * (1 << 30), ram_bytes=args.ram_gb * (1 << 30),
                       weight_dtype=weight_dtype, group_size=group_size, kv_dtype=args.dtype,
                       block_size=args.block_size, max_num_seqs=args.batch_size,
                       target_context=args.max_model_len)
    kv = kv_bytes_per_token(cfg, args.dtype)
    print(f"KV per token: {format_bytes(kv.per_token)}  ({kv.describe()})")
    print(f"GPU-resident KV pool: {format_bytes(plan.kv_pool_bytes)} -> "
          f"{plan.kv_pool_tokens:,} tokens; decoder layers resident: "
          f"{plan.gpu_layers}/{plan.gpu_layers + plan.cpu_layers}")
    print(f"Logical context (max_model_len): {args.max_model_len:,} tokens -- a scheduling "
          f"bound, not an allocation")
    for note in plan.notes:
        print(f"  note: {note}")

    reset_memory_peak(device)
    t0 = time.perf_counter()
    model = build_model(cfg)
    if args.quant != "float32":
        # ``--quant`` names the storage scheme the CLI exposes; ``apply_quantization`` takes
        # the internal scheme name, so the two must be mapped (they are not the same string).
        scheme = {"int8": "int8_per_channel", "int4": "int4_groupwise", "awq": "awq_style"}[args.quant]
        apply_quantization(model, scheme, group_size)
    load_weights(args.model, model, device=device, dtype=dtype)
    model.eval()
    load_s = time.perf_counter() - t0

    if args.int4_kernel:
        from qwen25vl_lite.int4_triton import can_use, verify_against_reference

        worst, checked = 0.0, 0
        for m in model.quantized_modules()[:4]:
            qt = m.quantized()
            if not can_use(torch.empty(8, m.in_features, device=device, dtype=dtype), qt):
                continue
            worst = max(worst, verify_against_reference(qt, dtype=dtype))
            checked += 1
        if checked and worst <= 2e-2:
            set_int4_kernel(True)
            print(f"fused INT4 kernel: ON (worst |fused - reference| = {worst:.2e} over {checked} layers)")
        else:
            print(f"fused INT4 kernel: refused (checked={checked}, worst={worst:.2e}); reference path kept")
    print(f"fused INT4 kernel enabled: {int4_kernel_enabled()}")

    if args.paged_attn_kernel:
        from qwen25vl_lite.paged_attn_triton import set_paged_attn_kernel
        from qwen25vl_lite.paged_attn_triton import verify_against_reference as verify_paged

        text = cfg.text_config
        gen = torch.Generator(device=device).manual_seed(0)
        bs, nb = 16, 64
        kc = torch.randn(nb, bs, text.num_key_value_heads, text.head_dim, device=device,
                         dtype=dtype, generator=gen)
        vc = torch.randn(nb, bs, text.num_key_value_heads, text.head_dim, device=device,
                         dtype=dtype, generator=gen)
        bt = torch.arange(32, device=device).view(1, 32)
        ctx = torch.tensor([500], device=device)
        qq = torch.randn(1, text.num_attention_heads, 1, text.head_dim, device=device,
                         dtype=dtype, generator=gen)
        worst = verify_paged(qq, kc, vc, bt, ctx)
        if worst <= 2e-2:
            set_paged_attn_kernel(True)
            print(f"fused paged attention: ON (worst |fused - reference| = {worst:.2e})")
        else:
            print(f"fused paged attention: refused (worst={worst:.2e}); reference path kept")
    print(f"fused paged attention enabled: {paged_attn_kernel_enabled()}")

    ecfg = EngineConfig(
        num_blocks=max(plan.kv_pool_tokens // args.block_size + 8, 64),
        block_size=args.block_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_prefix_cache=args.prefix_cache,
        vision_tier=args.tier,
        # Without these the KV cache would be allocated in EngineConfig's float32 default,
        # i.e. twice the planned footprint, and the run dies on the first KV allocation.
        device=device,
        dtype=dtype,
        cuda_graph=args.cuda_graph,
        cuda_graph_bucket=args.cuda_graph_bucket,
    )
    runner = ModelRunner(model, ecfg, device=device)
    engine = LLMEngine(runner, ecfg)
    if runner.decode_graphs is not None:
        print(f"cuda graph decode: enabled (bucket {ecfg.cuda_graph_bucket} blocks)")

    vocab = cfg.text_config.vocab_size
    prompt_ids = [make_prompt_ids(vocab, args.prompt_len) for _ in range(args.batch_size)]

    t0 = time.perf_counter()
    for ids in prompt_ids:
        engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=args.gen_len,
                                               ignore_eos=True))
    outputs = engine.run_until_complete()
    wall = time.perf_counter() - t0

    stats = engine.stats()
    gen_tokens = sum(len(o.output_token_ids) for o in outputs)
    prefill_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    ttfts = [o.metrics.get("ttft_s", 0.0) for o in outputs]

    result = {
        "load_s": load_s,
        "wall_s": wall,
        "prefill_tokens": prefill_tokens,
        "prefill_tok_per_s": prefill_tokens / wall,
        "gen_tokens": gen_tokens,
        "decode_tok_per_s": gen_tokens / max(wall - max(ttfts, default=0.0), 1e-9),
        "ttft_ms": (max(ttfts) if ttfts else 0.0) * 1000,
        "peak_mem_mb": device_memory_used(device) / 1024 / 1024,
        "kv_blocks_used": stats.get("kv_blocks_used"),
    }

    print()
    print(f"load:                 {load_s:.2f} s")
    print(f"prefill:              {prefill_tokens} tok in {wall:.2f} s "
          f"-> {result['prefill_tok_per_s']:.1f} tok/s")
    print(f"decode:               {gen_tokens} tok -> {result['decode_tok_per_s']:.1f} tok/s "
          f"(batch {args.batch_size})")
    print(f"TTFT:                 {result['ttft_ms']:.1f} ms")
    print(f"peak device memory:   {result['peak_mem_mb']:.1f} MB")
    print(f"KV blocks in flight:  {result['kv_blocks_used']}")
    print()
    print("These are measurements of *this* configuration only. The README makes no claims "
          "about numbers it has not seen produced.")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="local model directory (nothing is downloaded)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--quant", default="float32",
                   choices=["float32", "int8", "int4", "awq"],
                   help="float32 means no weight quantisation")
    p.add_argument("--tier", default="full", help="vision tier: full / half / thumbnail / pruned")
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--gen-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--prefix-cache", action="store_true")
    p.add_argument("--int4-kernel", action="store_true",
                   help="enable the fused Triton INT4 dequant+GEMM kernel (self-checked first)")
    p.add_argument("--cuda-graph", action="store_true",
                   help="capture the decode step into a CUDA graph (self-checked against eager)")
    p.add_argument("--cuda-graph-bucket", type=int, default=16,
                   help="round the graph's block-table width up to a multiple of this many "
                        "blocks, so a graph is re-captured every bucket*block_size tokens "
                        "instead of every 16")
    p.add_argument("--paged-attn-kernel", action="store_true",
                   help="use the fused Triton flash-decoding kernel for paged decode "
                        "attention (self-checked against the eager path first)")
    p.add_argument("--vram-gb", type=float, default=8.0, help="the target card's memory, for the plan")
    p.add_argument("--ram-gb", type=float, default=96.0, help="the target machine's RAM, for the plan")
    p.add_argument("--memory-margin", type=float, default=0.5,
                   help="refuse if weights exceed this fraction of available memory")
    p.add_argument("--force", action="store_true", help="override the memory guard")
    args = p.parse_args()

    if not os.path.isdir(args.model):
        print(f"--model must be a local directory; got {args.model!r}")
        return 1
    bench(args)
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
