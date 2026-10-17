"""Command line entry points.

``qwen25vl-lite plan``     show the memory plan for a target device (and the KV arithmetic)
``qwen25vl-lite probe``    report which SDPA backends actually work on this machine
``qwen25vl-lite tiers``    list the vision-token strategies
``qwen25vl-lite bench``    run a short benchmark against a model directory
``qwen25vl-lite serve``    start the OpenAI-compatible server
``qwen25vl-lite monitor``  poll a running server's /stats and print a live decode rate
``qwen25vl-lite camera``   describe live camera frames by posting them to a running server
``qwen25vl-lite generate`` one-shot generation from a prompt

Run as ``python -m qwen25vl_lite.cli ...`` or via the ``qwen25vl-lite`` console script.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from collections import deque
from typing import List, Optional

import torch

from .attention import probe_backends
from .config import Qwen25VLConfig
from .engine import EngineConfig, LLMEngine, ModelRunner
from .model import apply_quantization, build_model, load_weights
from .planner import format_bytes, kv_bytes_per_token, plan_memory
from .paged_attn_triton import set_paged_attn_kernel
from .quant import set_int4_kernel
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


def _enable_int4_kernel(model, device: str, dtype: torch.dtype) -> None:
    """Turn on the fused INT4 kernel only if it matches the reference path on real layers.

    A fused kernel that is subtly wrong produces plausible output, so the design doc's rule is
    to verify it against the reference path before trusting it.  If no quantised module has a
    layout the kernel supports, or the worst error is above tolerance, the reference path is
    kept and the reason is printed -- silently falling back would hide a configuration mistake.
    """
    from .int4_triton import can_use, verify_against_reference

    mods = model.quantized_modules()
    if not mods:
        print("--int4-kernel: model has no quantised modules; using the reference path",
              file=sys.stderr)
        return

    checked, worst = 0, 0.0
    for m in mods[:4]:
        qt = m.quantized()
        probe = torch.empty(8, m.in_features, device=device, dtype=dtype)
        if not can_use(probe, qt):
            continue
        worst = max(worst, verify_against_reference(qt, dtype=dtype))
        checked += 1

    if checked == 0:
        print("--int4-kernel: no quantised module matches the kernel's layout "
              "(needs int4 groupwise, group in 32/64/128/256, no AWQ rescale); "
              "using the reference path", file=sys.stderr)
        return

    atol = 2e-2
    if worst > atol:
        print(f"--int4-kernel: self-check failed, worst relative |fused - reference| = {worst:.2e} > {atol:.0e}; "
              "using the reference path", file=sys.stderr)
        return

    set_int4_kernel(True)
    print(f"--int4-kernel: enabled; worst relative |fused - reference| = {worst:.2e} over {checked} layer(s)")


def _enable_paged_attn_kernel(model, device: str, dtype: torch.dtype) -> None:
    """Turn on the fused paged decode-attention kernel only if it matches the eager path.

    The check runs on a synthetic paged cache rather than a served request, because it has to
    happen before any request exists.  A wrong attention kernel produces fluent text rather
    than an error, so a failed check keeps the reference path and says so.
    """
    from .paged_attn_triton import can_use, verify_against_reference

    if not str(device).startswith("cuda"):
        print("--paged-attn-kernel: needs a CUDA device; using the reference path", file=sys.stderr)
        return

    text = model.cfg.text_config
    heads, kv_heads, head_dim = text.num_attention_heads, text.num_key_value_heads, text.head_dim
    if head_dim not in (64, 128, 256):
        print(f"--paged-attn-kernel: head_dim {head_dim} is not 64/128/256; "
              "using the reference path", file=sys.stderr)
        return

    block_size, num_blocks = 16, 64
    gen = torch.Generator(device=device).manual_seed(0)
    k_cache = torch.randn(num_blocks, block_size, kv_heads, head_dim, device=device,
                          dtype=dtype, generator=gen)
    v_cache = torch.randn(num_blocks, block_size, kv_heads, head_dim, device=device,
                          dtype=dtype, generator=gen)
    block_tables = torch.arange(32, device=device).view(1, 32)
    context_lens = torch.tensor([500], device=device)
    q = torch.randn(1, heads, 1, head_dim, device=device, dtype=dtype, generator=gen)

    if not can_use(q, k_cache, v_cache, context_lens):
        print("--paged-attn-kernel: configuration not supported by the kernel "
              "(needs fp16/bf16, a power-of-two block size, head_dim in 64/128/256); "
              "using the reference path", file=sys.stderr)
        return

    worst = verify_against_reference(q, k_cache, v_cache, block_tables, context_lens)
    atol = 2e-2
    if worst > atol:
        print(f"--paged-attn-kernel: self-check failed, worst relative "
              f"|fused - reference| = {worst:.2e} > {atol:.0e}; using the reference path",
              file=sys.stderr)
        return

    set_paged_attn_kernel(True)
    print(f"--paged-attn-kernel: enabled; worst relative |fused - reference| = {worst:.2e}")


def _engine(args: argparse.Namespace, model):
    if getattr(args, "int4_kernel", False):
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[args.dtype]
        _enable_int4_kernel(model, args.device, dtype)
    if getattr(args, "paged_attn_kernel", False):
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[args.dtype]
        _enable_paged_attn_kernel(model, args.device, dtype)
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
        cuda_graph=getattr(args, "cuda_graph", False),
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


def _gpu_sample() -> Optional[dict]:
    """GPU utilisation and memory from nvidia-smi, or None if it is unavailable.

    Returns a dict rather than a formatted string so ``--json`` can carry the raw numbers.
    """
    import subprocess

    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        )
    except Exception:  # noqa: BLE001 - monitoring must not fail because nvidia-smi is absent
        return None
    rows = done.stdout.strip().splitlines()
    if not rows:
        return None
    try:
        util, used, total = (float(x.strip()) for x in rows[0].split(","))
    except ValueError:
        return None
    return {"util_pct": util, "mem_used_mib": used, "mem_total_mib": total}


def _gpu_line(sample: Optional[dict]) -> str:
    if sample is None:
        return ""
    return (f"gpu {sample['util_pct']:3.0f}%  "
            f"{sample['mem_used_mib'] / 1024:5.1f}/{sample['mem_total_mib'] / 1024:.1f} GiB")


# One glyph per sample, low to high, for the history column.
_SPARK_GLYPHS = "▁▂▃▄▅▆▇█"


def _sparkline(values: List[float], width: int) -> str:
    """Render the recent rates as a fixed-width history bar.

    A flat series is drawn at mid height rather than at the bottom, so a steady rate reads as a
    steady line instead of as "nothing happening"; a genuinely idle series (all zeros) stays at
    the bottom.
    """
    if width <= 0:
        return ""
    recent = list(values)[-width:]
    if not recent:
        return " " * width
    lo, hi = min(recent), max(recent)
    if hi <= lo:
        return (_SPARK_GLYPHS[3] if lo > 0 else _SPARK_GLYPHS[0]) * len(recent)
    span = hi - lo
    return "".join(
        _SPARK_GLYPHS[min(len(_SPARK_GLYPHS) - 1, int((v - lo) / span * len(_SPARK_GLYPHS)))]
        for v in recent
    )


def cmd_monitor(args: argparse.Namespace) -> int:
    """Poll a running server's /stats and render it as a live line.

    A /stats round trip measures 0.43 ms on localhost, so the default 0.5 s interval is set by
    what is comfortable to read, not by cost -- ``--interval 0.1`` gives 10 Hz for well under a
    percent of a core.  nvidia-smi is different: it costs ~16 ms per call, so it is refreshed on
    its own slower cadence (``--gpu-interval``) and reused in between.

    The rate comes from the server's rolling window of decode-step timings, and the server
    zeroes it once the engine has gone quiet, so an idle server reads 0.0 rather than holding
    the last value it saw.  The ``idle`` column says how long it has been quiet.
    """
    import urllib.error
    import urllib.request

    url = args.url.rstrip("/")
    width = args.sparkline
    header = (f"{'time':8s} {'decode tok/s':>12s} {'step ms':>8s} {'run':>4s} {'wait':>5s} "
              f"{'swap':>5s} {'kv blocks':>13s} {'prefix hit':>10s} {'idle':>6s} "
              f"{'gpu':<22s} {'history':<{width}s}")
    overwrite = not args.json and sys.stdout.isatty() and not args.rows
    history: deque = deque(maxlen=max(width, 1))
    gpu_sample, gpu_at = None, -1e18

    if not args.json:
        print(header, flush=True)

    samples = 0
    try:
        while True:
            now = time.perf_counter()
            stamp = time.strftime("%H:%M:%S")
            error = None
            try:
                with urllib.request.urlopen(f"{url}/stats", timeout=args.timeout) as resp:
                    stats = json.load(resp)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                stats, error = None, f"{type(exc).__name__} ({url})"

            if args.json:
                print(json.dumps({"time": stamp, "error": error, **(stats or {})}), flush=True)
            else:
                if error is not None:
                    line = f"{stamp:8s} unreachable: {error}"
                else:
                    rate = stats.get("decode_tokens_per_s", 0.0)
                    history.append(rate)
                    if now - gpu_at >= args.gpu_interval:
                        gpu_sample, gpu_at = _gpu_sample(), now
                    idle = stats.get("decode_idle_s", -1.0)
                    idle_text = ("—" if stats.get("decode_active", 0.0)
                                 else ("never" if idle < 0 else f"{idle:.1f}s"))
                    line = (f"{stamp:8s} {rate:12.1f} "
                            f"{stats.get('decode_step_ms', 0.0):8.2f} "
                            f"{int(stats.get('running', 0)):4d} "
                            f"{int(stats.get('waiting', 0)):5d} "
                            f"{int(stats.get('swapped', 0)):5d} "
                            f"{int(stats.get('kv_blocks_used', 0)):6d}/"
                            f"{int(stats.get('kv_blocks_total', 0)):<6d} "
                            f"{stats.get('prefix_hit_rate', 0.0):10.1%} "
                            f"{idle_text:>6s} "
                            f"{_gpu_line(gpu_sample):<22s} "
                            f"{_sparkline(history, width)}")
                # Padded to the header's width so an overwriting '\r' line leaves no tail of
                # the previous, longer line behind.
                print(("\r" if overwrite else "") + line.ljust(len(header)),
                      end="" if overwrite else "\n", flush=True)

            samples += 1
            if args.count and samples >= args.count:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    if overwrite:
        print()
    return 0


def _downscale(rgb, width: int):
    """Shrink an RGB array to ``width`` columns, preserving the aspect ratio.

    Vision tokens scale with *area*, so halving the width cuts them roughly fourfold.  This is
    the client-side half of the vision-token lever: it is under the caller's control and needs
    no server restart, unlike the server's ``--vision-tier``.
    """
    if width <= 0:
        return rgb
    from PIL import Image

    import numpy as np

    height, cols = rgb.shape[:2]
    if width >= cols:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize(
        (width, max(1, round(height * width / cols))), Image.BILINEAR))


def _rgb_to_data_url(rgb, quality: int = 85) -> str:
    """JPEG-encode an RGB array and wrap it in the ``data:`` URL the server's parser accepts.

    JPEG rather than PNG because a camera frame is photographic: at 640x480 this is ~44 KB
    against ~600 KB, and the vision tower downsamples it anyway.
    """
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _post_image(base_url: str, data_url: str, prompt: str, max_tokens: int,
                timeout: float, extra: Optional[dict] = None) -> dict:
    """Send one image plus a prompt to a running ``serve`` and return its parsed reply.

    ``extra`` carries request-body fields such as ``vision_tier`` / ``max_vision_tokens``.  The
    server reads the image pixel budget from the *request*, not from its ``--vision-tier`` flag
    (that flag only reaches ``EngineConfig.vision_tier``, which drives FastV pruning), so this
    is the only way to ask for a smaller vision budget over HTTP.
    """
    import urllib.request

    body = {
        "model": "qwen25vl-lite",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        **(extra or {}),
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _open_camera(device: int):
    """Open a V4L2 camera.  OpenCV is an optional extra, so the import is deferred."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SystemExit(
            "the camera subcommand needs OpenCV: pip install 'qwen25vl-lite[camera]'"
        ) from exc

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"could not open camera {device} (/dev/video{device})")
    # A 30 fps camera behind a ~600 ms loop would otherwise hand back frames from hundreds of
    # milliseconds ago, which is not what "live" means here.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _grab_rgb(cap):
    """Newest frame from ``cap`` as an RGB array, discarding whatever is already buffered.

    ``retrieve`` on its own can return a stale frame, so the buffer is drained with ``grab``
    first.  cv2 hands back BGR; the model wants RGB.
    """
    import cv2

    for _ in range(2):
        if not cap.grab():
            return None
    ok, frame = cap.retrieve()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def cmd_camera(args: argparse.Namespace) -> int:
    """Describe live camera frames by posting each one to a running ``serve``.

    The frames go to the *running server* rather than a second in-process copy of the model:
    the served model already holds ~5.9 GB of an 8 GB card, so a second copy would OOM.  One
    frame at 640x480 costs ~618 ms on an RTX 5060 (421 prompt tokens, a short completion), so
    the loop runs at ~1.6 frames/s -- fast enough for scene-level description, and the honest
    ceiling for this card.  ``--interval`` adds deliberate spacing on top of that.
    """
    import urllib.error

    url = args.url.rstrip("/")
    if args.interval < 0:
        raise SystemExit("--interval must not be negative")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must not be negative")
    if args.vision_tier not in TIERS:
        raise SystemExit(f"unknown --vision-tier {args.vision_tier!r}; known: {sorted(TIERS)}")
    if args.max_vision_tokens is not None and args.max_vision_tokens <= 0:
        raise SystemExit("--max-vision-tokens must be positive")
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # The image budget comes from the request body, not from the server's --vision-tier flag
    # (that one only reaches EngineConfig.vision_tier, which drives FastV pruning), so these
    # fields are what actually shrink the vision token count.  "full" is the server default,
    # so it is not sent.
    extra: dict = {}
    if args.vision_tier != "full":
        extra["vision_tier"] = args.vision_tier
    if args.max_vision_tokens is not None:
        extra["max_vision_tokens"] = args.max_vision_tokens

    cap = _open_camera(args.device)
    frames, failures = 0, 0
    try:
        while True:
            rgb = _grab_rgb(cap)
            if rgb is None:
                failures += 1
                print("camera returned no frame", file=sys.stderr, flush=True)
                if failures > 10:
                    return 1
                time.sleep(0.2)
                continue

            rgb = _downscale(rgb, args.width)
            if args.save_dir:
                from PIL import Image
                Image.fromarray(rgb).save(
                    os.path.join(args.save_dir, f"frame-{frames:05d}.jpg"),
                    format="JPEG", quality=args.jpeg_quality)

            t0 = time.perf_counter()
            try:
                reply = _post_image(url, _rgb_to_data_url(rgb, args.jpeg_quality),
                                    args.prompt, args.max_tokens, args.timeout, extra or None)
                text = reply["choices"][0]["message"]["content"].strip()
                error = None
            except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
                text, error = "", f"{type(exc).__name__}: {exc}"
                failures += 1
            elapsed = time.perf_counter() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            if args.json:
                print(json.dumps({"time": time.strftime("%H:%M:%S"), "frame": frames,
                                  "latency_s": elapsed, "fps": fps,
                                  "text": text, "error": error}), flush=True)
            elif error is not None:
                print(f"[{time.strftime('%H:%M:%S')}] {elapsed * 1000:6.0f} ms  "
                      f"ERROR {error}", flush=True)
            elif args.quiet:
                print(text, flush=True)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] {elapsed * 1000:6.0f} ms "
                      f"({fps:.2f} fps)  {text}", flush=True)

            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
            if args.interval:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()

    # Every frame failing means the server never answered; a partial failure is reported in
    # line but is not a reason to fail the whole run.
    return 1 if frames and failures == frames else 0


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
    service = EngineService(engine, tokenizer=tokenizer,
                            model_name=os.path.basename(args.model.rstrip("/")),
                            model_dir=args.model)
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
        p.add_argument("--int4-kernel", action="store_true",
                       help="use the fused Triton INT4 dequant+GEMM kernel (opt-in; it changes "
                            "fp16 summation order, so it is verified against the reference path "
                            "at startup and refused if the check fails)")
        p.add_argument("--attn-backend", default=None,
                       help="comma-separated SDPA backends to pin, e.g. math,flash "
                            "(default: probe at runtime and pick the fastest that works)")
        p.add_argument("--cuda-graph", action="store_true",
                       help="capture the decode step into a CUDA graph (opt-in). A decode step "
                            "is launch-bound, so this removes the CPU dispatch cost. Each graph "
                            "is checked against the eager forward at capture time and refused "
                            "if it differs at all")
        p.add_argument("--paged-attn-kernel", action="store_true",
                       help="use the fused Triton flash-decoding kernel for paged decode "
                            "attention (opt-in). It reads the block table directly instead of "
                            "gathering a dense context and masking it; verified against the "
                            "eager path at startup and refused if the check fails")

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

    p = sub.add_parser("monitor", help="live view of a running server's decode rate and KV usage")
    p.add_argument("--url", default="http://127.0.0.1:8081",
                   help="base URL of a running `serve` instance (its /stats endpoint)")
    p.add_argument("--interval", type=float, default=0.5,
                   help="seconds between /stats samples (default 0.5 = 2 Hz; a sample costs "
                        "~0.43 ms, so 0.1 is affordable if you want 10 Hz)")
    p.add_argument("--gpu-interval", type=float, default=1.0,
                   help="seconds between nvidia-smi calls, reused in between (each call costs "
                        "~16 ms, ~40x a /stats round trip)")
    p.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout per sample")
    p.add_argument("--count", type=int, default=0,
                   help="stop after this many samples (0 = run until interrupted)")
    p.add_argument("--rows", action="store_true",
                   help="print one line per sample instead of overwriting a single line "
                        "(also the default when stdout is not a tty)")
    p.add_argument("--sparkline", type=int, default=24,
                   help="width of the recent-rate history bar (0 disables it)")
    p.add_argument("--json", action="store_true",
                   help="emit one JSON object per sample instead of a formatted line")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("camera", help="describe live camera frames in real time")
    p.add_argument("--url", default="http://127.0.0.1:8081",
                   help="base URL of a running `serve` instance; frames are posted to it "
                        "rather than loading a second copy of the model")
    p.add_argument("--device", type=int, default=0,
                   help="V4L2 device index, i.e. /dev/videoN (default 0)")
    p.add_argument("--prompt", default="Describe what you see in one short sentence.",
                   help="the question asked about every frame")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="completion cap per frame (a one-sentence description needs ~16)")
    p.add_argument("--interval", type=float, default=0.0,
                   help="extra seconds to sleep between frames (0 = as fast as the server "
                        "answers; the model itself sets the pace)")
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per frame")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after this many frames (0 = run until interrupted)")
    p.add_argument("--jpeg-quality", type=int, default=85,
                   help="JPEG quality for the frame sent to the model")
    p.add_argument("--width", type=int, default=0,
                   help="downscale each frame to this many columns before sending it "
                        "(0 = native). Vision tokens scale with area, so this is the cheapest "
                        "lever on the per-frame latency; the server's --vision-tier only bites "
                        "at coarser settings")
    p.add_argument("--vision-tier", default="full", choices=sorted(TIERS),
                   help="per-request vision budget tier (default full). Sent in the request "
                        "body, because the server reads the image budget from the request, not "
                        "from its own --vision-tier flag")
    p.add_argument("--max-vision-tokens", type=int, default=None,
                   help="per-request ceiling on vision tokens, overriding --vision-tier; the "
                        "most direct control over the vision budget")
    p.add_argument("--save-dir", default=None,
                   help="also write every captured frame to this directory")
    p.add_argument("--json", action="store_true",
                   help="emit one JSON object per frame instead of a formatted line")
    p.add_argument("--quiet", action="store_true",
                   help="print only the descriptions, with no timestamp or latency prefix")
    p.set_defaults(func=cmd_camera)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
