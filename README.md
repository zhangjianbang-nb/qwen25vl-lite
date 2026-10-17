# qwen25vl-lite

[中文说明](README.zh-CN.md)

A small, readable Qwen2.5-VL inference engine aimed at **8 GB consumer GPUs** (RTX 5060 /
4060 class) with plenty of host RAM, and at any machine where you would rather read the code
than trust it.

It is not a vLLM replacement. It is a complete, self-contained implementation of the pieces
that actually matter for fitting a vision-language model into a small card and serving it
with sane throughput — and a test suite that checks those pieces against HuggingFace
`transformers` numerically instead of asserting shapes and hoping.

```bash
pip install -e ".[hf]"            # torch + transformers (the server needs the tokenizer)
pip install -e ".[dev]"           # ... plus pytest and the pinned transformers the tests expect
qwen25vl-lite plan --model ./Qwen2.5-VL-3B-Instruct --vram-gb 8 --ram-gb 96 --weight-dtype int4
qwen25vl-lite serve --model ./Qwen2.5-VL-3B-Instruct --device cuda --weight-dtype int4 \
                    --kv-dtype float16 --num-blocks 4000 \
                    --int4-kernel --paged-attn-kernel --cuda-graph
qwen25vl-lite monitor --url http://127.0.0.1:8081     # live decode rate, KV usage and GPU load
qwen25vl-lite camera --url http://127.0.0.1:8081      # describe live camera frames (~1.9 fps)
```

The `qwen25vl-lite` entry point lives in whichever environment you pip-installed into. If your
shell is activated on a *different* env (a bare `base`, say), that command is not on `PATH` and
you get `command not found` — activate the env it was installed into, or call it by full path.
Either way the module is importable from any directory:

```bash
conda activate dl                                  # ... or whichever env holds the install
qwen25vl-lite monitor --url http://127.0.0.1:8081
/home/whu/miniconda3/envs/dl/bin/qwen25vl-lite monitor --url http://127.0.0.1:8081
/home/whu/miniconda3/envs/dl/bin/python -m qwen25vl_lite.cli monitor --url http://127.0.0.1:8081
```

`monitor` polls the running server's `/stats` (default 0.5 s; `--interval 0.1` for 10 Hz) and
rewrites a single line. During a request, and then into the idle period after it:

```
time     decode tok/s  step ms  run  wait  swap     kv blocks prefix hit   idle gpu                    history
07:58:58         77.6    11.68    1     0     0      9/4000         0.0%      — gpu  98%    5.9/8.0 GiB ▁█
07:58:58         87.2    11.37    1     0     0     11/4000         0.0%      — gpu  98%    5.9/8.0 GiB ▁▅█
07:59:00         60.6    13.22    1     0     0     17/4000         0.0%      — gpu  98%    6.0/8.0 GiB ▂▆█▆▄▆█▆▁
07:59:01         75.1    13.17    0     0     0      2/4000         0.0%      — gpu  98%    6.0/8.0 GiB ▂▆█▆▄▆█▆▁▃▇▅
07:59:01          0.0     0.00    0     0     0      2/4000         0.0%   0.5s gpu  97%    6.0/8.0 GiB ▇███▇███▆▇█▇▇▁
07:59:03          0.0     0.00    0     0     0      2/4000         0.0%   2.1s gpu   0%    6.0/8.0 GiB ▇███▇███▆▇█▇▇▁▁▁▁▁▁▁
```

The rate comes from the server's rolling window of the last 64 decode steps, and the server
counts only the steps recent enough to describe the present — so an idle engine reads `0.0` and
the `idle` column counts how long it has been quiet, rather than holding the last rate it saw.
Polling is cheap: a `/stats` round trip is 0.43 ms idle and 1.23 ms median under load, so 10 Hz
costs about a percent of a core. `nvidia-smi` is ~40x that, so it is refreshed on its own
`--gpu-interval` cadence and reused in between. `--json` emits one object per sample for
scripting, and `--sparkline 0` drops the history column.

`camera` grabs frames from a V4L2 device and posts each one to the running server, so it reuses
the model that is already loaded instead of needing a second copy (the served model holds
~5.9 GB of an 8 GB card):

```
$ qwen25vl-lite camera --url http://127.0.0.1:8081 --device 0 --max-frames 3
[08:16:44]    521 ms (1.92 fps)  A bed with a computer monitor on a table in the background.
[08:16:44]    504 ms (1.99 fps)  A cluttered room with a bed, desk, and computer monitor.
[08:16:45]    516 ms (1.94 fps)  A cluttered room with a bed, desk, and computer monitor.
```

At 640x480 that is ~510 ms per frame, about **1.9 frames/s** — scene-level description, not
video rate. That is the honest ceiling for this card: the vision tower plus a short completion
costs what it costs, and `--interval` only adds spacing on top. OpenCV is an optional extra
(`pip install 'qwen25vl-lite[camera]'`); `--quiet` prints just the descriptions, `--json` emits
one object per frame, and `--save-dir` also keeps the frames it captured.

Vision tokens are where the per-frame time goes, and there are two levers — `--width` shrinks
the frame before sending it, `--max-vision-tokens` caps the budget the server applies. Measured
on one native 640x480 frame, normalised to a fixed 16-token answer:

| setting | vision tokens | prefill | fps |
|---|---|---|---|
| native 640, full (default) | 421 | 251 ms | 2.31 |
| `--width 320` | 129 | 133 ms | **3.20** (+38.5%) |
| `--max-vision-tokens 128` | 147 | 138 ms | **3.23** (+40.1%) |
| `--vision-tier thumbnail` | 264 | 184 ms | 2.66 (+15.3%) |

Descriptions hold up at the lower budgets — the same frame gave "A person is laying on a bed
with a phone and a pillow" at 421 tokens and "A person is lying on a bed with a phone in their
hand" at 129. Below ~130 tokens the prefill flattens onto a ~130 ms floor, so smaller frames
buy nothing further. Note that `--vision-tier` and `--max-vision-tokens` are *request* fields:
the server's own `--vision-tier` flag does not set the image budget, it only drives FastV
pruning.

---

## What is actually verified

| Claim | How it is checked | Status |
|---|---|---|
| MRoPE matches HF **bit for bit** | `tests/test_model_vs_hf.py::test_mrope_cos_sin_and_apply_match_hf` uses `torch.equal` (zero tolerance) | passing |
| Vision tower matches HF | `test_vision_tower_matches_hf`, `atol=1e-5` | passing |
| End-to-end logits match HF (text and multimodal) | `test_full_logits_match_hf`, `test_text_only_logits_match_hf` | passing |
| Multimodal logits match HF when windowed vision attention engages | `test_multimodal_logits_match_hf_with_windowing_engaged` (a 22x30 patch grid = 12 windows) | passing |
| Chunked prefill == single-pass prefill | 20 combinations of chunk size {16, 8, 5, 3} x length {15, 16, 17, 33, 64}, compared to HF | passing |
| Paged decode == HF's own incremental decoding, step by step | `test_paged_decode_matches_hf_incremental` | passing |
| Batched decode == sequential decode (no KV-padding leakage) | `test_decode_batch_matches_independent_single_sequences` | passing |
| Prefix-cache reuse == recomputing the prefix | `test_prefix_cache_reuses_and_matches_fresh_compute` | passing |
| Two requests sharing a prefix reuse it (not just identical prompts) | `test_requests_sharing_a_prefix_reuse_it`, `tests/test_prefix_block_keys.py` | passing |
| The sync-free paged path == the reference gather/mask | `tests/test_paged_fast_path.py` | passing |
| Greedy engine output == `transformers.generate` | `test_greedy_generation_matches_hf_generate` | passing |
| Preemption keeps the block pool in budget and every request still finishes | `test_preemption_keeps_the_pool_within_budget` (+ a recompute variant), which also asserts the pool returns to empty | passing |
| Quantiser is self-consistent (scale axis, INT4 packing, AWQ objective) | `test_quantization_error_bounds`, `test_int4_pack_unpack_roundtrip`, `test_awq_beats_plain_groupwise`, `test_symmetric_int4_preserves_negative_weights` | passing |
| The prefix cache evicts and returns blocks instead of leaking | `test_prefix_cache_eviction_actually_evicts_and_returns_blocks` | passing |
| The block pool rejects unknown and double frees | `test_block_manager_rejects_bogus_frees` | passing |
| Sampling penalties and validation actually apply | `test_engine_applies_sampling_penalties`, `test_engine_rejects_invalid_and_unsupported_sampling_params` | passing |
| The fused paged decode-attention kernel == the eager gather + length-mask + SDPA path | `tests/test_paged_attn_kernel.py` (8 shapes: batch 1-4, head_dim 64/128/256, block_size 16/32, fp16 and bf16) | passing |
| A padded block table cannot change the fused kernel's result | `test_padded_block_table_tail_does_not_change_the_result` | passing |
| CUDA-graph decode == eager decode, token for token | `tests/test_cuda_graph.py` (also asserts the capture-time difference is exactly `0.0`) | passing |
| The `/stats` payload is JSON-serialisable and reports a live decode rate | `tests/test_engine.py` (`json.dumps` the payload; the rate must be 0 before any decode step and > 0 after one, and must equal the rolling-window average rather than a cumulative tally) | passing |
| A decode rate goes to zero once the engine is idle instead of holding a stale value | `test_decode_rate_falls_to_zero_once_the_engine_goes_idle` (a 10 s old step stops counting; `decode_active`, `decode_idle_s` and both rate columns follow) | passing |
| `monitor` renders that payload, shows an idle engine as idle, and survives an unreachable server | `tests/test_monitor.py` (a stub HTTP server, `--json` output, plus a refused connection and a payload from an older server that lacks the new keys) | passing |
| `camera` posts each frame as a JPEG data URL, prints the reply, and fails loudly when the server never answers | `tests/test_camera.py` (the frame source is monkeypatched, so it needs neither a camera nor OpenCV; a stub server records what was posted, including the request-level vision budget) | passing |

```
$ python -m pytest tests/
176 passed, 2 skipped
```

Everything runs on CPU with tiny random weights and **no model download**, so the suite is
usable in CI. (`pyproject.toml` already sets `addopts = "-q"`, so pass `-q` again only if you
want the summary line suppressed.)

## What is *not* verified — read this before quoting a number

Being explicit here is part of the design, not a disclaimer bolted on at the end:

* **The throughput figures below are one machine's, not a guarantee.** They were measured on
  the RTX 5060 named in the design notes, so they are an existence proof rather than a
  portable claim; another card will differ. No memory figure is quoted at all.
  `bench/run_bench.py` and `qwen25vl-lite bench` exist so *you* can produce both on your own
  hardware.
* **INT4 quality on real Qwen2.5-VL weights is measured only at the mechanism level.** The
  quantiser is unit-tested for reconstruction error and the AWQ objective is tested to be
  non-worse than plain groupwise on a synthetic case. Gate B
  (`scripts/quality_gate.py --gate B --schemes int8,int4`) on the real 3B checkpoint gives
  **66.7% (int8) and 38.9% (int4)** token agreement against its float32 baseline over the
  harness's fixed three-prompt, six-token-per-prompt set. That is 18 tokens of a mechanism
  check, not a benchmark: "how much accuracy does INT4 cost on DocVQA" still needs a task
  dataset, which this repository does not ship.
* **The fused INT4 kernel and the fused paged-attention kernel are opt-in and are not the
  tested path.** `--int4-kernel` selects a Triton kernel (`qwen25vl_lite/int4_triton.py`) that
  applies `(q - zero) * scale` in registers and feeds the GEMM directly, so the fp16 copy
  `dense_weight()` materialises is never written; `--paged-attn-kernel` selects a Triton
  flash-decoding kernel (`qwen25vl_lite/paged_attn_triton.py`) that reads the paged cache
  through the block table instead of gathering a dense context and masking it. Both are off
  by default because they change the summation order of the path the correctness claims above
  describe. Each runs its own self-check at startup and refuses itself if the check fails.
* **`--cuda-graph` captures the decode step into a CUDA graph.** A decode step is
  launch-bound, not compute-bound, so this removes the CPU dispatch cost. It does not change
  any number -- a replay is the same kernel sequence -- and each graph is checked against the
  eager forward at capture time. The block-table width is bucketed (`--cuda-graph-bucket`),
  which is what keeps captures rare enough to pay for themselves.
* **The three flags are multiplicative, and measured together** on an RTX 5060 with
  Qwen2.5-VL-3B INT4/group-128, batch 1, 128 generated tokens
  (`bench/run_bench.py --quant int4 --int4-kernel --paged-attn-kernel --cuda-graph`):

  | prompt | decode, reference INT4 path | decode, all three | TTFT before | TTFT after |
  |---|---|---|---|---|
  | 512 | 52.1 tok/s | **73.2 tok/s** | 336 ms | 223 ms |
  | 2,048 | 47.9 tok/s | **67.8 tok/s** | 832 ms | 722 ms |
  | 4,096 | 39.9 tok/s | **64.5 tok/s** | 1,908 ms | 1,791 ms |

  Individually: `--int4-kernel` alone takes decode 4.4 -> 51.1 tok/s and prefill 33.0 -> 180.6
  (and its GEMV path for the M == 1 decode shape takes a step's 252 projections from 8.4 to
  7.3 ms); `--cuda-graph` alone takes 512-token decode 51.3 -> 61.8 but *slows* the 2,048-token
  case (47.9 -> 45.4), because there the eager step is already GPU-bound and the graph only adds
  scheduling cost; `--paged-attn-kernel` alone takes 2,048-token decode 47.9 -> 49.7 but slows
  512 (52.1 -> 50.5), because two Triton launches per layer cost more to enqueue than the one
  SDPA call they replace. They are worth enabling together, not separately.
* **The decode decay with context is much flatter now.** With the reference gather, 512 ->
  4,096 tokens cost 1.31x; with the fused attention kernel the gather and the padded attention
  no longer grow with the context, so it costs 1.14x. Batching still recovers throughput
  (164 tok/s aggregate at batch 4) without raising per-sequence speed.
* **The CPU fallback is the tested path.** CUDA-specific behaviour (which SDPA backend is
  selected, PCIe transfer overlap) is *probed at runtime* and reported, never assumed.
* **FastV-style pruning is off by default and is approximate.** See the section below.

## The three context numbers

A frequent source of dishonest READMEs is quoting one number for "context length" when three
different limits exist. This engine keeps them apart, and `qwen25vl-lite plan` prints all
three.

For Qwen2.5-VL-3B the KV cost per token is computed, not guessed:

```
2 (K and V) x 36 layers x 2 kv heads x 128 head_dim x 2 bytes = 36,864 bytes = 36 KiB/token
```

So `250,000` tokens of fp16 KV is **8.58 GiB — more than the entire card**. Any plan that
claims 250k tokens on 8 GB is talking about swap capacity, not GPU-resident capacity.

| number | meaning |
|---|---|
| logical context | the longest sequence accepted; bounded by `max_position_embeddings` and by what fits after swapping |
| GPU-resident KV | tokens whose KV lives in VRAM; these decode at full speed |
| CPU swap capacity | extra tokens that can be paged to host memory; decode over these is slower and must be measured |

A worked plan for the target machine:

```
$ qwen25vl-lite plan --vram-gb 8 --ram-gb 96 --weight-dtype int4 --context 32768
KV cost: 2 x 36 layers x 2 kv_heads x 128 head_dim x 2 B = 36.0 KiB/token

memory plan
  weights (int4 )        3.53 GiB
  KV per token             36.0 KiB
  GPU-resident KV pool     3.00 GiB (87,350 tokens, float16)
  vision/prefill reserve   95.00 MiB
  runtime + allocator      589.82 MiB
  ----------------------------------------
  total on GPU             7.20 GiB
  decoder layers on GPU    36 of 36
  max concurrent sequences 42
  prefill chunk size       3328
  note: host memory could hold roughly 2,690,615 more KV tokens (92.37 GiB); these are swap capacity, not GPU-resident capacity.
  note: logical context ceiling is 128,000 tokens from the config; the GPU-resident figure above is what is fast, and the host figure is what is merely possible.
```

The runtime reserve is **8% of the usable budget**, computed rather than hard-coded: CUDA
context size, cuBLAS workspaces and allocator slack all scale with workload, so a constant
like "500 MB" would be a guess dressed up as a fact.

## Architecture

```
qwen25vl_lite/
  config.py         config.json parsing (both the legacy `rope_scaling` key and `rope_parameters`)
  rope.py           MRoPE for text, 2-D RoPE for vision, `get_rope_index`
  quant.py          INT8 per-channel, INT4 groupwise, activation-aware (AWQ-style) INT4
  int4_triton.py    opt-in fused INT4 dequantise + GEMM, self-checked against the reference
  attention.py      runtime SDPA backend probing, paged attention reference, masks
  paged_attn_triton.py  opt-in fused paged decode attention (flash-decoding), self-checked
  imageproc.py      smart_resize + patchification matching the HF processor
  vision.py         the vision tower (window attention + PatchMerger)
  text.py           the decoder (GQA, SwiGLU, MRoPE, paged/chunked/dense attention)
  layers.py         shared RMSNorm (plain formulation, float32 compute)
  model.py          assembly, streaming safetensors loading, forward pass
  cache.py          block manager, paged KV cache, radix prefix cache
  graphs.py         opt-in CUDA-graph capture of the decode step, self-checked
  vision_budget.py  resolution budgets, pooling, ToME merge, FastV selection
  planner.py        device-memory planning (three context numbers kept separate)
  offload.py        static CPU/GPU layer placement with double-buffered prefetch
  sampling.py       penalties, top-k/top-p/min-p, per-sequence RNG
  engine.py         continuous batching, chunked prefill, preemption, FastV prefill
  server.py         OpenAI-compatible HTTP API (stdlib only)
  cli.py            `plan`, `probe`, `tiers`, `bench`, `generate`, `serve`, `monitor`, `camera`
```

Supporting scripts: `scripts/serve.sh` (an example launcher for an 8 GB card, every value
overridable from the environment), `scripts/demo.py` (end-to-end walkthrough),
`scripts/make_tiny_model.py` (the random fixture used by the gates and tests),
`scripts/quality_gate.py` (Gates A/B/C), `bench/run_bench.py` (throughput and memory on your
hardware).

### The five things that make it fit in 8 GB

1. **Weight-only quantisation.** INT4 groupwise (g=128 by default) brings 3B weights from
   7.5 GB (bf16) to ~3.5 GB. INT8 per-channel is the quality baseline. A third scheme,
   activation-aware INT4 (`awq_style`), rescales weight columns by measured activation
   magnitudes and picks the exponent by grid search — this is the *actual* AWQ idea, and the
   plain INT8 path is never called AWQ, because it is not.
2. **Paged KV cache.** 16-token blocks, reference counted, so a sequence grows incrementally
   instead of reserving `max_model_len` up front. Optional INT8 KV halves the footprint with
   a per-(token, head) scale, which keeps writes order-independent under chunked prefill.
3. **Radix prefix cache.** A multi-turn VL conversation re-sends the same image and history.
   Re-prefilling an image costs seconds; reusing its KV costs nothing. The cache key is token
   ids **plus** a fingerprint of the vision embeddings and the RoPE configuration, because
   `<|image_pad|>` runs of equal length are indistinguishable by token id alone.
4. **Chunked prefill.** A long prompt is prefilled in pieces, bounding the activation peak.
   The absolute position of each chunk is handled explicitly (mask offset `context_len -
   query_len`), and the equivalence with a single-pass prefill is tested at several chunk
   sizes.
5. **Vision token budget.** The resolution ceiling is expressed in *vision tokens*, which is
   what actually costs prefill time, and it is clamped in three places (`min_pixels`,
   `max_pixels`, token budget).

### Vision reduction: one lever per tier, no quality promises

The external review was blunt that token pruning is **not** lossless on Qwen2.5-VL, and that
FastV cannot be treated as a plug-in optimisation because the tower mixes window attention
(`window_size=112`) with global attention at `fullatt_block_indexes = [7, 15, 23, 31]`.

So each tier applies **exactly one** reduction, so that a quality change can be attributed to
it. Tiers are listed in increasing order of risk:

| tier | mechanism | measured token ratio vs `full` | expected cost |
|---|---|---|---|
| `full` | full dynamic resolution, ceiling 1280 vision tokens | 1.00 | the quality reference |
| `half` | the same pipeline with the ceiling halved to 640 | 0.495 | small text gets blurrier |
| `thumbnail` | ceiling reduced to 256 | 0.190 | fine detail and OCR degrade noticeably |
| `pruned` | full resolution, then FastV drops the 50% lowest-attention image tokens after layer 2 | 1.00 (drops *after* encoding) | task-dependent; validate with `scripts/quality_gate.py` |

The ratios are the ones `scripts/quality_gate.py --gate C` prints on a 900x1200 image; they
are a property of the ceilings, not of any particular model. `pruned` keeps all tokens
through the tower and prunes inside the language model, so its *prompt* is unchanged until
pruning and its ratio is reported as 1.00 at the vision-token stage.

**Resolution is the only lever wired into a tier.** It is also the honest one: fewer vision
tokens is exactly what prefill time and KV pressure are made of, and it needs no change to
the checkpoint's feature geometry.

`pool_merge` (2x2 grid-preserving average pool) and `tome_merge` (bipartite soft matching)
are implemented and unit-tested, but they are **not** wired into any tier, and the README
does not claim otherwise. Both replace trained feature geometry — a pool averages merge units
the checkpoint never averaged, ToME fuses tokens by similarity — so their quality cost is
model-specific and must be measured per model rather than assumed. Hiding that measurement
behind a tier name would be exactly the kind of unverified claim this repository avoids.
`pool_merge` keeps the layout rectangular, so MRoPE position ids stay valid with no extra
bookkeeping; `tome_merge` returns **explicit position ids**, because survivors are no longer a
grid — callers must use them instead of `get_rope_index`.

Selecting the `pruned` tier is enough to enable FastV: `ModelRunner.fastv_enabled` is true
when either `enable_fastv` is set or the selected tier carries a non-zero `fastv_ratio`.
Setting `enable_fastv=True` explicitly makes `EngineConfig.fastv_ratio` and `fastv_layer`
govern instead. `tests/test_components.py::test_pruned_tier_actually_enables_fastv` pins this
down, because a tier whose ratio is never read is a label that does nothing.

FastV runs **after** the vision tower, in the language model, not inside the tower: pruning
inside the tower would change window membership and spatial coordinates. The implementation
runs layers `0..L` densely (so importance is measured with full context), drops the
low-share image tokens, writes only the survivors to the paged cache, then continues from
`L+1`. It requires the whole prompt to fit in one chunk and refuses otherwise rather than
producing a partially-pruned sequence.

### Offload: static placement, not an LRU

The review's recommendation is followed: layers are placed **statically** (`gpu_layers` stay
resident, the rest stream from pinned host memory) with double-buffered prefetch, because a
decoder visits layers in a fixed order and an LRU cannot reduce total weight traffic — it
only adds jitter. `offload.StaticLayerOffloader.report` prints the effective H2D bandwidth and
the overlap ratio, which is the number that tells you whether streaming is actually paying
off or has become transfer-bound.

## Quality gates

The review separated two kinds of risk that are easy to conflate: **implementation bugs**
(wrong position ids, wrong KV writes, wrong masks, wrong scale indices) and **algorithmic
approximations** (quantisation, merging, pruning, resolution reduction). Without that split,
you cannot tell whether a changed output is expected degradation or a bug.

Three gates, all runnable with `python scripts/quality_gate.py --model <dir> --gate all`
(`--gate` defaults to `A`, so name the ones you want; `--tiny` uses a random fixture instead of
a real checkpoint):

* **Gate A — reference equivalence.** Quantisation and pruning off; compare intermediate
  tensors and final logits against `transformers`. This is what the test suite automates.
* **Gate B — quantisation regression.** Each scheme evaluated separately on a fixed prompt
  set, so a change to the quantiser cannot silently trade accuracy for speed.
* **Gate C — vision-strategy regression.** Each tier evaluated separately; `full`'s
  correctness is not evidence for `half` or `thumbnail`.

### A trap in the reference itself

`Qwen2_5_VLModel.compute_3d_position_ids` sets
`can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and (...)`.
Omit `mm_token_type_ids` when calling HuggingFace and HF does **not** raise: it falls back to
plain 1-D positions and ignores the image grid entirely. A multimodal comparison that forgets
to pass it is therefore measuring this engine against a model that never used MRoPE, and
reports a spurious divergence around `1e-2` that looks exactly like a real bug.

`scripts/quality_gate.py` passes it, and
`tests/test_model_vs_hf.py::test_hf_silently_skips_mrope_without_mm_token_type_ids` asserts
both halves of the trap: that HF's two call forms differ, and that this engine tracks the
MRoPE form.

### Memory safety when running a gate

Loading a model is not free on a unified-memory machine: a large load evicts whatever else is
running, including a served model. Both `scripts/quality_gate.py` and `bench/run_bench.py`
therefore refuse to load weights that exceed a fraction of reclaimable memory (default 50%,
`--memory-margin`/`--force` to override), read `MemAvailable` rather than `MemFree`, and
compute the footprint from the config *before* allocating anything. Neither downloads a model:
`--model` must be a local directory.

## Known limitations

Stated plainly, because the alternative is a user discovering them:

* Prefill is executed **one sequence per chunk**; heterogeneous-length prefills are not
  batched together. Decode *is* batched. Continuous batching (mixing prefill and decode in
  one iteration, admitting requests mid-flight) is implemented; prefill batching is not.
* FastV requires the prompt to fit in a single chunk.
* The `awq_style` scheme needs calibration statistics; without them it silently degrades to
  plain groupwise INT4 (the code says so in a comment rather than pretending).
* No tensor parallelism. CUDA graphs and custom kernels exist but are **opt-in** and
  self-checked (`--cuda-graph`, `--int4-kernel`, `--paged-attn-kernel`); the default path is
  the one every correctness claim above describes.
* The server needs a tokenizer for chat requests; `/v1/completions` with
  `prompt_token_ids` works without one.
* **`float16` is not safe for large images.** With `--dtype float16` the vision tower overflows
  to NaN on a large grid: a 1312x1199 image at the `full` tier (1258 vision tokens) produces
  `NaN` features, so every logit is NaN and the model emits one repeated token (`!`).
  `bfloat16` is finite and correct on the same input. HuggingFace `transformers` fails
  identically in fp16, so this is the checkpoint's precision limit rather than an engine bug.
  The checkpoint declares `torch_dtype: bfloat16`; use it.
* `bench/run_bench.py` and `qwen25vl-lite bench` produce throughput and memory numbers on your
  hardware; the figures quoted above are the ones measured on the RTX 5060 in the design notes.

## Tests

```bash
pip install -e ".[dev]"              # pytest + the pinned transformers
python -m pytest tests/              # 176 tests, CPU-only, no downloads
python -m pytest tests/test_model_vs_hf.py    # the equivalence gate on its own
python -m pytest tests/test_chunked_prefill.py
```

Three files exercise GPU-only paths and skip on a CPU box: `tests/test_int4_kernel.py`,
`tests/test_paged_attn_kernel.py` and `tests/test_cuda_graph.py`.

The suite needs `transformers` for the reference model, and builds a tiny random-weight
Qwen2.5-VL on the fly — no checkpoint download.

Every equivalence claim was measured against **`transformers` 5.16.1**, and the `dev` extra
pins exactly that version. A reference implementation is allowed to change its internals
between releases, and 5.20 did: it renamed a rotary-embedding class, replaced
`apply_multimodal_rotary_pos_emb` with a generic helper, and moved the MRoPE interleave into
the cos/sin tables. So "matches transformers" is a claim about a version, not a moving target.
The two tests that inspect HF internals use instantiated modules rather than private class
names and skip with a stated reason on versions whose convention differs; the end-to-end
equivalence tests run everywhere. Measured on `transformers` 5.18.0: 176 passed, 2 skipped —
the two skips are exactly those HF-internal tests, which run on the pinned 5.16.1.

## License

Apache-2.0. See `LICENSE`.

## Acknowledgements

The design was reviewed externally on 2026-10-15; the full review is archived verbatim in
`docs/chatgpt-review-full.txt` and its conclusions are folded into `docs/design.md`. Where the
review contradicted an earlier decision, the review won, and the change is recorded.
