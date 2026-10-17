# Changelog

All notable changes are recorded here. The format follows Keep a Changelog; versions follow
Semantic Versioning.

## [Unreleased]

### Added

- **An example launcher and documentation drift guards, for publishing the repository.**
  - `scripts/serve.sh` is the 8 GB-card configuration as a runnable script.  Unlike the
    machine-specific notes it was derived from, every value is overridable from the
    environment (`MODEL=... PORT=8000 ./scripts/serve.sh`) and nothing is hard-coded to one
    host.  It carries the two comments that cost real debugging time: use `bfloat16` (fp16
    overflows the vision tower to NaN on a large grid), and set `PYTHONUNBUFFERED=1` (in the
    background stdout is not a tty, so the startup self-checks otherwise only appear at exit
    and the log looks empty).  Verified end to end on a spare port: it loads 824 tensors, both
    kernel self-checks pass, and `/stats` answers 200.
  - `tests/test_docs.py` fails when a subcommand, a vision tier or a documented serve flag
    exists in the code but is missing from either `README.md` or `README.zh-CN.md`.  It checks
    the direction that actually drifts -- "everything the parser registers is documented" --
    and was confirmed to fire by removing a subcommand from a copy of the Chinese README.
  - `.gitignore` now excludes `Qwen2.5-VL-*/`, `*.safetensors` and `.deploy-notes/`.  Without
    those rules a `git add .` would try to commit a 7.1 GB checkpoint whose shards are each
    over GitHub's 100 MB file limit, and would publish private material that happened to sit
    in `.deploy-notes/` (webcam photos used for vision checks, a desktop screenshot, webui
    uploads and request logs).
- **Chinese documentation** (`README.zh-CN.md`). A full walkthrough in Chinese -- supported
  models, deployment, usage, and the acceleration principles -- so the project is readable in
  Chinese as well as English; `README.md` links to it at the top. It carries the same measured
  numbers as the English README, including the correction that the server's `--vision-tier` flag
  does not set the image budget (only the per-request `vision_tier` / `max_vision_tokens` do),
  and the note that the two stale `147` test counts in the English README were updated to `169`.
- **Live camera descriptions** (`qwen25vl-lite camera`). Grabs frames from a V4L2 device with
  OpenCV and posts each one to a running `serve` as a base64 `data:` URL, printing the
  description of the scene.
  - The frames go to the *running server* rather than a second in-process copy of the model:
    the served model already holds ~5.9 GB of an 8 GB card, so a second copy would OOM, and
    reusing it also avoids a ~20 s model reload.
  - Measured on an RTX 5060 with a 640x480 frame at JPEG quality 85 (~44 KB): **~510 ms per
    frame, ~1.9 frames/s** (421 prompt tokens, ~16 completion tokens). That is the honest
    ceiling for this card -- scene-level description, not video rate. `--interval` adds spacing
    on top of it rather than competing with it.
  - The frame buffer is drained (`grab` twice) and `CAP_PROP_BUFFERSIZE` is set to 1. A 30 fps
    camera behind a ~500 ms loop would otherwise hand back frames from hundreds of
    milliseconds ago, which would make the "live" description describe the past.
  - OpenCV is an optional extra (`pip install 'qwen25vl-lite[camera]'`), so the subcommand
    defers the import and says how to install it when it is missing. `--quiet` prints only the
    descriptions, `--json` emits one object per frame, `--save-dir` also writes the frames it
    captured, and `--max-frames` makes a run finite.
  - Verified against a live camera: three frames produced "A bed with a computer monitor on a
    table in the background." then "A cluttered room with a bed, desk, and computer monitor."
    at 521/504/516 ms, and the saved frame was inspected to confirm the description matches
    what the camera actually saw.
  - **Vision tokens are the dominant per-frame cost, and there are two independent levers for
    them.**  Both measured on the same native 640x480 frame, interleaved to cancel drift and
    normalised to a fixed 16-token answer so answer-length variance cannot drive the result:

    | setting | vision tokens | prefill | fps (16-token answer) |
    |---|---|---|---|
    | native 640, full (server default) | 421 | 251 ms | 2.31 |
    | `--width 320` (client-side) | 129 | 133 ms | **3.20** (+38.5%) |
    | `--max-vision-tokens 128` (request field) | 147 | 138 ms | **3.23** (+40.1%) |
    | `--vision-tier thumbnail` (request field) | 264 | 184 ms | 2.66 (+15.3%) |

    Description quality held at every budget: the same frame gave "A person is laying on a bed
    with a phone and a pillow" at 421 tokens and "A person is lying on a bed with a phone in
    their hand" at 129, with no degradation into nonsense.  Below ~130 vision tokens the prefill
    flattens onto a ~130 ms floor (the text prompt plus fixed per-request overhead), so
    downscaling further buys nothing.
  - **The server's `--vision-tier` flag does not set the image pixel budget.**  `serve.sh` passes
    `--vision-tier full`, but `EngineService.prepare()` reads `vision_tier` (and an optional
    `max_vision_tokens`) from the **request body**, defaulting to `full`; the CLI flag only
    reaches `EngineConfig.vision_tier`, which drives FastV pruning.  Restarting the server with
    `--vision-tier thumbnail` therefore changed nothing about the vision token count -- measured
    421 before and after -- and the per-request field is the lever that works.  `camera` now
    exposes both as `--vision-tier` and `--max-vision-tokens`.
- **Live monitoring** (`qwen25vl-lite monitor`, `EngineStats` in `qwen25vl_lite/engine.py`).
  `/stats` only exposed cumulative counters, so the only way to get a decode rate was to
  difference `decode_steps` between two polls -- which averages over everything since startup
  and cannot see a slowdown that has already happened. `EngineStats` now keeps a rolling
  window of the last 64 decode steps and reports `decode_tokens_per_s` and `decode_step_ms`
  (the *median* step, so one preemption does not swing the number) directly from it.
  - Each window entry is `(end_time, seconds, batch)`. The end time is what makes the rate
    *live*: only steps recent enough to describe the present are counted, so an idle engine
    reads `0.0` instead of freezing on the last value it saw. The freshness bound is
    `max(0.5 s, 4 x median step)` -- one step's duration is also the gap to the next, so four
    step-times is generous for "still decoding", and the 0.5 s floor stops a fast model (12 ms
    steps here) from flickering to zero in the instant between two of them. `decode_active`
    and `decode_idle_s` expose the same judgement to a client.
  - The window is deliberately **not** a dataclass field: it is a `deque` and not
    JSON-serialisable, and `stats()` builds the `/stats` payload from the declared fields, so a
    field would break the endpoint. `stats()` walks `fields(st)` and adds the derived values by
    hand; `tests/test_engine.py` asserts the payload round-trips through `json.dumps`.
  - The decode step is timed from just before the forward pass through sampling, so the
    reported rate is the per-token latency a client actually sees -- including the device sync
    the sampler needs -- not just the forward pass.
  - `monitor` polls `/stats` and renders one line: decode tok/s, median step ms,
    running/waiting/swapped, KV blocks used/total, prefix hit rate, how long the engine has
    been idle, GPU utilisation and memory, and a bar of the recent rates. It overwrites a
    single line on a tty and prints one row per sample otherwise (`--rows`); `--count` makes it
    finite, `--url` points it at a non-default port, `--json` emits one object per sample, and
    `--sparkline 0` drops the history column.
  - Polling is cheap enough to be genuinely real-time: a `/stats` round trip on localhost
    measures 0.43 ms while idle and 1.23 ms median / 1.54 ms p95 under load, so the default
    0.5 s interval is set by readability, not cost -- `--interval 0.1` gives 10 Hz for about a
    percent of a core. `nvidia-smi` is the expensive one at ~16 ms per call (~40x a `/stats`
    round trip), so it is refreshed on its own `--gpu-interval` cadence and reused in between.
    Measured against a live `serve` on an RTX 5060 at 4 Hz: 60.6 -> 87.2 tok/s with `running=1`
    and the GPU at 98-99%, then `0.0` with the idle column counting 0.5 s -> 3.6 s once the
    request finished.
- **CUDA-graph decode** (`qwen25vl_lite/graphs.py`, `--cuda-graph`). A decode step is
  **launch-bound, not compute-bound**: measured on an RTX 5060 with Qwen2.5-VL-3B
  INT4/group-128 at a 512-token context, the CPU needs 19.79 ms to enqueue one step while the
  GPU work behind it is 13.60 ms, so the card idles for a third of every token. The step runs
  ~2,100 Python-level operator dispatches and 252 Triton launches; capturing the decoder
  forward into one graph replays all of it with a single launch.
  - Shapes inside a graph are fixed, so a graph is captured per `(batch, block-table width)`.
    The width grows by one block every 16 tokens, and a capture costs a warm-up forward plus
    a verification replay (~34 ms) -- more than the ~4.5 ms a graph saves -- so the width is
    rounded up to a bucket (`--cuda-graph-bucket`, default 16 blocks = 256 tokens) and the
    table tail is padded. Captures then happen every 256 tokens instead of every 16, and the
    measured cost amortises to a small fraction of a millisecond per token. Bucket 16 was the
    best of 4/8/16/32/64/256 on the target card.
  - The captured buffers are zero-filled, not `-1`-filled: the fused paged-attention kernel
    indexes the cache with the block id it reads, so `-1` would address the bytes before the
    tensor. The capture happens before any real id has been written, which is how this
    surfaced as an illegal memory access during capture rather than at run time.
  - Every graph is checked against the eager forward **at capture time** and refused if it
    differs at all (`verify_tolerance=0.0`), because a graph that silently differed would
    still emit fluent tokens. Measured difference: exactly `0.0`; `tests/test_cuda_graph.py`
    asserts token-for-token equality through the whole engine.
  - On its own the graph helps only where the CPU is the limiter. At a 512-token context it
    is **51.3 -> 61.8 tok/s**, but at 2,048 tokens the eager step is already GPU-bound and the
    graph's own scheduling costs ~1 ms (47.9 -> 45.4). Pairing it with the fused attention
    kernel below removes that regime, because the kernel cuts the GPU work that made the
    long-context step GPU-bound.

- **Fused paged decode attention** (`qwen25vl_lite/paged_attn_triton.py`, `--paged-attn-kernel`).
  The reference path gathers every layer's whole context into dense tensors and hands them to
  SDPA with an additive length mask. On a decode step -- one query per sequence -- that is
  badly matched to the hardware: one layer costs 50 us at a 512-token context and 183 us at
  2,048, against a memory bound of 5-22 us, and the gather costs another 0.15 ms / 0.54 ms per
  step. The kernel reads the paged cache directly: it walks the block table, stops at
  `context_lens`, and runs an online softmax, so no dense gather and no mask tensor exist. It
  is flash-decoding -- the context is split across programs and a second small kernel combines
  the partials.
  - On its own it trades GPU time for CPU time and loses at short context (52.1 -> 50.5 at 512
    tokens), because two Triton launches per layer cost more to enqueue than the one SDPA call
    they replace. It wins at long context (47.9 -> 49.7 at 2,048). Combined with the graph --
    which removes the enqueue cost entirely -- it wins everywhere.
  - `can_use()` checks only static metadata. Nothing may read a device value: the check runs
    inside a graph capture, where `context_lens.min()` is an illegal device-to-host copy. A
    zero-length context is handled in the kernel instead.
  - It is decode-only: chunked prefill keeps the reference path, which the kernel does not
    cover. `verify_against_reference` rebuilds the reference exactly as `text.py` builds it
    (gather, length mask, SDPA) rather than calling `attention.paged_attention`, whose decode
    branch passes no mask and would therefore attend to the padded slots this kernel skips.
  - `tests/test_paged_attn_kernel.py` covers 8 shapes (batch 1-4, head_dim 64/128/256,
    block_size 16/32, heterogeneous context lengths, fp16 and bf16) and asserts that padding
    the block table cannot change the result.

- **Combined** (`--int4-kernel --paged-attn-kernel --cuda-graph`, batch 1, 128 generated
  tokens, `bench/run_bench.py`):

  | prompt | decode before | decode after | TTFT before | TTFT after |
  |---|---|---|---|---|
  | 512 | 52.1 tok/s | **73.2 tok/s** | 336 ms | 223 ms |
  | 2,048 | 47.9 tok/s | **67.8 tok/s** | 832 ms | 722 ms |
  | 4,096 | 39.9 tok/s | **64.5 tok/s** | 1,908 ms | 1,791 ms |

  That is +40% / +42% / +62% against the reference INT4 path. The long-context decay flattens
  as well: 512 -> 4,096 tokens cost 1.31x before and 1.14x after, because the gather and the
  padded attention were what grew with the context.

- **A GEMV path for the M == 1 decode shape** (`_int4_groupwise_gemv`). The tiled kernel needs
  ``BM >= 16`` because ``tl.dot`` does, so a decode step computes 16 rows and throws 15 away --
  16x the useful FLOPs on a step that is memory-bound. The GEMV kernel drops ``tl.dot`` for a
  broadcast-multiply-and-reduce over one activation row, and applies the scale and zero once
  per output column instead of once per weight, using
  ``sum_k x_k*(q-z)*s == s*(sum_k x_k*q - z*sum_k x_k)``.
  - Measured over a whole step's 252 projections, interleaved against the tiled kernel to
    cancel laptop clock drift: **8.4 -> 7.3 ms, 186 -> 202 GB/s**, every round. An earlier
    single-shot comparison showed 7.16 ms for the same kernel and 7.70 ms for another, so the
    interleaved form is what the claim rests on.
  - Prefill (M > 1) keeps the tiled kernel, where the tensor cores are the right tool.
    ``verify_against_reference`` now probes row counts 1, 8 and 64 so both kernels are covered
    -- checking only one would leave the other unverified.
  - ``_tile_for`` is unchanged; the GEMV path has no BM at all, so the tile policy it used to
    share is now only the prefill policy.

- **Two other INT4 experiments were run and not adopted.** The step's 252 kernels read
  1.474 GB of quantised weights in 8.4 ms -- 186 GB/s, against the 372 GB/s the unquantised
  `lm_head` gemv reaches on the same card:
  - A tile sweep over `(BM, BN, num_stages, num_warps)` on the *aggregate* of one step's 252
    GEMMs (a per-shape microbenchmark is misleading: a single 12 MB weight stays in L2, so
    its 368 GB/s is L2 bandwidth, not DRAM). The current shape-aware policy won: 7.77 ms
    against 8.04 ms for the best alternative.
  - Transposing scales and zeros from `[N, K/G]` to `[K/G, N]` should have made the
    per-iteration load contiguous instead of a 32-byte-strided 2-byte gather; it measured
    7.70 -> 7.59 ms, i.e. 1.4%, so the sector waste was not the bottleneck.
  - Re-packing the weights group-major (`[K/G, N, G/2]`) makes a tile one contiguous run
    instead of `BN` strided 64-byte segments, and a load-only probe went from 256 to 277 GB/s.
    That is ~8%, and it would cost a second copy of the packed weights (~1.4 GB on an 8 GB
    card) plus a layout the reference dequantiser no longer reads. Not worth it.

- **Opt-in fused INT4 dequantisation + GEMM** (`qwen25vl_lite/int4_triton.py`, `--int4-kernel`).
  The reference path materialises a dense fp16 copy of every quantised weight on every decode
  step; measured on an RTX 5060 with Qwen2.5-VL-3B INT4/group-128, that unpack is ~92% of a
  decode step, so a 2048x11008 projection costs 1.83 ms of which 1.71 ms is dequantisation.
  The Triton kernel reads the packed nibbles once, applies `(q - zero) * scale` in registers
  and feeds `tl.dot` directly. Through the real `QuantLinear.forward`, one decode step's seven
  projections drop from 200.2 ms to 12.0 ms per token across 36 layers (16.7x; a weights-only
  ceiling of 5.0 -> 83.3 tok/s). Prefill is compute-bound and gains only 1.1x.
  - Off by default: it changes the fp16 summation order of every quantised projection, so the
    README's exactness claims describe the reference path unless it is enabled.
  - `can_use()` gates it to int4 groupwise, group in 32/64/128/256, no AWQ rescale, fp16/bf16
    on CUDA; anything else falls back to the reference path unchanged.
  - `--int4-kernel` runs `verify_against_reference` at startup over the first quantised layers
    and refuses the kernel if the worst **relative** error exceeds `2e-2`, rather than falling
    back silently. The criterion is relative because the reference is computed in the same
    dtype: bf16's own rounding noise is ~1e-2 of the output magnitude (8 mantissa bits), so an
    absolute threshold rejected a kernel whose difference was smaller than one bf16 ulp. The
    probe input is now seeded, so the startup decision is reproducible.
  - `tests/test_int4_kernel.py` checks the kernel against the reference path on GPU (and the
    guards on CPU). The kernel's `BK == group_size` choice is what lets scales and zeros load
    as a vector and broadcast instead of a per-element gather; that was worth ~8x on its own.
  - **Decode measured 4.4 -> 51.1 tok/s** on an RTX 5060 with Qwen2.5-VL-3B INT4/group-128
    (`bench/run_bench.py --quant int4 --int4-kernel`, 512-token prompt, 128 generated), and
    prefill 33.0 -> 180.6 tok/s. The tile policy adapts to N: at M=1 a narrow BN=32 is what
    keeps enough CTAs to saturate bandwidth (BN=128 left the 256-row k/v projections with two
    CTAs), which took a layer's seven projections from 383 us to 195 us.

- **Sync-free paged attention.** The paged path read device scalars once per *layer*
  (`.item()` / `.tolist()` in `gather_paged_kv` and the mask builders). At 36 layers that was
  ~180 CUDA-queue drains a decode step, and the profiler showed 9,352 device-to-host copies
  for 64 tokens. Now the engine builds the slot index and the length mask once per step and
  every layer reuses them, and the reference path stays available behind `max_len=None`:
  - `build_paged_slots` / `gather_slots` replace the per-layer Python loop with one device-side
    gather; the result is left non-contiguous because SDPA accepts it (measured the same 12.4 us
    as a contiguous copy, which cost an extra 3.8 us).
  - `build_decode_mask(..., max_len=)` and `build_chunk_mask(..., max_q=, max_ctx=)` build the
    mask on device from bounds the caller already knows.
  - Decode went 33.7 -> 51.1 tok/s from this alone, and TTFT 626 -> 333 ms.
  - `tests/test_paged_fast_path.py` pins the fast path to the reference path's numbers.

- **The CUDA path no longer needs a full-precision copy of the decoder.** `QuantLinear` is built
  with `cache_dequant=False` in `model.py`, so a dequantised weight is transient -- one layer at
  a time. Caching it would hold a second copy of the 2.6 B-parameter decoder (~10.5 GB in fp32,
  ~5.2 GB in fp16), which defeats the point of int4 on an 8 GB card. Two further fixes make the
  int4 path runnable on CUDA at all: the quantised branch of the weight loader now moves its
  packed buffers to `device` like the dense branch (they previously stayed on CPU, and the first
  forward died with `mat2 is on cpu, different from other tensors on cuda`), and
  `quant.dequantize` accumulates in the caller's float dtype when that is fp16/bf16/fp32 instead
  of always routing through float32, which wrote and re-read every layer twice for nothing.

### Fixed

- **Prefix-cache reuse followed the whole prompt, not the prefix.** `_adopt_prefix` passed
  `seq.prefix_key` -- a fingerprint of the *entire* prompt -- as the extra key of every radix
  node, so two requests sharing a system prompt but differing afterwards never matched: the
  cache only ever hit on byte-identical prompts, and a chat template's system prompt bought
  nothing. `prefix_block_keys` now returns one key per whole block, each carrying only the
  images that *start* before that block ends:
  - a block before the first image depends on its tokens alone, so it is shared across
    different images;
  - a block after an image carries that image's quantised features and grid, so it is never
    shared across different pixels (or a different grid, which shifts MRoPE positions).
  - Measured end to end on the real model: a second request sharing a 200-token system prompt
    now reuses 192 tokens (12 full blocks) where it previously reused 0.
  - `RadixPrefixCache.match`/`insert` keep the old single `extra_key` argument working, so
    existing callers and tests are unchanged.
  - `tests/test_prefix_block_keys.py` covers the shared-prefix, pre-image, different-pixel and
    different-grid cases; `test_engine.py::test_requests_sharing_a_prefix_reuse_it` covers it
    through the scheduler.

- **`bench/run_bench.py` could not run an INT4 benchmark at all** -- four defects, each fatal:
  `apply_quantization(model, args.quant, ...)` passed the CLI's `int4` where the internal scheme
  name `int4_groupwise` is required; `plan.gpu_kv_tokens` does not exist (`kv_pool_tokens`);
  `EngineConfig` was built without `dtype`, so the KV cache was allocated in its float32
  default -- twice the planned footprint, which OOMs on the first KV allocation; and
  `cfg.text` does not exist (`cfg.text_config`). The harness now also takes `--int4-kernel`.

- **Image normalisation used the wrong statistics.** `imageproc._to_tensor` normalised with
  `mean = std = 0.5` -- the CLIP `[0, 1]` rescale -- instead of Qwen2.5-VL's channel statistics
  (`0.48145466, 0.4578275, 0.40821073` / `0.26862954, 0.26130258, 0.27577711`). The tower runs
  happily on the wrong input, so the bug is silent: it describes a *different* image. On a test
  image that reads "QWEN-VL TEST" under the correct statistics, the 0.5 rescale produced
  "CAFE YET CAFELES". The values are now read from the checkpoint's `preprocessor_config.json`
  (`imageproc.image_mean_std`, falling back to the Qwen2.5-VL constants when the file or keys
  are absent), and the server passes them into `process_image` instead of relying on defaults.

- **`patchify` emitted patches in plain row-major order, not the HF processor's merge-block
  order.** For patch `(h, w)` the correct index is
  `((h // m) * (W // m) + (w // m)) * m * m + (h % m) * m + (w % m)`. The two orders coincide
  when `grid_w == merge_size` -- which is exactly why a narrow test image never caught it -- and
  differ on a 22x46 grid, where the tower received a scrambled image and still produced fluent
  output for the wrong picture. `patchify` now regroups the grid into `(h2, m, w2, m)` before
  flattening, and rejects a grid that is not a multiple of `merge_size`;
  `test_patchify_matches_a_naive_reference` pins the block-major order and asserts that plain
  row-major differs.

## [0.1.0] - 2026-10-15

First release. The engine is complete and its correctness claims are backed by tests, but no
GPU-measured performance numbers ship with it (see "Not included" below).

### Added

- **Model**: faithful port of the Qwen2.5-VL vision tower and text decoder, plus streaming
  safetensors loading on the `meta` device so construction allocates nothing.
- **MRoPE**: text rotary embeddings with the canonical six-slice interleave, and 2-D vision
  RoPE. Verified against `transformers` with zero tolerance.
- **Quantisation**: weight-only INT8 per-channel, INT4 groupwise (configurable group size),
  and activation-aware INT4 (AWQ-style) with an exponent grid search. Scale-axis consistency
  is checked by `verify_scale_axis`.
- **Paged KV cache**: reference-counted 16-token blocks, optional INT8 storage with a
  per-(token, head) scale so writes stay order-independent under chunked prefill.
- **Radix prefix cache**: whole-block prefix reuse keyed on token ids *plus* a fingerprint of
  the vision embeddings and RoPE configuration.
- **Scheduler**: continuous batching (prefill and decode in one iteration, requests admitted
  mid-flight), chunked prefill, and preemption by swap or by recompute. A stall guard raises
  instead of spinning when no progress is made.
- **Vision budgeting**: resolution ceilings expressed in vision tokens; grid-preserving
  pooling; ToME-style bipartite merging with explicit position ids; FastV-style pruning in the
  language model after the vision tower. Each *tier* applies exactly one lever (a resolution
  ceiling, or FastV); pooling and ToME are library utilities that no tier uses, and the README
  says so — see "Fixed after the gates were run".
- **Memory planner**: computes KV cost per token from the config and reports logical context,
  GPU-resident KV and CPU swap capacity as three separate numbers.
- **Offload**: static CPU/GPU layer placement with double-buffered prefetch and an
  effective-bandwidth / overlap report.
- **Serving**: OpenAI-compatible HTTP API on the standard library only (`/v1/chat/completions`,
  `/v1/completions`, streaming, `/health`, `/stats`, `/v1/models`), plus a CLI with `plan`,
  `probe`, `tiers`, `bench`, `generate`, `serve`.
- **Benchmark harness** (`bench/run_bench.py`): prefill/decode throughput, TTFT, peak memory
  and KV occupancy, with a memory guard that refuses to load a model too large for the
  machine's reclaimable memory.
- **Quality gates** (`scripts/quality_gate.py`): Gate A against `transformers`, Gate B per
  quantisation scheme, Gate C per vision tier. Runnable without a real checkpoint (`--tiny`).
- **Tests**: 111 tests, CPU-only, no checkpoint download. Includes a 20-combination chunked
  prefill equivalence matrix, step-by-step paged-decode parity with HF, batched-vs-sequential
  decode parity, prefix-cache reuse parity, greedy generation parity with
  `transformers.generate`, and a multimodal case with a large enough image to engage windowed
  vision attention.

### Changed after external design review (2026-10-15)

- KV cost per token corrected from an understated 18 KiB to the computed 36 KiB for 3B fp16;
  the 250k-token GPU-resident claim was removed and is now refuted by a test.
- The plain INT8 path is no longer labelled "AWQ"; a real activation-aware scheme was added
  alongside it.
- Offload switched from a layer LRU to static placement with prefetch.
- SDPA backend selection is probed empirically at runtime instead of inferred from the GPU
  architecture.
- FastV moved out of the vision tower to the language model and is off by default.

### Fixed after the gates were run (2026-10-15)

Writing `scripts/quality_gate.py` and running it found four defects that the test suite had
not reached. All four are fixed, and each now has a test:

- **`VisionTier.pool_factor` was dead configuration.** The `half` and `thumbnail` tiers
  advertised grid-preserving pooling, but nothing in the engine or server ever applied it —
  only the resolution ceiling took effect, and the tiers' own arithmetic was incoherent
  (`half` claimed a 2x2 pool *and* a halved ceiling, i.e. ~1/8, while calling itself "half").
  Tiers now apply exactly one lever, `pool_factor` is gone, and `pool_merge`/`tome_merge` are
  documented as unapplied utilities. Gate C now shows `half` at 0.495x and `thumbnail` at
  0.190x of `full`'s vision tokens.
- **The `pruned` tier did not prune.** `TIERS["pruned"].fastv_ratio` was never read, so
  selecting the tier changed nothing. `ModelRunner.fastv_enabled`/`fastv_ratio`/`fastv_layer`
  now resolve from the tier unless `enable_fastv` is set explicitly.
- **Gate A called HuggingFace without `mm_token_type_ids`**, which makes HF silently skip
  MRoPE and fall back to 1-D positions. The gate reported a spurious 4.6e-3 divergence that
  looked like an engine bug. The gate now passes it, Gate A reports exactly `0.0`, and
  `test_hf_silently_skips_mrope_without_mm_token_type_ids` pins the trap down.
- **Gate A's pass criterion was absolute** (`max|diff| < 1e-3`), which is the wrong test for
  logits of order 1-10: a correct float32 result can exceed it. It now uses the same relative
  criterion as the test suite (`atol=1e-4, rtol=1e-3`).

Also: `RequestOutput.metrics` now carries `ttft_s`. The README advertised TTFT but the engine
never recorded it, so the benchmark had nothing to read.

### Fixed after CI (2026-10-15)

The first CI run failed on both Python versions while the suite passed locally. The cause was
an unpinned test dependency, not the engine:

- `dev` required `transformers>=4.41`, so CI resolved the newest release, while the equivalence
  tests import HF internals. `transformers` 5.20 renamed `Qwen2_5_VisionRotaryEmbedding` to
  `Qwen2_5_VLVisionRotaryEmbedding`, dropped `apply_multimodal_rotary_pos_emb` in favour of a
  generic `apply_rotary_pos_emb` with no `mrope_section` argument, and moved the MRoPE
  interleave into the cos/sin tables.

Two changes:

- **`dev` now pins `transformers==5.16.1`**, the version every equivalence claim was actually
  measured against. "Matches transformers" is not a claim that survives an arbitrary upgrade,
  and a reference implementation is allowed to change its internals between releases.
- **The two tests that inspect HF internals no longer import private class names.** They use
  the instantiated modules (`model.visual.rotary_pos_emb`, `model.language_model.rotary_emb`),
  and skip with an explicit reason on versions whose convention differs. The end-to-end
  equivalence tests run on every version and pass on both 5.16.1 and 5.20.

Result: `pytest tests/` → 111 passed on 5.16.1; 111 passed, 2 skipped on 5.20. (Note that
`pyproject.toml` already sets `addopts = "-q"`, so adding `-q` again suppresses the summary
line — the counts above come from `pytest tests/`.)

### Fixed after delivery validation (2026-10-15)

Two independent reviewers went through the codebase and the documentation. Both scored below
the release bar, and every substantive finding is fixed here with a regression test:

**Engine correctness**

- **Symmetric INT4 was numerically broken.** It used `scale = amax/7` with a zero point of
  `0`, so `round(w/scale).clamp(0, 15)` clamped every negative weight to zero and
  dequantisation returned `0` for it. The signed range needs a zero point of `8`; fixed, and
  the zero point is now always stored. `test_symmetric_int4_preserves_negative_weights`,
  `test_symmetric_int4_uses_the_signed_zero_point`.
- **`verify_scale_axis` rejected its own symmetric quantiser** — it hard-coded the asymmetric
  formula `(max-min)/15` for 4-bit regardless of scheme. It now branches on the scheme.
  `test_verify_scale_axis_accepts_the_symmetric_quantiser`.
- **`RadixPrefixCache._evict_if_needed` never evicted anything.** It popped only the victim's
  children, never removed the victim, and never released its blocks, so the tree grew past its
  budget while permanently draining the block pool. It now detaches the victim, drops its whole
  subtree, and frees the cache's block references.
  `test_prefix_cache_eviction_actually_evicts_and_returns_blocks`.
- **Prefix-cache user counts only ever rose.** `release` was the only decrement path and the
  engine never called it, so no node was ever evictable; and `insert` counted the *donating*
  sequence as a user, which left every donated node pinned. `insert` now starts nodes at
  `refcount = 0`, `release` only decrements (it was also double-freeing), and the engine calls
  `release` when a request finishes.
  `test_prefix_cache_release_makes_a_node_evictable`,
  `test_prefix_cache_references_are_released_when_a_request_finishes`.
- **`BlockManager.free_blocks` accepted unknown, out-of-range and duplicate ids**, pushing a
  bogus id onto the free list that `allocate` would then hand out — a slot mapping that writes
  outside the KV tensor. It now raises. `test_block_manager_rejects_bogus_frees`.
- **Sampling penalties were implemented but never applied.** `apply_penalties` had no call site,
  so `repetition_penalty` / `presence_penalty` / `frequency_penalty` silently did nothing;
  `SamplingParams.validate()` was never called; and string `stop` sequences were ignored. All
  three now go through one `_sample` helper, params are validated at `add_request`, and string
  stops are resolved against decoded text in the server (the engine refuses them rather than
  ignoring them). `test_engine_applies_sampling_penalties`,
  `test_engine_rejects_invalid_and_unsupported_sampling_params`.
- **Top-p dropped the token that crossed the threshold** (`cumulative <= top_p`), so the
  retained mass was strictly less than `top_p`. Now the crossing token is kept.
  `test_top_p_keeps_the_token_that_crosses_the_threshold`.
- **Pass 2 of the scheduler could adopt a prefix twice** on a preemption retry, forking blocks
  and double-counting hits. `_adopt_prefix` is now idempotent and adoption happens after the
  "nothing left to prefill" check.
- `EngineConfig` now validates the tier itself, so the `pruned` tier's FastV settings cannot
  bypass validation; an unknown tier raises a message listing the valid names; a fractional
  `second_per_grid_ts` is refused instead of silently truncated.

**Tests that asserted nothing**

- `test_quant_linear_moves_buffers_with_to` wrote `ql.to(...) if False else ql`, so `.to()` was
  never called. `test_preemption_keeps_the_pool_within_budget` asserted
  `kv_blocks_used <= kv_blocks_total`, which holds by construction; it now asserts the pool
  returns to **zero** blocks after every request finishes.

**Documentation**

- The README's test count said 98 (actual 111) and its `pytest tests/ -q` transcript could not
  be produced by that command, because `addopts` already sets `-q` and `-qq` suppresses the
  summary. Corrected.
- The README documented a `--attn-backend` flag that did not exist. The flag is now implemented
  in the CLI and wired to `EngineConfig.attn_backends`.
- "transformers is test-only" was wrong (the server needs it for the tokenizer), and the
  install instructions used the unpinned `.[hf]` extra while the correctness claim is about the
  pinned 5.16.1. Both fixed, and the Tests section now says to install `.[dev]`.
- `python scripts/quality_gate.py --model <dir>` defaults to `--gate A`, so it does not run all
  three gates; the README now says `--gate all`.
- The architecture listing omitted `layers.py` and the supporting scripts; the module map is
  complete, and the quantiser row now names its tests.
- Docstrings in `attention.py` and `rope.py` referenced a `native.py` that does not exist, the
  wrong test filename, and a `vision_position_ids` temporal-axis rule the code does not follow.
  All corrected.
- The server's streaming `/v1/completions` emitted a terminal event typed
  `chat.completion.chunk`; it now matches the stream's `text_completion`.

### Not included

- GPU-measured throughput or memory figures. The development machine has no CUDA-enabled
  PyTorch build; `bench/run_bench.py` exists so these can be produced on the target hardware.
- Custom CUDA kernels. Quantised weights are dequantised in PyTorch; a fused kernel should
  land only with a self-check against the reference path.
- Measured INT4 quality on real weights. Gate B in `scripts/quality_gate.py` is the harness
  for producing it.
