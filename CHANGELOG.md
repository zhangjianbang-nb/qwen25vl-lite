# Changelog

All notable changes are recorded here. The format follows Keep a Changelog; versions follow
Semantic Versioning.

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
  says so — see "Vision reduction: one lever per tier" there.
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
- **Tests**: 116 tests, CPU-only, no checkpoint download. Includes a 20-combination chunked
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

Result: `pytest tests/` → 116 passed on 5.16.1; 116 passed, 2 skipped on 5.20. (Note that
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
