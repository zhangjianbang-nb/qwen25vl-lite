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
- **Tests**: 135 tests, CPU-only, no checkpoint download. Includes a 20-combination chunked
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

Also: `RequestOutput.metrics` now carries `ttft_s`. The benchmark asked for TTFT but the
engine never recorded it, so it had nothing to read.

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

Result: `pytest tests/` → 135 passed on 5.16.1; under 5.20 it is 133 passed and 2 skipped (135 collected). (Note that
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
  bypass validation, and an unknown tier raises a message listing the valid names. Separately,
  `rope.py` refuses a fractional `second_per_grid_ts` instead of silently truncating it (the
  check lives with the position-id arithmetic, not in the config).

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

### Fixed after the third validation round (2026-10-15)

Two independent reviewers were run against the published tree; both had to name at least ten
defects. What they found, and what changed:

- **Prefix-cache reference leak (blocker).** `RadixPrefixCache.match` raises the user count on
  every node it walks, but a sequence keeps one token back for prefill, so a prompt whose
  length is an exact multiple of `block_size` adopts one node fewer than it matched. The
  surplus node stayed pinned forever — never evictable, its block never returned. `_adopt_prefix`
  now releases the nodes it discards, and the aligned case has its own regression test.
- **FastV double normalisation.** `TextModel.forward` applied the final RMSNorm whenever the
  layer slice reached the end of the stack, so a prune layer equal to the last layer normalised
  the hidden states twice. The norm is now applied only by a full-stack call
  (`stop_layer is None`).
- **INT4 `axis=1` dequantised transposed.** The quantiser transposes before grouping along K;
  the dequantiser never undid it, returning a `[K, N]` reconstruction for an `[N, K]` weight.
  Fixed, and pinned by a test.
- **`plan_memory` counted offloaded layers as resident.** `gpu_bytes` added the whole
  `weights_bytes` even when layers had been placed in host memory, so the reported footprint
  could exceed the budget the plan was built from. The planner now separates the always-resident
  weights (vision tower, embeddings, head) from the decoder linears, reports `gpu_weights_bytes`,
  and refuses to claim a layer fits when the resident weights alone do not.
- **`EngineConfig.kv_dtype` did nothing.** The field was never read. It is now the single source
  of truth for KV storage: `None` means "the compute dtype", `int8` selects the quantised
  layout, and the float names select the storage width.
- **A prune layer beyond the model depth failed mid-request.** `ModelRunner` now refuses such a
  config at construction and names the depth, instead of raising "expected attention
  probabilities" from inside a forward pass.
- **FastV rewrote the caller's prompt.** The response reported the pruned token list, so
  `prompt_tokens` disagreed with what was submitted. `SequenceState` keeps the submitted ids and
  the response reports those; `vision_tokens_total` now counts the image tokens that were
  actually prefilled rather than the pre-prune count.
- **`bench/run_bench.py --quant int4` crashed.** The flag value was passed straight to
  `apply_quantization`, which expects a scheme name; the documented example command died on
  "unknown scheme 'int4'". The mapping now lives in `_SCHEMES`. Running the script end to end
  (rather than reading it) surfaced four more defects on the same path: it read
  `plan.gpu_kv_tokens` and `cfg.text`, neither of which exists; it allocated the planner's whole
  *capacity* figure as blocks, which on a small model is millions of blocks and gets the process
  OOM-killed; it built the engine without the dtype the weights were loaded at, so every
  attention call failed on a dtype mismatch; and its pseudo-prompt could contain the image
  placeholder, which the engine correctly refuses without pixels. All four are fixed, and the
  pool is now capped at what the run can actually use.
- **The CLI defaulted `--kv-dtype` to `float16` while `--dtype` defaulted to `float32`**, so
  `qwen25vl-lite generate` failed on a dtype mismatch out of the box. `--kv-dtype` now defaults
  to `None`, meaning "the compute dtype".
- **`BlockManager.fork` was laxer than `free_blocks`.** It invented a refcount entry for an
  unknown block id, hiding exactly the ownership bug `free_blocks` was hardened against. It is
  strict now.
- **`PagedKVCache.copy_blocks` claimed a use it does not have.** The docstring said preemption
  copied blocks; preemption actually swaps them. The docstring says what the method is for.
- **`/v1/completions` returned an empty string with no tokenizer.** The generated ids are now
  returned in `choices[0]["token_ids"]` (an extension field), so the endpoint is usable without
  a tokenizer, and the README says so instead of claiming plain text works.
- **Two 404 responses used a bare string** where every other error path returns the OpenAI error
  object. Both are objects now.
- **Documentation claims that the code did not support**: `offload` was described as wired into
  the engine (it is a library the engine never instantiates — `plan_memory` reports the
  arithmetic, and `report` is a dataclass, so `report.summary()` is what prints); `README`
  attributed the tiny fixture to the test suite (the suite builds its own, in `tests/conftest.py`);
  `CHANGELOG` attributed the `second_per_grid_ts` check to `EngineConfig` (it lives in `rope.py`)
  and claimed the README advertised TTFT (it did not); `design.md` said the backend list is
  printed at startup (it is exposed on `/health`, and only `bench` prints it).
- **Tests that asserted nothing meaningful** were strengthened: the "three context numbers" test
  now checks they are three *distinct* quantities rather than three labels, and the planner's
  offload branch — previously uncovered — has its own test.
- `offload` gained the tests its docstring claimed, `vision_budget.apply_fastv` is now the single
  pruning implementation (the engine had a second, untested copy), and FastV's own regression
  test observes pruning through the statistic instead of the rewritten prompt.

### Fixed after the fourth validation round (2026-10-15)

The re-scoring pass (70/100 and 77/100) confirmed the previous fixes but found more, including
two defects in FastV itself that the earlier rounds had not touched:

- **FastV's dense pass was not causal.** `attention_with_probs` applied a causal mask only when
  `k > q`; the FastV stage runs a dense full-sequence pass (`q_len == k_len`) with no explicit
  mask, so it attended to future tokens — the importances it measured and the hidden states it
  forwarded were both wrong. A decoder is causal by default now, and a test compares it against
  SDPA's causal path.
- **FastV did not recompute `rope_delta` after pruning.** It is `max(position) + 1 - seq_len`,
  and pruning changes both terms; the stale value made the first decoded token fall *behind* the
  prompt's own positions, corrupting every generated token's rotation.
- **FastV and the prefix cache disagreed about what the key described.** The cache key was built
  from the submitted prompt but the node stored the pruned sequence, so a FastV request could
  never hit — and if it had adopted a prefix, it would have skipped pruning entirely, making the
  same prompt produce different output depending on cache state. FastV requests now stay out of
  the prefix cache, and that is stated rather than left implicit.
- **A prompt too long for one step with chunked prefill disabled deadlocked** (the stall guard
  reported "no progress"). It is now refused with the real reason named.
- **`run_until_complete` re-reported earlier requests**, because `self.finished` accumulates.
- **The prefix fingerprint used `hash(str) & 0xFFFFFFFF`** — salted per process and only 32 bits,
  so keys differed across runs and could collide, reusing KV computed for a different image. It
  is a blake2b digest now.
- **`QuantizedTensor.shape` held the transposed dims for `axis=1`**, so `out_features` and
  `in_features` were swapped even though `dequantize` was right.
- **The planner understated an int8 KV cache** by omitting the per-(token, kv_head) fp16 scales.
- **`kv_dtype` naming a different float width than the compute dtype** was accepted and then
  failed inside the first attention call; it is refused at construction, with the reason.
- **Streaming `/v1/completions` chunks omitted `token_ids`**, so the README's "always" was false
  and a client without a tokenizer got nothing in streaming mode.
- Documentation that still overstated the code: `offload`'s docstring said `report` prints (it is
  a dataclass), `attention.py` said the backend is written to logs (nothing is logged), README
  said PCIe overlap is probed at runtime (only the SDPA backend is; the offload class is never
  instantiated), README's serve example passed `--kv-dtype float16` with the default float32
  compute dtype (which cannot work), and `design.md` described cache eviction as leaf-only (it
  evicts any subtree with no live user).
- The test count in the previous entry was itself wrong; the suite is 135 tests.

### Not included

- GPU-measured throughput or memory figures. The development machine has no CUDA-enabled
  PyTorch build; `bench/run_bench.py` exists so these can be produced on the target hardware.
- Custom CUDA kernels. Quantised weights are dequantised in PyTorch; a fused kernel should
  land only with a self-check against the reference path.
- Measured INT4 quality on real weights. Gate B in `scripts/quality_gate.py` is the harness
  for producing it.
