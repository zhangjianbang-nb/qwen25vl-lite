# qwen25vl-lite

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
# serve keeps KV at --dtype; add --kv-dtype int8 for a quantised cache
qwen25vl-lite serve --model ./Qwen2.5-VL-3B-Instruct --device cuda --dtype float16 \
                    --weight-dtype int4 --num-blocks 4000
```

---

## What is actually verified

| Claim | How it is checked | Status |
|---|---|---|
| MRoPE matches HF **bit for bit** | `tests/test_model_vs_hf.py::test_mrope_cos_sin_and_apply_match_hf` uses `torch.equal` (zero tolerance) | passing on 5.16.1; skips on 5.20, where HF moved the interleave into the cos/sin tables |
| Vision tower matches HF | `test_vision_tower_matches_hf`, `atol=1e-5` | passing |
| End-to-end logits match HF (text and multimodal) | `test_full_logits_match_hf`, `test_text_only_logits_match_hf` | passing |
| Multimodal logits match HF when windowed vision attention engages | `test_multimodal_logits_match_hf_with_windowing_engaged` (a 22x30 patch grid = 12 windows) | passing |
| Chunked prefill == single-pass prefill | 20 combinations of chunk size {16, 8, 5, 3} x length {15, 16, 17, 33, 64}, compared to HF | passing |
| Paged decode == HF's own incremental decoding, step by step | `test_paged_decode_matches_hf_incremental` | passing |
| Batched decode == sequential decode (no KV-padding leakage) | `test_decode_batch_matches_independent_single_sequences` | passing |
| Prefix-cache reuse == recomputing the prefix | `test_prefix_cache_reuses_and_matches_fresh_compute` | passing |
| Greedy engine output == `transformers.generate` | `test_greedy_generation_matches_hf_generate` | passing |
| Preemption keeps the block pool in budget and every request still finishes | `test_preemption_keeps_the_pool_within_budget` (+ a recompute variant), which also asserts the pool returns to empty | passing |
| Quantiser is self-consistent (scale axis, INT4 packing, AWQ objective) | `test_quantisation_error_bounds`, `test_int4_pack_unpack_roundtrip`, `test_awq_style_beats_plain_groupwise_when_channels_differ`, `test_symmetric_int4_preserves_negative_weights` | passing |
| The prefix cache evicts and returns blocks instead of leaking | `test_prefix_cache_eviction_actually_evicts_and_returns_blocks` | passing |
| The block pool rejects unknown and double frees | `test_block_manager_rejects_bogus_frees` | passing |
| Sampling penalties and validation actually apply | `test_engine_applies_sampling_penalties`, `test_engine_rejects_invalid_and_unsupported_sampling_params` | passing |

```
$ python -m pytest tests/
135 passed in ~10s
```

Everything runs on CPU with tiny random weights and **no model download**, so the suite is
usable in CI. (`pyproject.toml` already sets `addopts = "-q"`, so pass `-q` again only if you
want the summary line suppressed.)

## What is *not* verified — read this before quoting a number

Being explicit here is part of the design, not a disclaimer bolted on at the end:

* **No throughput or memory number in this README was measured on a GPU.** The author's
  development machine has no CUDA-enabled PyTorch build. `bench/run_bench.py` and
  `qwen25vl-lite bench` exist so *you* can produce those numbers on your card; nothing is
  claimed in advance.
* **INT4 quality on real Qwen2.5-VL weights is not measured.** The quantiser is unit-tested
  for reconstruction error and the AWQ objective is tested to be non-worse than plain
  groupwise on a synthetic case, but "how much accuracy does INT4 cost on DocVQA" is a
  question this repository answers only through `scripts/quality_gate.py`, which you run.
* **No custom CUDA kernels.** Quantised weights are dequantised in PyTorch before the GEMM.
  This is deliberate: the external design review (2026-10-15) warned against fusing
  dequantisation into a kernel *before* the reference path was proven, because a wrong fused
  kernel produces plausible-looking output. A fused kernel is the obvious next step, and it
  should land with a self-check against the reference path.
* **The CPU fallback is the tested path.** The SDPA backend actually in use is *probed at
  runtime* (and surfaced on `/health`), never assumed. PCIe transfer overlap is **not** probed:
  `offload.StaticLayerOffloader` measures it, but the engine never instantiates that class, so
  nothing in a runnable path reports it.
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
  attention.py      runtime SDPA backend probing, paged attention reference, masks
  imageproc.py      smart_resize + patchification matching the HF processor
  vision.py         the vision tower (window attention + PatchMerger)
  text.py           the decoder (GQA, SwiGLU, MRoPE, paged/chunked/dense attention)
  layers.py         shared RMSNorm (plain formulation, float32 compute)
  model.py          assembly, streaming safetensors loading, forward pass
  cache.py          block manager, paged KV cache, radix prefix cache
  vision_budget.py  resolution budgets, pooling, ToME merge, FastV selection
  planner.py        device-memory planning (three context numbers kept separate)
  offload.py        static CPU/GPU layer placement with double-buffered prefetch
  sampling.py       penalties, top-k/top-p/min-p, per-sequence RNG
  engine.py         continuous batching, chunked prefill, preemption, FastV prefill
  server.py         OpenAI-compatible HTTP API (stdlib only)
  cli.py            `plan`, `probe`, `tiers`, `bench`, `generate`, `serve`
```

Supporting scripts: `scripts/demo.py` (end-to-end walkthrough), `scripts/make_tiny_model.py`
(the random fixture used by `--tiny` in the gates and by the demo; the test suite builds its
own fixture in-process, see `tests/conftest.py`), `scripts/quality_gate.py` (Gates A/B/C),
`bench/run_bench.py` (throughput and memory on your hardware).

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

The ratios are the ones `scripts/quality_gate.py --gate C --tiers full,half,thumbnail` prints on
a 900x1200 image; they are a property of the ceilings, not of any particular model. `pruned`
keeps all tokens through the tower and prunes inside the language model, so its *prompt* is
unchanged until pruning and its ratio is 1.00 at the vision-token stage — which is why Gate C's
default tier list does not include it, and why the `1.00` above is stated rather than measured
there. Pass `--tiers full,pruned` to compare it directly.

Two levers are wired into tiers: a **resolution ceiling** (`full` / `half` / `thumbnail`) and
**FastV pruning** (`pruned`). Resolution is the honest default: fewer vision tokens is exactly
what prefill time and KV pressure are made of, and it needs no change to the checkpoint's
feature geometry. FastV is the riskier one, which is why it is a separate tier rather than a
flag applied to the others.

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

The review's recommendation is followed in the *design*: layers are placed **statically**
(`gpu_layers` stay resident, the rest stream from pinned host memory) with double-buffered
prefetch, because a decoder visits layers in a fixed order and an LRU cannot reduce total
weight traffic — it only adds jitter.

**This is a library, not a wired-in feature.** `offload.StaticLayerOffloader` implements the
placement arithmetic (`placement_for`), the prefetch/materialise bookkeeping, and the report;
its scheduling logic is unit-tested in `tests/test_components.py`. But nothing in
`engine.py`, `cli.py` or `bench/run_bench.py` instantiates it, so loading a model through the
engine keeps **every** layer resident. `plan_memory` still reports how many layers *would*
fit and how many bytes would then be resident (`gpu_bytes` counts only those), which is the
number the planner exists to produce. `report` is an `OffloadReport` dataclass —
`report.summary()` is what prints the effective H2D bandwidth and the overlap ratio.

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
* No CUDA graphs, no custom kernels, no tensor parallelism.
* The server needs a tokenizer to produce text. `/v1/completions` accepts `prompt_token_ids`
  and returns the generated ids in `choices[0]["token_ids"]` (an extension field, present in
  both the JSON response and every streaming chunk); with no tokenizer the `text` field is an
  empty string, so a client without a tokenizer should read the ids.
* `bench/` and the CLI can produce numbers, but this repository ships none that were measured
  on a GPU.

## Tests

```bash
pip install -e ".[dev]"              # pytest + the pinned transformers
python -m pytest tests/              # 135 tests, CPU-only, no downloads
python -m pytest tests/test_model_vs_hf.py    # the equivalence gate on its own
python -m pytest tests/test_chunked_prefill.py
```

The suite needs `transformers` for the reference model, and builds a tiny random-weight
Qwen2.5-VL on the fly — no checkpoint download.

Every equivalence claim was measured against **`transformers` 5.16.1**, and the `dev` extra
pins exactly that version. A reference implementation is allowed to change its internals
between releases, and 5.20 did: it renamed a rotary-embedding class, replaced
`apply_multimodal_rotary_pos_emb` with a generic helper, and moved the MRoPE interleave into
the cos/sin tables. So "matches transformers" is a claim about a version, not a moving target.
The two tests that inspect HF internals use instantiated modules rather than private class
names and skip with a stated reason on other versions; the end-to-end equivalence tests run
everywhere. On the pinned 5.16.1 the suite is 135 passed; under 5.20 it is 133 passed and
2 skipped (135 collected). That second number was produced by running the suite in a scratch virtualenv with
`pip install -U transformers`; it is not reproducible from the pinned `dev` extra by design,
because the pin exists to keep CI on the verified version.

## License

Apache-2.0. See `LICENSE`.

## Acknowledgements

The design was reviewed externally on 2026-10-15; the full review is archived verbatim in
`docs/chatgpt-review-full.txt` and its conclusions are folded into `docs/design.md`. Where the
review contradicted an earlier decision, the review won, and the change is recorded.
