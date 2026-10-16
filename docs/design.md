# Design notes

This document records *why* the engine is shaped the way it is, including the places where an
earlier design was wrong and got corrected. The original draft is preserved verbatim in
`design-draft-v0.1.md` and the full external review in `chatgpt-review-full.txt`;
conclusions drawn from the review are marked **[review 2026-10-15]**.

## Target

RTX 5060, 8 GB VRAM, Blackwell (sm_120), 96 GB host RAM, Qwen2.5-VL-3B.

The binding constraint is not "can the weights fit" — 3B in bf16 is 7.5 GB and does not fit,
but INT4 brings it to ~3.5 GB and it does. The binding constraint is **weights + KV + vision
activations + runtime overhead simultaneously**, and the first version of this design
underestimated all three of the latter.

## Module map

| module | responsibility | verified by |
|---|---|---|
| `config` | parse `config.json`, count parameters | `test_components.py` |
| `rope` | MRoPE + vision 2-D RoPE + `get_rope_index` | `test_mrope_*`, zero tolerance |
| `quant` | INT8 / INT4 groupwise / activation-aware INT4 | `test_quantisation_error_bounds`, `test_awq_style_*` |
| `attention` | runtime SDPA probing, paged attention, masks | `test_components.py::test_paged_attention_*` |
| `imageproc` | smart_resize, patchification | `test_patchify_matches_a_naive_reference` |
| `vision` | vision tower | `test_vision_tower_matches_hf` |
| `text` | decoder with dense/paged/chunked attention | `test_chunked_prefill.py` |
| `model` | assembly + streaming loader | `test_model_vs_hf.py` |
| `cache` | block manager, paged KV, radix prefix cache | `test_components.py` |
| `vision_budget` | resolution budget, pooling, ToME, FastV | `test_components.py` |
| `planner` | memory planning | `test_components.py::test_plan_*` |
| `offload` | static placement + prefetch | `test_components.py::test_offload_*`; transfers must be measured on device |
| `engine` | scheduling | `test_engine.py` |

## Decisions that changed after the external review

### 1. The KV budget was wrong, and the claim built on it was wrong

The first draft said "18 KB/token" and promised a 250k-token KV pool on 8 GB. Both were
wrong. The correct arithmetic is:

```
2 (K and V) x 36 layers x 2 kv_heads x 128 head_dim x 2 bytes = 36,864 B = 36 KiB/token
```

so 250k fp16 tokens is 8.58 GiB — more than the whole card, before weights. **[review
2026-10-15]**

Consequences now encoded in the code:

* `planner.kv_bytes_per_token` computes the number from the config and exposes the arithmetic
  in `KVBytes.describe()` so it can be audited;
* the planner reports **three separate numbers** — logical context, GPU-resident KV, CPU swap
  capacity — and never merges them;
* `tests/test_components.py::test_250k_token_claim_is_refuted_by_the_arithmetic` pins the
  refutation in code, so the claim cannot creep back into a README.

### 2. "AWQ" was a mislabel

The first draft called plain INT8-per-channel-plus-dynamic-INT8-activations "AWQ-style". It is
not AWQ: AWQ is activation-*aware weight* quantisation, and a per-output-channel scale has
nothing to do with activation magnitudes. **[review 2026-10-15]**

Now there are three distinct schemes, kept separate so a regression can be attributed:

* `int8_per_channel` — symmetric INT8, one scale per output channel. A baseline, not AWQ.
* `int4_groupwise` — asymmetric INT4, `group_size` weights per scale and zero point.
* `awq_style` — rescales weight columns by measured activation magnitude, exponent chosen by
  grid search on reconstruction error. This is the actual AWQ idea. Without calibration data
  it degrades to plain groupwise INT4 and says so in the code.

### 3. Fused QKV quantisation needed no special handling

The first draft worried that a concatenated `q|k|v` matrix would need per-submatrix scales.
It does not: both schemes group along the output channel (or along K with a fixed group size),
so the scale array simply follows the concatenated row order. What *does* need checking is
that the dequantiser indexes scales with the same axis the quantiser used — that is
`quant.verify_scale_axis`, exercised in tests. **[review 2026-10-15]**

### 4. SDPA backend selection cannot be inferred from the SM version

The first draft asserted that sm_120 would fall back to the math kernel for non-128-multiple
head dimensions. SM version alone does not determine the backend; dtype, shapes, mask, strides
and the PyTorch/CUDA build all matter, and torch's context manager can silently ignore a
backend on some devices. **[review 2026-10-15]**

So the engine **probes empirically** (`attention.probe_backends`): it runs a real SDPA call
inside each backend's context manager and records the outcome. On CPU, the fused backends are
reported as "not applicable on this device" rather than "available", because torch does not
raise when they are requested there. The chosen list is exposed on `/health`, and `cli.py bench` prints it
before running; the server itself logs only its bind address.

### 5. Offload: static placement, not an LRU

The first draft proposed a layer LRU. A decoder visits layers in a fixed order, so an LRU
cannot reduce the total weight traffic — it only adds bookkeeping and jitter. **[review
2026-10-15]**

Now: static placement (the first `gpu_layers` stay resident, the rest stream from pinned host
memory) with double-buffered prefetch, plus a report of effective H2D bandwidth and overlap
ratio so you can tell whether streaming is paying off or has become transfer-bound.

### 6. FastV is not a plug-in optimisation

The first draft listed FastV as a safe 3-tier knob. It is not: Qwen2.5-VL's tower mixes window
attention with global attention at four block indexes, so pruning inside the tower would
change window membership and spatial coordinates. **[review 2026-10-15]**

Now FastV runs **in the language model, after the vision tower**, off by default, and requires
the prompt to fit in one chunk. The tiers are documented as strategies with expected costs
rather than quality promises, and Gate C exists so each tier gets its own quality report.

## Risk register (current state)

| # | risk | verdict after review | mitigation in code |
|---|---|---|---|
| R1 | SDPA falls back to math on sm_120 | partially wrong — must be measured, not inferred | `probe_backends` + `/health` + `--attn-backend` |
| R2 | MRoPE interleave diverges silently | correct, highest priority | zero-tolerance test vs HF; the `mrope_section * 2` list-repetition trap is pinned by a test |
| R3 | fused QKV scale grouping | partially wrong — grouping is fine, axis consistency is what matters | `verify_scale_axis` + tests |
| R4 | chunked prefill + paged KV alignment | correct | 20-combination equivalence matrix vs HF |
| R5 | 8 GB budget optimistic | completely correct | planner computes reserves; three numbers reported separately |

## Quality gates

Two different kinds of risk must not be conflated:

* **implementation errors** — position ids, KV writes, masks, scale indices, prefix reuse.
  These are bugs and must be eliminated by correctness tests.
* **algorithmic approximations** — INT4 error, token merging, FastV pruning, resolution
  reduction. These change quality by design and must be measured per configuration.

Hence three gates (`scripts/quality_gate.py`):

* **A**: quantisation and pruning off, compare against HF. Automatable, and automated here.
* **B**: each quantisation scheme evaluated separately on a fixed prompt set.
* **C**: each vision tier evaluated separately; `full`'s result is not evidence for `half`.

## What is deliberately *not* implemented

* **Fused dequantisation kernels.** The review's ordering argument is followed: prove the
  reference path first. A fused kernel that is subtly wrong produces plausible output, and
  this repository has no GPU on which to verify one.
* **Prefix-cache eviction beyond LRU-over-nodes.** The radix tree evicts least-recently-used
  leaves when it exceeds `max_nodes`; anything smarter needs workload data.
* **Speculative decoding, CUDA graphs, tensor parallelism.** Out of scope for one 8 GB card.

## Test strategy

Every numerical claim is a comparison, not a shape assertion. A wrong MRoPE interleave or a
mis-ordered patch matrix still produces correctly shaped tensors — shape checks would pass and
the model would answer nonsense. So the suite compares against `transformers` on a tiny
random-weight model: exact equality where the computation is identical, `1e-4` where a
different kernel is involved.

The tiny model is built in-process (`tests/conftest.py`), so CI needs no checkpoint download.

## What the gates found (2026-10-15)

Writing `scripts/quality_gate.py` was not paperwork. Running it surfaced four defects the
test suite had not reached, which is the argument for having a gate that is *separate* from
the tests:

1. **Dead configuration.** `VisionTier.pool_factor` was read nowhere in the engine or server.
   The `half`/`thumbnail` tiers advertised grid-preserving pooling, but only the resolution
   ceiling ever took effect. Worse, the tiers' arithmetic did not add up: `half` claimed a
   2x2 pool *and* a halved ceiling — roughly 1/8, not half. The fix was to give each tier
   exactly one lever, delete the unused field, and state in the README that `pool_merge` and
   `tome_merge` exist but are not applied by any tier. A tier whose name does not match its
   mechanism is a claim nobody can verify.

2. **A tier that did nothing.** `TIERS["pruned"].fastv_ratio` was never read, so selecting
   the tier changed no behaviour. FastV is now enabled by the tier unless `enable_fastv` is
   set explicitly, and a test pins the precedence.

3. **The reference was being called wrongly.** HF's `compute_3d_position_ids` requires
   `mm_token_type_ids`; without it HF falls back to 1-D positions instead of raising. Gate A
   was therefore comparing against a model that never used MRoPE, and reported a 4.6e-3 gap
   that reads exactly like a real bug. The lesson is not "the engine was wrong" but "a
   comparison is only as good as the reference call": an equivalence gate must pin down how
   the reference is invoked, not just what it returns.

4. **A wrong pass criterion.** Gate A used `max|diff| < 1e-3` while the suite used
   `allclose(atol=1e-4, rtol=1e-3)`. For logits of order 1-10 the relative term dominates, so
   the gate was stricter than the property it meant to test. It now uses the suite's
   criterion.

All four are fixed, and each has a regression test — including
`test_multimodal_logits_match_hf_with_windowing_engaged`, which uses a 300x420 image (11x15
merge units, 12 vision windows) because the session fixture's 56x84 image fits inside a single
window and never exercised `get_window_index`'s multi-window path at all.

The gate-driven fixes are a reminder of the ordering that matters: an engine can be
numerically exact and still be wrong about *what it does*. Exactness tests cannot catch a
misnamed tier or an unread config field; only a gate that exercises the feature end to end can.
